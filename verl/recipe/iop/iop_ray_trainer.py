# Copyright 2026 Individual Contributor: d fei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IOP implemented through existing rollout-postprocessing and worker loss hooks."""

import asyncio
from functools import partial
from math import isfinite, lcm
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from recipe.iop.algorithms import align_tokens, repair_messages, repair_scores, select_repair
from recipe.iop.losses import iop_loss
from recipe.iop.sampler import DeferredSampler, IndexedDataset
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_extern_object
from verl.utils.py_functional import rename_dict
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.utils.padding import left_right_2_no_padding


def _objects(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


class IOPRayPPOTrainer(RayPPOTrainer):
    rollout_postprocessing = True

    def __init__(self, config, tokenizer, *args, **kwargs):
        self._validate_config(config, kwargs.get("processor"))
        self.iop = config.iop
        self.audit = (
            load_extern_object(self.iop.audit_path, self.iop.audit_name) if self.iop.audit_path is not None else None
        )
        super().__init__(config, tokenizer, *args, **kwargs)

    @staticmethod
    def _validate_config(config, processor=None):
        actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
        if config.algorithm.adv_estimator != "iop" or rollout.n < 2:
            raise ValueError("IOP requires adv_estimator=iop and rollout.n>=2")
        if actor.ppo_epochs != 1 or actor.policy_loss.loss_mode != "gspo":
            raise ValueError("IOP uses one joint update with the recipe's GSPO callback")
        if actor.strategy not in ("fsdp", "fsdp2", "megatron"):
            raise ValueError("IOP currently supports FSDP/Megatron static data parallel training")
        if config.data.dataloader_num_workers != 0:
            raise ValueError("IOP deferred ordering requires data.dataloader_num_workers=0")
        if type(config.iop.repair_n) is not int or config.iop.repair_n < 2:
            raise ValueError("iop.repair_n must be an integer >=2")
        for key in ("truncation_k", "generation_concurrency"):
            if type(config.iop[key]) is not int or config.iop[key] < 1:
                raise ValueError(f"iop.{key} must be a positive integer")
        if any(not isfinite(config.iop[k]) or config.iop[k] < 0 for k in ("edit_weight", "repair_weight")):
            raise ValueError("IOP objective weights must be finite and nonnegative")
        if (
            processor is not None
            or rollout.name != "vllm"
            or rollout.multi_turn.enable
            or rollout.agent.default_agent_loop != "single_turn_agent"
        ):
            raise ValueError("IOP currently supports text-only single_turn_agent vLLM")
        if actor.entropy_coeff != 0 or config.algorithm.use_kl_in_reward:
            raise ValueError("IOP uses its own joint KL callback and no entropy/reward-KL term")
        if any(
            OmegaConf.select(config, key, default=False)
            for key in (
                "critic.enable",
                "distillation.enabled",
                "reward.reward_model.enable",
                "actor_rollout_ref.rollout.skip.enable",
                "actor_rollout_ref.actor.use_prefix_grouper",
                "actor_rollout_ref.actor.megatron.dynamic_context_parallel",
            )
        ):
            raise ValueError("IOP does not combine critic, distillation, learned RM, cached rollout or dynamic CP")
        model = config.actor_rollout_ref.model
        if model.get("use_remove_padding", False):
            raise ValueError("IOP requires padded model input to isolate hybrid recurrent states")
        if model.get("lora_rank", 0) or model.get("lora", {}).get("rank", 0):
            raise ValueError("IOP currently uses full shared model weights, without LoRA")
        for role in ("actor", "ref"):
            for path in ("ulysses_sequence_parallel_size", "fsdp_config.ulysses_sequence_parallel_size"):
                if OmegaConf.select(config, f"actor_rollout_ref.{role}.{path}", default=1) != 1:
                    raise ValueError("IOP requires Ulysses SP1")
        if any(
            settings.get(key) is not None
            for settings, key in (
                (actor, "ppo_micro_batch_size"),
                (rollout, "log_prob_micro_batch_size"),
                (config.actor_rollout_ref.ref, "log_prob_micro_batch_size"),
            )
        ):
            raise ValueError("IOP requires per-GPU rather than deprecated global microbatch sizes")
        correction = config.algorithm.get("rollout_correction")
        if correction and any(correction.get(k) for k in ("bypass_mode", "rollout_is", "rollout_rs")):
            raise ValueError("IOP recomputes old probabilities without rollout IS/RS")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if train_dataset is None:
            from verl.trainer.main_ppo import create_rl_dataset

            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        train_dataset = IndexedDataset(train_dataset)
        self.iop_sampler = DeferredSampler(
            train_dataset, self.config.data.shuffle, self.config.data.get("seed", 42) or 42
        )
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, self.iop_sampler)

    def init_workers(self):
        super().init_workers()
        actor = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        self.actor_rollout_wg.set_loss_fn(partial(iop_loss, config=actor))
        self.iop_client = self.llm_server_manager.get_client()

    @staticmethod
    def _tokens(batch, row):
        return batch.batch["responses"][row][batch.batch["response_mask"][row].bool()].tolist()

    @staticmethod
    def _prompt(batch, row):
        width = batch.batch["prompts"].shape[1]
        return batch.batch["prompts"][row][batch.batch["attention_mask"][row, :width].bool()].tolist()

    def _text(self, tokens):
        if tokens and tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _generate(self, requests, phase, timing):
        """Generate actual continuation tokens from exact token prefixes via the shared live servers."""

        async def run():
            semaphore = asyncio.Semaphore(self.iop.generation_concurrency)

            async def one(index, request):
                prompt, budget = request
                if budget < 1:
                    raise ValueError("IOP continuation has no remaining token budget")
                rollout = self.config.actor_rollout_ref.rollout
                if rollout.max_model_len is not None and len(prompt) + budget > rollout.max_model_len:
                    raise ValueError("IOP repair/graft context exceeds rollout.max_model_len; increase the budget")
                params = {
                    "max_tokens": budget,
                    "temperature": rollout.temperature,
                    "top_p": rollout.top_p,
                    "top_k": rollout.top_k,
                    "logprobs": False,
                    "seed": (self.iop.seed + self.global_steps * 2**32 + self._request_index + index) % (2**63 - 1),
                }
                async with semaphore:
                    output = await self.iop_client.generate(uuid4().hex, prompt_ids=prompt, sampling_params=params)
                if not output.token_ids or output.stop_reason == "aborted" or len(output.token_ids) > budget:
                    raise RuntimeError("IOP generation returned empty, aborted or over-budget tokens")
                return list(output.token_ids)

            return await asyncio.gather(*(one(i, request) for i, request in enumerate(requests)))

        if not requests:
            return []
        with marked_timer(f"iop_generate_{phase}", timing):
            result = asyncio.run(run())
        self._request_index += len(requests)
        self._metrics[f"iop/{phase}_input_tokens"] = self._metrics.get(f"iop/{phase}_input_tokens", 0) + sum(
            len(p) for p, _ in requests
        )
        self._metrics[f"iop/{phase}_generated_tokens"] = self._metrics.get(f"iop/{phase}_generated_tokens", 0) + sum(
            map(len, result)
        )
        return result

    def _build_batch(self, source, rows):
        """Build real contexts for policy, repair and verifier rows; never truncate a candidate."""
        if not rows:
            rows = [
                {
                    "parent": 0,
                    "prompt": self._prompt(source, 0),
                    "tokens": self._tokens(source, 0),
                    "kind": 2,
                    "uid": "iop-empty",
                    "reward": 0.0,
                }
            ]
        width = max(len(row["prompt"]) for row in rows)
        response_width = self.config.data.max_response_length
        inputs = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        responses = torch.full((len(rows), response_width), self.tokenizer.pad_token_id, dtype=torch.long)
        prompt_mask, response_mask = torch.zeros_like(inputs), torch.zeros_like(responses)
        gates = torch.zeros_like(responses, dtype=torch.bool)
        for i, row in enumerate(rows):
            prompt, tokens = row["prompt"], row["tokens"]
            if not prompt or not tokens or len(tokens) > response_width:
                raise ValueError("IOP needs nonempty complete contexts/responses within configured budgets")
            inputs[i, -len(prompt) :] = torch.tensor(prompt)
            prompt_mask[i, -len(prompt) :] = 1
            responses[i, : len(tokens)] = torch.tensor(tokens)
            if row["kind"] != 2:
                response_mask[i, : len(tokens)] = 1
            gate = row.get("gate", [False] * len(tokens))
            if len(gate) != len(tokens):
                raise ValueError("IOP gate length differs from its complete response")
            gates[i, : len(tokens)] = torch.tensor(gate, dtype=torch.bool)
        attention = torch.cat((prompt_mask, response_mask), -1)
        # A dummy keeps a legitimate context while every direct objective weight stays zero.
        for i, row in enumerate(rows):
            if row["kind"] == 2:
                attention[i, width : width + len(row["tokens"])] = 1
        tensors = {
            "prompts": inputs,
            "responses": responses,
            "response_mask": response_mask,
            "attention_mask": attention,
            "input_ids": torch.cat((inputs, responses), -1),
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
            "iop_gate": gates,
            "iop_kind": torch.tensor([r["kind"] for r in rows]),
            "iop_pg_weight": torch.tensor([r.get("weight", 0.0) for r in rows]),
            "iop_kl_weight": torch.tensor([r.get("weight", 0.0) for r in rows]),
        }
        if all("reward" in row for row in rows):
            scores = torch.zeros_like(responses, dtype=torch.float32)
            for i, row in enumerate(rows):
                if row["kind"] != 2:
                    scores[i, len(row["tokens"]) - 1] = row["reward"]
            tensors["rm_scores"] = scores
        parents = [row["parent"] for row in rows]
        metadata = {
            key: source.non_tensor_batch[key][parents].copy()
            for key in ("raw_prompt", "reward_model", "extra_info", "data_source", "iop_dataset_index")
        }
        metadata["extra_info"] = _objects(
            [
                {**dict(info), "iop_phase": rows[i].get("phase", "training")}
                for i, info in enumerate(metadata["extra_info"])
            ]
        )
        metadata["uid"] = _objects([r["uid"] for r in rows])
        metadata["iop_source_uid"] = source.non_tensor_batch["uid"][parents].copy()
        metadata["raw_prompt"] = _objects(
            [row.get("messages", metadata["raw_prompt"][i]) for i, row in enumerate(rows)]
        )
        metadata["multi_modal_inputs"] = _objects([{} for _ in rows])
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=metadata,
            meta_info={
                "reward_extra_keys": [],
                "global_steps": self.global_steps,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "multi_turn": False,
            },
        )

    def _verify(self, batch, timing):
        with marked_timer("iop_verify", timing):
            if "rm_scores" in batch.batch:
                scores = batch.batch["rm_scores"].double().sum(-1).tolist()
            else:
                workers = self.reward_loop_manager.reward_loop_workers
                chunks = np.array_split(np.arange(len(batch)), min(len(batch), len(workers)))
                outputs = ray.get(
                    [
                        worker.compute_score_batch.remote(batch.select_idxs(indices))
                        for worker, indices in zip(workers, chunks, strict=False)
                    ]
                )
                scores = [float(row["reward_score"]) for chunk in outputs for row in chunk]
            if len(scores) != len(batch) or any(score not in (0.0, 1.0) for score in scores):
                raise ValueError("IOP requires one unshaped binary task reward per complete response")
            return scores

    def _padding_divisor(self):
        cfg = self.config.actor_rollout_ref
        sizes = [self._get_dp_size(self.actor_rollout_wg, "actor")]
        for settings, key, dynamic in (
            (cfg.actor, "ppo_micro_batch_size_per_gpu", "use_dynamic_bsz"),
            (cfg.rollout, "log_prob_micro_batch_size_per_gpu", "log_prob_use_dynamic_bsz"),
            (cfg.ref, "log_prob_micro_batch_size_per_gpu", "log_prob_use_dynamic_bsz"),
        ):
            sizes.append(1 if settings.get(dynamic, False) else settings.get(key) or 1)
        divisor = sizes[0] * lcm(*sizes[1:])
        if self.use_reference_policy:
            divisor = lcm(divisor, self._get_dp_size(self.ref_policy_wg, "ref") * sizes[-1])
        return divisor

    def _prepare_training_batch(self, batch, metrics, timing_raw):
        self._metrics, self._request_index = metrics, 0
        required = {"raw_prompt", "reward_model", "extra_info", "data_source", "uid", "iop_dataset_index"}
        if not required.issubset(batch.non_tensor_batch):
            raise ValueError(f"IOP needs original metadata: {sorted(required - batch.non_tensor_batch.keys())}")
        rewards = self._verify(batch, timing_raw)
        labels = batch.non_tensor_batch["uid"]
        rng = np.random.default_rng(self.iop.seed + self.global_steps)
        errors, requests = [], []
        self.last_iop_trace = {"initial_rewards": rewards, "groups": []}
        for uid in dict.fromkeys(labels.tolist()):
            indices = np.flatnonzero(labels == uid).tolist()
            if len(indices) != self.config.actor_rollout_ref.rollout.n:
                raise ValueError("IOP initial group cardinality differs from rollout.n")
            correct = [i for i in indices if rewards[i] == 1]
            wrong = [i for i in indices if rewards[i] == 0]
            dataset_index = batch.non_tensor_batch["iop_dataset_index"][indices[0]]
            self.iop_sampler.defer(dataset_index, bool(wrong and not correct))
            if not wrong or not correct:
                continue
            anchor = int(rng.choice(correct))
            for row in wrong:
                original = self._tokens(batch, row)
                messages = repair_messages(
                    list(batch.non_tensor_batch["raw_prompt"][row]),
                    self._text(original),
                    self._text(self._tokens(batch, anchor)),
                )
                kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
                custom = self.config.actor_rollout_ref.model.get("custom_chat_template")
                if custom is not None:
                    kwargs["chat_template"] = custom
                prompt = normalize_token_ids(
                    self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
                )
                errors.append(
                    {
                        "parent": row,
                        "anchor": anchor,
                        "original": original,
                        "messages": messages,
                        "prompt": prompt,
                        "question_uid": uid,
                    }
                )
                requests.extend([(prompt, self.config.data.max_response_length)] * self.iop.repair_n)
        candidates = self._generate(requests, "repair", timing_raw)
        verifier_rows = [
            {
                "parent": error["parent"],
                "prompt": error["prompt"],
                "tokens": tokens,
                "kind": 1,
                "uid": f"iop-repair:{self.global_steps}:{i}",
                "phase": "repair",
            }
            for i, error in enumerate(errors)
            for tokens in candidates[i * self.iop.repair_n : (i + 1) * self.iop.repair_n]
        ]
        candidate_rewards = self._verify(self._build_batch(batch, verifier_rows), timing_raw) if verifier_rows else []
        accepted = []
        for i, error in enumerate(errors):
            offset = i * self.iop.repair_n
            tokens = candidates[offset : offset + self.iop.repair_n]
            task = candidate_rewards[offset : offset + self.iop.repair_n]
            row, anchor = error["parent"], error["anchor"]
            audits = []
            for candidate in tokens:
                decision = (
                    1
                    if self.audit is None
                    else self.audit(
                        question=list(batch.non_tensor_batch["raw_prompt"][row]),
                        original=self._text(error["original"]),
                        repair=self._text(candidate),
                        reference=self._text(self._tokens(batch, anchor)),
                        extra_info=dict(batch.non_tensor_batch["extra_info"][row]),
                    )
                )
                if decision not in (0, 1):
                    raise ValueError("IOP audit callback must return binary 0/1; no fallback is allowed")
                audits.append(int(decision))
            scores = repair_scores(error["original"], tokens, task, audits, self.iop.edit_weight)
            best = select_repair(scores, task, audits)
            trace = {
                "parent": row,
                "anchor": anchor,
                "scores": scores,
                "task_rewards": task,
                "audits": audits,
                "selected": best,
                "graft_trials": [],
            }
            self.last_iop_trace["groups"].append(trace)
            if best is None:
                continue
            alignment = align_tokens(error["original"], tokens[best])
            if alignment.distance == 0:
                continue
            chosen = alignment.distance
            for limit in (self.iop.truncation_k, 2 * self.iop.truncation_k):
                if limit >= alignment.distance:
                    break
                prefix = alignment.graft_prefix(limit)
                remaining = self.config.data.max_response_length - len(prefix)
                if remaining and (not prefix or prefix[-1] != self.tokenizer.eos_token_id):
                    suffix = self._generate([(self._prompt(batch, row) + prefix, remaining)], "graft", timing_raw)[0]
                else:
                    suffix = []
                trial = {
                    "parent": row,
                    "prompt": self._prompt(batch, row),
                    "tokens": prefix + suffix,
                    "kind": 0,
                    "uid": "iop-graft",
                    "phase": "graft",
                }
                success = self._verify(self._build_batch(batch, [trial]), timing_raw)[0]
                trace["graft_trials"].append({"limit": limit, "prefix": prefix, "reward": success})
                if success == 1:
                    chosen = limit
                    break
            negative_gate, positive_gate = alignment.masks(chosen)
            trace.update(
                chosen_edits=chosen,
                distance=alignment.distance,
                negative_gate=negative_gate,
                positive_gate=positive_gate,
            )
            if self.iop.get("trace", False):
                trace.update(original=error["original"], candidates=tokens, repair_prompt=error["prompt"])
            accepted.append(
                {
                    **error,
                    "tokens": tokens,
                    "scores": scores,
                    "best": best,
                    "negative_gate": negative_gate,
                    "positive_gate": positive_gate,
                }
            )
        counts = {}
        for error in accepted:
            counts[error["question_uid"]] = counts.get(error["question_uid"], 0) + 1
        for uid in dict.fromkeys(labels.tolist()):
            row = int(np.flatnonzero(labels == uid)[0])
            if any(rewards[i] == 0 for i in np.flatnonzero(labels == uid)):
                self.iop_sampler.defer(batch.non_tensor_batch["iop_dataset_index"][row], uid not in counts)
        rows = []
        for i, error in enumerate(accepted):
            row = error["parent"]
            pairs = counts[error["question_uid"]]
            policy_weight = 1 / (2 * len(counts) * pairs)
            uid = f"iop-pol:{self.global_steps}:{i}"
            for tokens, gate, reward in (
                (error["original"], error["negative_gate"], 0.0),
                (error["tokens"][error["best"]], error["positive_gate"], 1.0),
            ):
                rows.append(
                    {
                        "parent": row,
                        "prompt": self._prompt(batch, row),
                        "tokens": tokens,
                        "kind": 0,
                        "uid": uid,
                        "gate": gate,
                        "reward": reward,
                        "weight": policy_weight,
                    }
                )
            weight = self.iop.repair_weight / (len(counts) * pairs * self.iop.repair_n)
            for tokens, reward in zip(error["tokens"], error["scores"], strict=True):
                rows.append(
                    {
                        "parent": row,
                        "prompt": error["prompt"],
                        "tokens": tokens,
                        "messages": error["messages"],
                        "kind": 1,
                        "uid": f"iop-rep:{self.global_steps}:{i}",
                        "gate": [True] * len(tokens),
                        "reward": reward,
                        "weight": weight,
                    }
                )
        result = self._build_batch(batch, rows)
        real_size = len(rows)
        result, pad = pad_dataproto_to_divisor(result, self._padding_divisor())
        if pad:
            for key in ("response_mask", "rm_scores", "iop_gate", "iop_pg_weight", "iop_kl_weight"):
                result.batch[key][-pad:].zero_()
            result.batch["iop_kind"][-pad:] = 2
            for index in range(len(result) - pad, len(result)):
                result.non_tensor_batch["uid"][index] = f"iop-pad:{self.global_steps}:{index}"
        metrics.update(
            {
                "iop/initial_correctness": sum(rewards) / len(rewards),
                "iop/errors": len(errors),
                "iop/repair_generated": len(candidates),
                "iop/qualified_groups": len(accepted),
                "iop/policy_pairs": len(accepted),
                "iop/active_questions": len(counts),
                "iop/train_rows": real_size,
                "iop/padding_rows": len(result) - real_size,
                "iop/audit_rejections": sum(1 - h for trace in self.last_iop_trace["groups"] for h in trace["audits"]),
                "iop/deferred_questions": len(self.iop_sampler.deferred),
                "iop/graft_trials": sum(len(g["graft_trials"]) for g in self.last_iop_trace["groups"]),
            }
        )
        return result

    def _update_actor(self, batch):
        if not (batch.batch["iop_pg_weight"] > 0).any():
            return DataProto.from_single_dict(data={}, meta_info={"metrics": {"iop/skipped_empty_update": 1.0}})
        batch.meta_info.update(temperature=self.config.actor_rollout_ref.rollout.temperature, multi_turn=False)
        data = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(
            data,
            calculate_entropy=False,
            distillation_use_topk=False,
            global_batch_size=len(batch),
            mini_batch_size=len(batch),
            epochs=1,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": False},
            compute_loss=True,
        )
        output = self.actor_rollout_wg.update_actor(data)
        metrics = rename_dict(tu.get(output, "metrics"), "actor/")
        metrics["perf/mfu/actor"] = metrics.pop("actor/mfu")
        metrics["iop/skipped_empty_update"] = 0.0
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

    def _compute_training_metrics(self, batch, timing_raw, n_gpus, gradient_norm=None):
        indices = (batch.batch["iop_kind"] < 2).nonzero(as_tuple=True)[0].cpu().numpy()
        if not len(indices):
            return {"iop/skipped_empty_batch": 1.0}
        real = batch.select_idxs(indices)
        real.meta_info = dict(batch.meta_info)
        real.meta_info["global_token_num"] = real.batch["attention_mask"].sum(-1).tolist()
        metrics = super()._compute_training_metrics(real, timing_raw, n_gpus, gradient_norm)
        width = real.batch["prompts"].shape[1]
        lengths = real.batch["attention_mask"][:, :width].sum(-1)
        model_limit = self.config.actor_rollout_ref.rollout.max_model_len
        repair_limit = (model_limit - self.config.data.max_response_length) if model_limit else 2**31
        limits = torch.where(real.batch["iop_kind"] == 1, repair_limit, self.config.data.max_prompt_length)
        metrics["prompt_length/clip_ratio"] = (lengths >= limits).float().mean().item()
        return metrics

    def _save_checkpoint(self):
        super()._save_checkpoint()
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        torch.save(self.iop_sampler.state_dict(), folder / "iop_sampler.pt")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps:
            folder = (
                Path(self.config.trainer.resume_from_path)
                if self.config.trainer.resume_mode == "resume_path"
                else Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            )
            self.iop_sampler.load_state_dict(torch.load(folder / "iop_sampler.pt", weights_only=False))
