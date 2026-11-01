# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GPU model check for native equal-length forks; no downloaded weights.

Prepare with the CPU environment, then run with the CUDA environment. Reports
correctness and scheduling evidence, not throughput or GPU utilization.

Run from a vLLM checkout with a CUDA build, for example::

    uv run --python .venv/bin/python --no-project \
        examples/basic/offline_inference/equal_length_gpu_check.py \
        --prompt-lengths 1,2,15,16,17
    uv run --python .venv/bin/python --no-project \
        examples/basic/offline_inference/equal_length_gpu_check.py \
        --family qwen3next --prefix-cache off --runner v2

Use --prepare-only with a CPU environment to create the small checkpoints.
Use --graph to validate CUDA Graph mode separately. The JSON report defaults
under /tmp; choose --output to retain it elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def prepare_model(family: str, root: Path) -> Path:
    import torch
    from transformers import (
        OPTConfig,
        OPTForCausalLM,
        Qwen3NextConfig,
        Qwen3NextForCausalLM,
    )

    path = root / family
    if (path / "config.json").exists() and (path / "model.safetensors").exists():
        config = json.loads((path / "config.json").read_text())
        architecture = "OPTForCausalLM" if family == "opt" else "Qwen3NextForCausalLM"
        valid = (
            config.get("architectures") == [architecture]
            and config.get("model_type") == ("opt" if family == "opt" else "qwen3_next")
            and config.get("vocab_size") == 128
            and config.get("max_position_embeddings", 0) >= 2048
        )
        if family == "qwen3next":
            layers = config.get("layer_types", [])
            sparse_step = config.get("decoder_sparse_step", 1)
            experts = config.get("num_experts", 0)
            selected_experts = config.get("num_experts_per_tok", 0)
            valid = (
                valid
                and len(layers) == config.get("num_hidden_layers")
                and set(layers) == {"linear_attention", "full_attention"}
                and sparse_step > 0
                and 1 <= selected_experts <= experts
                and any(
                    index not in config.get("mlp_only_layers", [])
                    and (index + 1) % sparse_step == 0
                    for index in range(len(layers))
                )
            )
        if not valid:
            raise ValueError(
                f"Incompatible existing checkpoint at {path}; remove the generated "
                "checkpoint or choose a different --model-root. Qwen3Next requires "
                "full attention, GDN and at least one MoE layer."
            )
        return path
    torch.manual_seed(41219)
    common = dict(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    if family == "opt":
        model = OPTForCausalLM(
            OPTConfig(
                **common,
                ffn_dim=256,
                word_embed_proj_dim=128,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
    else:
        model = Qwen3NextForCausalLM(
            Qwen3NextConfig(
                **common,
                intermediate_size=256,
                num_key_value_heads=1,
                head_dim=64,
                layer_types=["linear_attention", "full_attention"],
                linear_conv_kernel_dim=4,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                num_experts=2,
                num_experts_per_tok=1,
                moe_intermediate_size=128,
                shared_expert_intermediate_size=128,
                mlp_only_layers=[0],
            )
        )
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    return path


def run(args, model_path: Path) -> dict:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "v2" else "0"
    import torch

    assert torch.cuda.is_available(), "CUDA PyTorch cannot see a GPU"
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.rl.equal_length_rollout import generate_equal_length_pairs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.core.equal_length import (
        NativePairConfig,
        get_native_pair_config,
        with_native_pair_config,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, iter_layer_specs

    kwargs = dict(
        model=str(model_path),
        skip_tokenizer_init=True,
        dtype=args.dtype,
        gpu_memory_utilization=0.35,
        enforce_eager=not args.graph,
        async_scheduling=False,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        enable_prefix_caching=args.prefix_cache == "on",
        block_size=16,
        stream_interval=args.stream_interval,
        seed=41219,
    )
    if getattr(args, "router_fp32", False):
        kwargs["hf_overrides"] = {"eqlen_router_fp32": True}
    if args.family == "qwen3next":
        kwargs["mamba_cache_mode"] = "align" if args.prefix_cache == "on" else "none"
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.graph:
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4]}
    llm = LLM(**kwargs)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager
    assert engine.engine_core.supports_native_equal_length_pairs
    full_sizes = sorted(
        {
            spec.block_size
            for group in manager.kv_cache_config.kv_cache_groups
            for spec in iter_layer_specs(group.kv_cache_spec)
            if isinstance(spec, FullAttentionSpec)
        }
    )
    lengths = [int(value) for value in args.prompt_lengths.split(",")]
    if args.family == "qwen3next":
        for size in full_sizes:
            lengths.extend([size - 1, size, size + 1])
    lengths = sorted({value for value in lengths if 0 < value < 2040})
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        family=args.family,
        runner=args.runner,
        graph=args.graph,
        dtype=args.dtype,
        prefix_cache=args.prefix_cache,
        stream_interval=args.stream_interval,
        router_fp32=getattr(args, "router_fp32", False),
        full_attention_block_sizes=full_sizes,
        cases=[],
        forks=[],
        schedule=[],
        cancellation=None,
        status="running",
    )
    current: int | str | None = None
    step_index = 0
    original_schedule = scheduler.schedule
    original_fork = manager.fork_computed_prefix

    def traced_schedule(*a, **kw):
        nonlocal step_index
        warming = {
            group_id
            for group_id, pair in scheduler._native_pairs.items()
            if pair.warming
        }
        result = original_schedule(*a, **kw)
        warming.update(
            group_id
            for group_id, pair in scheduler._native_pairs.items()
            if pair.warming
        )
        step_index += 1
        native = {
            rid: count
            for rid, count in result.num_scheduled_tokens.items()
            if rid in scheduler.requests
            and get_native_pair_config(scheduler.requests[rid].sampling_params)
            is not None
        }
        if native:
            report["schedule"].append(
                dict(case=current, step=step_index, tokens=native)
            )
            grouped: dict[str, list[int]] = {}
            for rid, count in native.items():
                config = get_native_pair_config(scheduler.requests[rid].sampling_params)
                grouped.setdefault(config.group_id, []).append(count)
            for group_id, counts in grouped.items():
                if group_id in warming:
                    assert len(counts) == 1, "native problem prefix was computed twice"
                else:
                    assert counts == [1, 1], "native pair did not decode together"
        return result

    def traced_fork(source, target):
        computed = source.num_computed_tokens
        warmup = source.num_output_tokens == 0
        source_ids = manager.get_block_ids(source.request_id)
        ok = original_fork(source, target)
        assert ok, "native fork refused a settled prefix"
        assert target.num_computed_tokens == computed
        target_ids = manager.get_block_ids(target.request_id)
        groups = []
        for group, before, after in zip(
            manager.kv_cache_config.kv_cache_groups, source_ids, target_ids, strict=True
        ):
            specs = iter_layer_specs(group.kv_cache_spec)
            spec = specs[0]
            item = dict(kind=type(spec).__name__, block_size=spec.block_size)
            if isinstance(spec, FullAttentionSpec):
                complete = computed // spec.block_size
                assert before[:complete] == after[:complete], "full prefix pages copied"
                if computed % spec.block_size:
                    assert before[complete] != after[complete], "writable tail shared"
                item.update(
                    shared_complete_pages=complete,
                    partial_tail_copied=bool(computed % spec.block_size),
                )
            groups.append(item)
        report["forks"].append(
            dict(
                case=current,
                step=step_index,
                computed=computed,
                warmup=warmup,
                source=source.request_id,
                target=target.request_id,
                groups=groups,
            )
        )
        return ok

    scheduler.schedule = traced_schedule
    manager.fork_computed_prefix = traced_fork
    try:
        for prompt_len in lengths:
            current = prompt_len
            prompt = [3 + (position % 119) for position in range(prompt_len)]
            # Discover two independent first tokens, then make slot 0 stop at its
            # first token while slot 1 continues. This exercises a real EOS fork.
            seed = None
            for candidate in range(16):
                outputs = llm.generate(
                    [{"prompt_token_ids": prompt}] * 2,
                    [
                        SamplingParams(
                            temperature=1.0,
                            seed=candidate + slot,
                            max_tokens=1,
                            ignore_eos=True,
                            detokenize=False,
                        )
                        for slot in range(2)
                    ],
                    use_tqdm=False,
                )
                first = [out.outputs[0].token_ids[0] for out in outputs]
                if first[0] != first[1]:
                    seed, stop_id = candidate, first[0]
                    break
            assert seed is not None, "could not obtain different first samples"
            # Eliminate discovery-cache hits so warmup and tail CoW are real.
            llm.reset_prefix_cache()
            fork_start = len(report["forks"])
            [[rollout]] = generate_equal_length_pairs(
                engine,
                [prompt],
                SamplingParams(
                    temperature=1.0,
                    seed=seed,
                    max_tokens=8,
                    ignore_eos=True,
                    stop_token_ids=[stop_id],
                    logprobs=0,
                    detokenize=False,
                ),
            )
            all_forks = report["forks"][fork_start:]
            forks = [item for item in all_forks if not item["warmup"]]
            assert len(rollout.pairs) >= 2 and forks, "EOS did not renew a branch"
            assert forks[0]["computed"] == prompt_len
            for item in forks:
                next_step = next(
                    (
                        row
                        for row in report["schedule"]
                        if row["step"] > item["step"]
                        and item["target"] in row["tokens"]
                    ),
                    None,
                )
                assert next_step is not None, "replacement never scheduled"
                assert next_step["step"] == item["step"] + 1, "replacement waited"
                assert next_step["tokens"][item["target"]] == 1, "prefix replayed"
            trajectories = rollout.trajectories()
            scores = llm.generate(
                [
                    {"prompt_token_ids": prompt + trajectory.response_token_ids}
                    for trajectory in trajectories
                ],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    ignore_eos=True,
                    detokenize=False,
                    prompt_logprobs=0,
                ),
                use_tqdm=False,
            )
            max_error = 0.0
            for trajectory, score in zip(trajectories, scores, strict=True):
                assert trajectory.logprobs is not None
                assert score.prompt_logprobs is not None
                for index, (token, actual) in enumerate(
                    zip(trajectory.response_token_ids, trajectory.logprobs, strict=True)
                ):
                    position = score.prompt_logprobs[prompt_len + index]
                    assert position is not None and token in position
                    expected = position[token].logprob
                    error = abs(actual - expected)
                    assert error <= args.logprob_tolerance, (
                        f"prefix={prompt_len} response_index={index} token={token} "
                        f"native_logprob={actual} full_prefix_logprob={expected}"
                    )
                    max_error = max(max_error, error)
            # Change earlier prefix values, while keeping lengths and all
            # response tokens fixed. The immediate input before each response
            # score remains identical; a detectable score change demonstrates
            # that this model/reference can expose incorrect inherited state.
            sensitivity = None
            if prompt_len > 1:
                changed_prompt = [
                    3 + ((token - 3 + 53) % 119) for token in prompt[:-1]
                ] + prompt[-1:]
                controls = llm.generate(
                    [
                        {"prompt_token_ids": changed_prompt + t.response_token_ids}
                        for t in trajectories
                    ],
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        ignore_eos=True,
                        detokenize=False,
                        prompt_logprobs=0,
                    ),
                    use_tqdm=False,
                )
                sensitivity = max(
                    abs(
                        score.prompt_logprobs[prompt_len + index][token].logprob
                        - control.prompt_logprobs[prompt_len + index][token].logprob
                    )
                    for trajectory, score, control in zip(
                        trajectories, scores, controls, strict=True
                    )
                    for index, token in enumerate(trajectory.response_token_ids)
                )
                assert sensitivity > max(5 * max_error, args.logprob_tolerance), (
                    f"reference not sensitive enough: prefix={prompt_len}, "
                    f"control_delta={sensitivity}, native_error={max_error}"
                )
            assert not engine.has_unfinished_requests()
            assert not scheduler._native_pairs and not scheduler._native_pair_aliases
            assert not manager._unexecuted_prefix_forks
            report["cases"].append(
                dict(
                    prompt_tokens=prompt_len,
                    seed=seed,
                    stop_token=stop_id,
                    segment_pairs=len(rollout.pairs),
                    eos_forks=len(forks),
                    warmup_forks=sum(item["warmup"] for item in all_forks),
                    trajectories=len(trajectories),
                    max_logprob_error=max_error,
                    changed_prefix_score_delta=sensitivity,
                )
            )
        if getattr(args, "long", False):
            current = "long_router"
            long_prompt = [3 + (position % 119) for position in range(271)]
            discovery = llm.generate(
                [{"prompt_token_ids": long_prompt}],
                SamplingParams(
                    temperature=1.0,
                    seed=0,
                    max_tokens=1,
                    ignore_eos=True,
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            long_stop = discovery[0].outputs[0].token_ids[0]
            llm.reset_prefix_cache()
            [long_groups] = generate_equal_length_pairs(
                engine,
                [long_prompt],
                SamplingParams(
                    temperature=1.0,
                    seed=0,
                    max_tokens=640,
                    min_tokens=64,
                    ignore_eos=True,
                    stop_token_ids=[long_stop],
                    logprobs=0,
                    detokenize=False,
                ),
                num_pairs=2,
            )
            long_trajectories = [
                t for group in long_groups for t in group.trajectories()
            ]
            long_scores = llm.generate(
                [
                    {"prompt_token_ids": long_prompt + t.response_token_ids}
                    for t in long_trajectories
                ],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    ignore_eos=True,
                    detokenize=False,
                    prompt_logprobs=0,
                ),
                use_tqdm=False,
            )
            long_error = max(
                abs(value - score.prompt_logprobs[271 + index][token].logprob)
                for trajectory, score in zip(
                    long_trajectories, long_scores, strict=True
                )
                for index, (token, value) in enumerate(
                    zip(trajectory.response_token_ids, trajectory.logprobs, strict=True)
                )
            )
            report["long_case"] = dict(
                prompt_tokens=271,
                response_budget=640,
                min_tokens=64,
                segment_pairs=[len(group.pairs) for group in long_groups],
                trajectories=len(long_trajectories),
                max_logprob_error=long_error,
                max_response_tokens=max(
                    len(t.response_token_ids) for t in long_trajectories
                ),
            )
            assert long_error <= args.logprob_tolerance, report["long_case"]
            assert not engine.has_unfinished_requests()

        assert sum(not item["warmup"] for item in report["forks"]) >= 2, (
            "insufficient independent EOS fork coverage"
        )
        # Abort in the real gap after EOS fork and before its queued CoW copy
        # reaches the next GPU schedule. Then execute an ordinary request so
        # worker-side finished IDs and copy-retention fences are consumed.
        current = "cancel_after_eos_fork"
        assert seed is not None
        group_id = "gpu-native-cancel"
        cancel_start = len(report["forks"])
        for slot in range(2):
            params = with_native_pair_config(
                SamplingParams(
                    temperature=1.0,
                    seed=seed + slot,
                    max_tokens=8,
                    ignore_eos=True,
                    stop_token_ids=[stop_id],
                    detokenize=False,
                    output_kind=RequestOutputKind.DELTA,
                ),
                NativePairConfig(group_id, slot, slot),
            )
            engine.add_request(group_id, {"prompt_token_ids": prompt}, params)
        cancellation_fork = None
        for _ in range(12):
            assert engine.has_unfinished_requests()
            engine.step()
            cancellation_fork = next(
                (item for item in report["forks"][cancel_start:] if not item["warmup"]),
                None,
            )
            if cancellation_fork is not None:
                break
        assert cancellation_fork is not None, "cancel case did not reach EOS fork"
        target_id = cancellation_fork["target"]
        assert target_id in manager._unexecuted_prefix_forks
        engine.abort_request([group_id])
        assert not engine.has_unfinished_requests()
        assert not scheduler._native_pairs and not scheduler._native_pair_aliases
        assert not manager._unexecuted_prefix_forks
        ordinary = llm.generate(
            [{"prompt_token_ids": [7, 8, 9]}],
            SamplingParams(
                temperature=0.0,
                max_tokens=2,
                ignore_eos=True,
                detokenize=False,
                logprobs=0,
            ),
            use_tqdm=False,
        )
        assert len(ordinary[0].outputs[0].token_ids) == 2
        assert not engine.has_unfinished_requests()
        assert not scheduler.requests
        assert all(
            block.ref_cnt == 0
            for block in manager.block_pool.blocks
            if not block.is_null
        ), "cancel left live KV references"
        assert all(
            not item._pending_cow_copies
            for item in manager.coordinator.single_type_managers
        )
        assert not manager._unexecuted_prefix_forks
        report["cancellation"] = dict(
            target=target_id,
            after_eos_fork_before_next_schedule=True,
            group_aliases_cleared=True,
            unexecuted_copies_cleared=True,
            ordinary_gpu_forward_survived=True,
            live_kv_references=0,
        )
        report["status"] = "passed"
        return report
    finally:
        scheduler.schedule = original_schedule
        manager.fork_computed_prefix = original_fork
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["opt", "qwen3next"], default="opt")
    parser.add_argument(
        "--model-root", type=Path, default=Path("/tmp/vllm-native-models")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--runner", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--dtype", choices=["half", "float32", "bfloat16"], default="half"
    )
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--router-fp32", action="store_true")
    parser.add_argument(
        "--long", action="store_true", help="Check the 271/640-token MoE regression."
    )
    parser.add_argument("--prefix-cache", choices=["on", "off"], default="on")
    parser.add_argument("--stream-interval", type=int, default=10)
    parser.add_argument("--prompt-lengths", default="15,16,17")
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--logprob-tolerance", type=float, default=0.003)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/vllm-native-gpu-result.json")
    )
    args = parser.parse_args()
    model_path = prepare_model(args.family, args.model_root)
    if args.prepare_only:
        print(json.dumps(dict(model=str(model_path), status="prepared")))
        return
    try:
        result = run(args, model_path)
    except BaseException as error:
        result = json.loads(args.output.read_text()) if args.output.exists() else {}
        result.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    print(
        json.dumps(
            dict(
                status=result["status"],
                cases=len(result["cases"]),
                eos_forks=sum(not item["warmup"] for item in result["forks"]),
                warmup_forks=sum(item["warmup"] for item in result["forks"]),
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
