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
"""IOP losses installed through ActorRolloutRefWorker.set_loss_fn."""

from math import isfinite

import torch

from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding


def joint_policy_loss(
    log_prob,
    old_log_prob,
    advantages,
    response_mask,
    gate,
    is_repair,
    row_weights,
    clip_low=0.2,
    clip_high=0.2,
    dp_size=1,
):
    """Return the globally normalized gated policy plus full-repair GSPO surrogate."""
    if log_prob.ndim != 2 or any(t.shape != log_prob.shape for t in (old_log_prob, advantages, response_mask, gate)):
        raise ValueError("IOP token fields must have identical [rows,tokens] shapes")
    if response_mask.dtype != torch.bool or gate.dtype != torch.bool or (gate & ~response_mask).any():
        raise ValueError("IOP boolean gates must be contained in the response mask")
    if is_repair.shape != log_prob.shape[:1] or row_weights.shape != is_repair.shape:
        raise ValueError("IOP row fields must have one value per row")
    if is_repair.dtype != torch.bool or not torch.isfinite(row_weights).all() or (row_weights < 0).any():
        raise ValueError("IOP row flags/weights are invalid")
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError("IOP dp_size must be a positive integer")
    if not isfinite(clip_low) or not isfinite(clip_high) or not 0 <= clip_low < 1 or clip_high < 0:
        raise ValueError("IOP clip bounds must be finite, with 0<=low<1 and high>=0")
    mask = gate & response_mask & (row_weights > 0).unsqueeze(-1)
    current = torch.where(mask, log_prob, 0)
    old = torch.where(mask, old_log_prob.detach(), 0)
    advantage = torch.where(mask, advantages.detach(), 0)
    if not all(torch.isfinite(value).all() for value in (current, old, advantage)):
        raise ValueError("IOP active probabilities and advantages must be finite")
    counts = mask.sum(-1).clamp_min(1)
    ratio = ((current - old).sum(-1) / counts).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("IOP sequence ratios must be finite")
    policy_ratio = ratio.detach().unsqueeze(-1) * (current - current.detach()).exp()
    # The repair objective differentiates its full-sequence GSPO ratio.
    token_ratio = torch.where(is_repair.unsqueeze(-1), ratio.unsqueeze(-1), policy_ratio)
    terms = torch.maximum(-advantage * token_ratio, -advantage * token_ratio.clamp(1 - clip_low, 1 + clip_high))
    row_loss = torch.where(mask, terms, 0).sum(-1) / counts
    weighted = row_loss * row_weights.detach() * dp_size
    return weighted.sum(), {
        "iop_policy_loss": weighted.masked_fill(is_repair, 0).sum().detach(),
        "iop_repair_loss": weighted.masked_fill(~is_repair, 0).sum().detach(),
    }


def iop_loss(config, model_output, data, dp_group=None):
    """Use the standard engine output layout, with IOP's own global row weights."""
    del dp_group
    current = no_padding_2_padding(model_output["log_probs"], data)
    fields = ["response_mask", "old_log_probs", "advantages", "iop_gate", "iop_kind", "iop_pg_weight", "iop_kl_weight"]
    if config.use_kl_loss:
        fields.append("ref_log_prob")
    dp_size = tu.get_non_tensor_data(data, "dp_size", 1)
    batch = data.select(*fields).to_padded_tensor()
    response_mask = batch["response_mask"].bool()
    low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    loss, metrics = joint_policy_loss(
        current,
        batch["old_log_probs"],
        batch["advantages"],
        response_mask,
        batch["iop_gate"],
        batch["iop_kind"] == 1,
        batch["iop_pg_weight"],
        low,
        high,
        dp_size,
    )
    metrics["pg_loss"] = loss.detach()
    if config.entropy_coeff != 0:
        raise ValueError("IOP does not add an unspecified entropy objective")
    if config.use_kl_loss:
        if not isfinite(config.kl_loss_coef) or config.kl_loss_coef < 0:
            raise ValueError("IOP KL coefficient must be finite and nonnegative")
        weights = batch["iop_kl_weight"].detach()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("IOP KL row weights must be finite and nonnegative")
        mask = response_mask & (weights > 0).unsqueeze(-1)
        logp, ref = torch.where(mask, current, 0), torch.where(mask, batch["ref_log_prob"].detach(), 0)
        if not torch.isfinite(logp).all() or not torch.isfinite(ref).all():
            raise ValueError("IOP active KL probabilities must be finite")
        kld = kl_penalty(logp, ref, kl_penalty=config.kl_loss_type)
        kl = ((torch.where(mask, kld, 0).sum(-1) / mask.sum(-1).clamp_min(1)) * weights).sum() * dp_size
        loss = loss + config.kl_loss_coef * kl
        metrics["kl_loss"] = kl.detach()
    return loss, Metric.from_dict(metrics, aggregation=AggregationType.SUM)
