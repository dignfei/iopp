# -*- coding: utf-8 -*-
"""
普通 GRPO/DAPO 风格"独立 KL 分支"的梯度统计工具。
由 kl_hooks.before_train_batch 在每步第一个 mini-batch（KLSTATS_ENABLE=1 时）调用。
只依赖标准张量与 torch.distributed，不依赖 verl 版本细节。

统计量定义（与 paper/README.md 诊断一节一致）：
  S1 共享前缀 token 占比：全组 G 条回复的最长公共前缀（LCP）内的 token 数 / 全部回复 token 数
  S2 共享前缀上的系数级抵消：
       奖励分支 token 系数 a_it = A_i * r_it * gate_it   （PPO 符号相关裁剪门控）
       净值 net = Σ_pos |Σ_i a_it| ，毛量 gross = Σ_pos Σ_i |a_it| ，抵消率 = 1 - net/gross
       KL 分支 token 系数 k_it = β * d k3/d logπ = β * (1 - π_ref/π_θ)，同样定义净值/毛量
  S4 分支梯度范数：||g_PG||、||g_KL||、||g_PG+g_KL||、cos(g_PG,g_KL)、KL 占比 = ||g_KL|| / ||g_PG+g_KL||
  S5 共享前缀 token 单独贡献的梯度范数：||g_PG,shared||、||g_KL,shared||（分母与整批相同，可与 S4 直接比）
  S6 逐组：组内相似度（LCP 占比、成对 4-gram Jaccard）与该组的 ||g_PG,g||、||g_KL,g||
"""
import json
import os
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 1. 共享前缀与相似度（CPU，纯 python）
# ---------------------------------------------------------------------------
def _lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def group_by_uid(uids):
    groups = {}
    for idx, u in enumerate(uids):
        groups.setdefault(str(u), []).append(idx)
    return groups


def shared_prefix_stats(responses, response_mask, uids, ngram=4, max_pairs=40):
    """
    responses: LongTensor [B, L]; response_mask: [B, L]; uids: 长度 B。
    返回 (batch 级统计 dict, shared_mask[B, L] bool, 逐组列表)。
    """
    resp = responses.detach().cpu()
    mask = response_mask.detach().cpu().bool()
    lengths = mask.sum(-1).tolist()
    B, L = resp.shape
    shared_mask = torch.zeros(B, L, dtype=torch.bool)
    groups = group_by_uid(uids)
    tot_tokens = 0
    shared_tokens = 0
    pair_lcp_frac = []
    pair_jacc = []
    per_group = []
    for u, idxs in groups.items():
        seqs = [resp[i, : lengths[i]].tolist() for i in idxs]
        g_lcp = len(seqs[0])
        for s in seqs[1:]:
            g_lcp = min(g_lcp, _lcp_len(seqs[0], s))
            if g_lcp == 0:
                break
        for i in idxs:
            shared_mask[i, :g_lcp] = True
        gtok = sum(lengths[i] for i in idxs)
        tot_tokens += gtok
        shared_tokens += g_lcp * len(idxs)
        pairs = [(a, b) for a in range(len(seqs)) for b in range(a + 1, len(seqs))]
        if len(pairs) > max_pairs:
            step = len(pairs) / max_pairs
            pairs = [pairs[int(k * step)] for k in range(max_pairs)]
        grams = [set(tuple(s[k : k + ngram]) for k in range(max(0, len(s) - ngram + 1))) for s in seqs]
        pl, pj = [], []
        for a, b in pairs:
            l = _lcp_len(seqs[a], seqs[b])
            m = max(1, min(len(seqs[a]), len(seqs[b])))
            pl.append(l / m)
            inter = len(grams[a] & grams[b])
            union = max(1, len(grams[a] | grams[b]))
            pj.append(inter / union)
        pair_lcp_frac += pl
        pair_jacc += pj
        per_group.append(
            {
                "uid": str(u),
                "G": len(idxs),
                "rows": idxs,
                "group_lcp": g_lcp,
                "group_lcp_frac": g_lcp / max(1.0, gtok / len(idxs)),
                "pair_lcp_frac": sum(pl) / max(1, len(pl)),
                "pair_jaccard": sum(pj) / max(1, len(pj)),
                "mean_len": gtok / len(idxs),
            }
        )
    stats = {
        "shared_prefix/token_frac_group": shared_tokens / max(1, tot_tokens),
        "shared_prefix/pair_lcp_frac_mean": sum(pair_lcp_frac) / max(1, len(pair_lcp_frac)),
        "shared_prefix/pair_jaccard_mean": sum(pair_jacc) / max(1, len(pair_jacc)),
        "shared_prefix/groups": len(groups),
        "shared_prefix/tokens_total": tot_tokens,
        "shared_prefix/tokens_shared": shared_tokens,
    }
    return stats, shared_mask, per_group


# ---------------------------------------------------------------------------
# 2. 共享前缀上的系数级抵消（无需反向传播）
# ---------------------------------------------------------------------------
def kl_token_coef(log_prob, ref_log_prob, kl_coef, kl_type):
    """独立 KL 损失对 logπ 的逐 token 导数系数（乘 β）。"""
    if kl_type in ("low_var_kl", "k3"):
        return kl_coef * (1.0 - torch.exp(ref_log_prob - log_prob))
    if kl_type in ("kl", "k1"):
        return kl_coef * torch.ones_like(log_prob)
    if kl_type in ("mse", "k2"):
        return kl_coef * (log_prob - ref_log_prob)
    if kl_type in ("abs",):
        return kl_coef * torch.sign(log_prob - ref_log_prob)
    return kl_coef * (1.0 - torch.exp(ref_log_prob - log_prob))


@torch.no_grad()
def coefficient_stats(
    advantages, log_prob, old_log_prob, ref_log_prob, response_mask, shared_mask, uids,
    clip_low, clip_high, kl_coef, kl_type="low_var_kl",
):
    ratio = torch.exp(log_prob - old_log_prob)
    A = advantages
    pos = A > 0
    clipped = torch.where(pos, ratio > 1.0 + clip_high, ratio < 1.0 - clip_low)
    gate = (~clipped).float()
    a = A * ratio * gate
    kcoef = kl_token_coef(log_prob, ref_log_prob, kl_coef, kl_type)
    m = response_mask.bool()
    sm = shared_mask.to(m.device) & m
    groups = group_by_uid(uids)
    net = gross = knet = kgross = 0.0
    n_pos = 0
    for u, idxs in groups.items():
        ii = torch.tensor(idxs, device=a.device)
        g_sm = sm[ii]
        if not bool(g_sm.any()):
            continue
        cols = g_sm.any(0)
        a_g = (a[ii] * g_sm).sum(0)[cols]
        ab_g = (a[ii].abs() * g_sm).sum(0)[cols]
        k_g = (kcoef[ii] * g_sm).sum(0)[cols]
        kb_g = (kcoef[ii].abs() * g_sm).sum(0)[cols]
        net += a_g.abs().sum().item()
        gross += ab_g.sum().item()
        knet += k_g.abs().sum().item()
        kgross += kb_g.sum().item()
        n_pos += int(cols.sum().item())
    tot_a = (a.abs() * m).sum().item()
    tot_k = (kcoef.abs() * m).sum().item()
    n_tok = max(1, int(m.sum().item()))
    out = {
        "shared_prefix/coef_reward_net": net,
        "shared_prefix/coef_reward_gross": gross,
        "shared_prefix/coef_reward_cancel_ratio": 1.0 - net / max(gross, 1e-12),
        "shared_prefix/coef_kl_net": knet,
        "shared_prefix/coef_kl_gross": kgross,
        "shared_prefix/coef_kl_cancel_ratio": 1.0 - knet / max(kgross, 1e-12),
        "shared_prefix/coef_reward_gross_frac_of_batch": gross / max(tot_a, 1e-12),
        "shared_prefix/coef_kl_gross_frac_of_batch": kgross / max(tot_k, 1e-12),
        "shared_prefix/positions": n_pos,
        "coef/reward_abs_mean_per_token": tot_a / n_tok,
        "coef/kl_abs_mean_per_token": tot_k / n_tok,
        "coef/clip_frac": float(((clipped.float() * m).sum() / n_tok).item()),
        "coef/frac_tokens_kl_gt_reward": float((((kcoef.abs() > a.abs()).float() * m).sum() / n_tok).item()),
    }
    return out


# ---------------------------------------------------------------------------
# 3. 梯度向量工具（兼容 FSDP1 use_orig_params / FSDP2 DTensor / 普通模型）
# ---------------------------------------------------------------------------
def _local(g):
    return g.to_local() if hasattr(g, "to_local") else g


def grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        g = _local(p.grad).float()
        v = g.pow(2).sum()
        total = v if total is None else total + v
    if total is None:
        return 0.0
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    return float(total.sqrt().item())


def grad_snapshot(module):
    return [(_local(p.grad).detach().clone() if p.grad is not None else None) for p in module.parameters()]


def grad_diff_norm(module, snap):
    total = None
    for p, s in zip(module.parameters(), snap):
        if p.grad is None and s is None:
            continue
        g = _local(p.grad).float() if p.grad is not None else torch.zeros_like(s).float()
        d = g - (s.float() if s is not None else 0.0)
        v = d.pow(2).sum()
        total = v if total is None else total + v
    if total is None:
        return 0.0
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    return float(total.sqrt().item())


def grad_dot(module, snap):
    total = None
    for p, s in zip(module.parameters(), snap):
        if p.grad is None or s is None:
            continue
        v = (_local(p.grad).float() * s.float()).sum()
        total = v if total is None else total + v
    if total is None:
        return 0.0
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    return float(total.item())


def grad_zero(module):
    for p in module.parameters():
        p.grad = None


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------
class JsonlWriter:
    def __init__(self, path=None):
        self.path = path or os.environ.get("KLSTATS_OUT", "kl_grad_stats.jsonl")
        self.rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    def write(self, record):
        if self.rank != 0:
            return
        record = dict(record)
        record.setdefault("time", time.time())
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_rank0():
    return (not (dist.is_available() and dist.is_initialized())) or dist.get_rank() == 0
