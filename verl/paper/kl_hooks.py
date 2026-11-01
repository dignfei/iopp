# -*- coding: utf-8 -*-
"""
verl 0.9 钩子（单一代码路径，训练参数切换算法）：
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla  -> 普通 DAPO：裁剪 PG + 独立逐 token k3 KL 损失（GRPO 原版位置）
  actor_rollout_ref.actor.policy_loss.loss_mode=zcpo     -> ZCPO：C_i = A_i/s_q - β K_i（停止梯度，组内中心化的序列对数比），
                                                            裁剪与聚合都沿用基础算法（DAPO 的 dual-clip 与 loss_agg_mode，默认 token-mean），只把优势换成 C_i
  两种模式共用 use_kl_loss=True / kl_loss_coef=β（ZCPO 模式下 verl 不再额外加独立 KL 项，见 losses.py 补丁）。
  2026-09-22 起 ZCPO 不再用 seq-mean-token-sum-norm/T⋆ 聚合：论文写明"概率比处理、裁剪与聚合规则由基础算法提供"，
  旧实现写死了序列求和 /(|B|·T⋆)，与基线的 token-mean 只差一个随批变化的标量，但会改变梯度裁剪的触发与 Adam 的尺度。

接入点（engine_workers.py 与 losses.py 里标 [KL-EXP] 的补丁处）：
  1) engine_workers.TrainingWorker.train_mini_batch：每个 mini-batch 训练前调用 before_train_batch(self, mini_batch_td, batch_idx)
     - zcpo：用当前参数做一次无梯度前向，按 uid 分组算 C_i，写入 mini_batch_td["advantages"]（逐 token 常数）
     - KLSTATS：在每步第一个 mini-batch（优化器更新前，ratio≈1）分四次反向传播读各分支梯度范数
  2) losses.ppo_loss：zcpo 模式跳过独立 KL 项（仍记录 kl 指标）
  3) 本模块 import 时注册 "zcpo" 策略损失

诊断统计量（KLSTATS_ENABLE=1）的定义见 kl_grad_stats.py 模块说明与 paper/README.md 的"可选：复现附录 KL 梯度占比诊断"一节。
输出 jsonl: $KLSTATS_OUT（训练脚本设为 <运行目录>/kl_grad_stats.jsonl）
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kl_grad_stats import (  # noqa: E402
    JsonlWriter,
    _local,
    grad_norm,
    grad_zero,
    group_by_uid,
    kl_token_coef,
    shared_prefix_stats,
)

from verl.trainer.ppo.core_algos import agg_loss, kl_penalty, register_policy_loss  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.workers.utils.padding import no_padding_2_padding  # noqa: E402
import verl.utils.torch_functional as verl_F  # noqa: E402


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _env_flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def _loss_mode(config):
    try:
        return str(config.policy_loss.get("loss_mode", "vanilla"))
    except Exception:
        return "vanilla"


def _clip(config):
    low = config.clip_ratio_low if getattr(config, "clip_ratio_low", None) is not None else config.clip_ratio
    high = config.clip_ratio_high if getattr(config, "clip_ratio_high", None) is not None else config.clip_ratio
    return float(low), float(high), float(config.get("clip_ratio_c", 3.0))


def _uids(td):
    val = td.get("uid", None)
    if val is None:
        raise KeyError("mini-batch 缺少 uid，无法按组统计")
    if hasattr(val, "tolist"):
        val = val.tolist()
    return [str(x) for x in list(val)]


def _padded(td, *keys):
    return td.select(*keys).to_padded_tensor()


def _nested_like(ref_nested, per_row_values):
    """按 ref_nested（jagged, [B, L_i]）的每行长度，把每行标量展开成 nested tensor。"""
    lens = [t.shape[0] for t in ref_nested.unbind()]
    ts = [torch.full((L,), float(v), dtype=torch.float32) for L, v in zip(lens, per_row_values)]
    return tu.nested_tensor_from_tensor_list(ts)


def _nested_from_padded_bool(ref_nested, padded_bool):
    lens = [t.shape[0] for t in ref_nested.unbind()]
    ts = [padded_bool[i, :L].clone() for i, L in enumerate(lens)]
    return tu.nested_tensor_from_tensor_list(ts)


def _actor_config(worker):
    fn = getattr(worker, "loss_fn", None)
    if fn is not None and hasattr(fn, "keywords") and "config" in fn.keywords:
        return fn.keywords["config"]
    return worker.config


def _assign_engine_defaults(worker, td):
    """与 TrainingWorker.train_batch 相同的工程参数注入。"""
    defaults = dict(
        use_remove_padding=worker.model_config.get("use_remove_padding", False),
        use_dynamic_bsz=worker.engine_config.use_dynamic_bsz,
        max_token_len_per_gpu=worker.engine_config.max_token_len_per_gpu,
        micro_batch_size_per_gpu=worker.engine_config.micro_batch_size_per_gpu,
        use_fused_kernels=worker.engine_config.use_fused_kernels,
    )
    for k, v in defaults.items():
        if k not in td.keys():
            tu.assign_non_tensor(td, **{k: v})


class _CpuGradAccumulator:
    """梯度累加 buffer；KLSTATS_ACC_DEVICE=cuda（默认，快，约 2.4GB/份）或 cpu。"""

    def __init__(self, module):
        self.module = module
        self.buf = None
        self.device = os.environ.get("KLSTATS_ACC_DEVICE", "cuda")

    def add_current(self):
        grads = [(_local(p.grad).detach().float().to(self.device) if p.grad is not None else None) for p in self.module.parameters()]
        if self.buf is None:
            self.buf = grads
        else:
            for i, g in enumerate(grads):
                if g is None:
                    continue
                if self.buf[i] is None:
                    self.buf[i] = g
                else:
                    self.buf[i].add_(g)

    def norm(self):
        if self.buf is None:
            return 0.0
        s = None
        for b in self.buf:
            if b is None:
                continue
            v = b.double().pow(2).sum()
            s = v if s is None else s + v
        return float(s.sqrt().item()) if s is not None else 0.0

    def dot(self, other):
        if self.buf is None or other.buf is None:
            return 0.0
        s = None
        for a, b in zip(self.buf, other.buf):
            if a is None or b is None:
                continue
            v = (a.double() * b.double().to(a.device)).sum()
            s = v if s is None else s + v
        return float(s.item()) if s is not None else 0.0


# ---------------------------------------------------------------------------
# 1. ZCPO 策略损失（注册到 verl）
# ---------------------------------------------------------------------------
@register_policy_loss("zcpo")
def compute_policy_loss_zcpo(
    old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None, rollout_is_weights=None
):
    """advantages 中放的是停止梯度的联合系数 C_i（逐 token 常数，由 before_train_batch 写入）。
    裁剪（_vanilla_pg_mat，与 verl compute_policy_loss_vanilla 相同的 dual-clip）与聚合（传入的 loss_agg_mode 加
    config.global_batch_info，与 vanilla 的 agg_loss 调用逐字相同）都由基础算法提供，ZCPO 只替换优势。"""
    assert config is not None
    low, high, c = _clip(config)
    nak = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(nak)
    # 2026-09-11 用户定：ZCPO 不改裁剪，用 DAPO 默认的双裁剪（clip-higher 0.2/0.28 + dual-clip c=10），联合系数 C_i 当优势
    loss_mat = _vanilla_pg_mat(ratio, advantages, low, high, c)
    if rollout_is_weights is not None:
        loss_mat = loss_mat * rollout_is_weights
    # 与 verl core_algos.compute_policy_loss_vanilla 的聚合调用逐字相同：聚合规则由基础算法提供（论文"接入基础算法"一段）
    loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    _, gate_closed = _dapo_rho_and_closed(ratio, advantages, low, high, c)
    metrics = {
        "actor/pg_clipfrac": verl_F.masked_mean(gate_closed.float(), response_mask).detach().item(),
        "actor/ppo_kl": verl_F.masked_mean(-nak, response_mask).detach().item(),
        "actor/pg_clipfrac_lower": verl_F.masked_mean(((advantages < 0) & (ratio > c)).float(), response_mask).detach().item(),
        "zcpo/coef_abs_mean": verl_F.masked_mean(advantages.abs(), response_mask).detach().item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# 2. ZCPO 系数：mini-batch 级、当前参数、按 uid 分组
# ---------------------------------------------------------------------------
@torch.no_grad()
def _current_seq_logratio(worker, td):
    """返回 D_i = Σ_t (logπ_θ - logπ_ref)（当前参数），形状 [B]。"""
    B = td.batch_size[0]
    work = td.clone(recurse=False)
    work["klstats_rowid"] = torch.arange(B)
    _assign_engine_defaults(worker, work)
    store = {}

    def collector(model_output, data, dp_group=None):
        lp = no_padding_2_padding(model_output["log_probs"], data)
        d = _padded(data, "response_mask", "ref_log_prob")
        m = d["response_mask"].to(lp.device).to(torch.bool)
        ref = d["ref_log_prob"].to(lp.device)
        D = ((lp - ref) * m).sum(-1).float().cpu()
        dv = ref - lp
        k3 = (torch.clamp(torch.exp(dv) - dv - 1.0, -10.0, 10.0) * m).sum(-1).float().cpu()
        L = m.sum(-1).float().cpu()
        for rid, val, kv, lv in zip(data["klstats_rowid"].tolist(), D.tolist(), k3.tolist(), L.tolist()):
            store[int(rid)] = (val, kv, lv)
        return torch.zeros((), device=lp.device), {}

    worker.engine.forward_backward_batch(work, loss_function=collector, forward_only=True)
    D = torch.tensor([store[i][0] for i in range(B)], dtype=torch.float32)
    k3s = torch.tensor([store[i][1] for i in range(B)], dtype=torch.float32)
    Ls = torch.tensor([store[i][2] for i in range(B)], dtype=torch.float32)
    return D, k3s, Ls



def _position_kl_and_logps(logits_a, logits_r, target_ids, chunk=1024, W_a=None, W_r=None, temperature=1.0):
    """逐位置：κ_t = KL(softmax(a_t) || softmax(r_t))（对整个词表精确求和），以及采到 token 的 logp_a、logp_r。
    两种输入：(i) logits_a/logits_r: [L, V]；(ii) 给 W_a/W_r（[V, H] fp32）时 logits_a/logits_r 是隐藏状态 [L, H]，
    逐块用 fp32 矩阵乘得到 logits（避免 bf16 logits 的舍入噪声：bf16 logits 会给 KL 带来 ~1e-3/token 的噪声地板）。
    返回 (kappa[L], lpa[L], lpr[L]) fp32。"""
    L = logits_a.shape[0]
    kap = torch.empty(L, dtype=torch.float32, device=logits_a.device)
    lpa_t = torch.empty_like(kap); lpr_t = torch.empty_like(kap)
    for c0 in range(0, L, chunk):
        c1 = min(L, c0 + chunk)
        za = logits_a[c0:c1].float() @ W_a.t() if W_a is not None else logits_a[c0:c1].float()
        zr = logits_r[c0:c1].float() @ W_r.t() if W_r is not None else logits_r[c0:c1].float()
        if temperature != 1.0:   # 策略与参考的分布都是 softmax(logits/T)，与引擎算 logp 时一致
            za = za / temperature; zr = zr / temperature
        la = torch.log_softmax(za, dim=-1); lr = torch.log_softmax(zr, dim=-1)
        kap[c0:c1] = (la.exp() * (la - lr)).sum(-1)
        idx = target_ids[c0:c1].unsqueeze(-1)
        lpa_t[c0:c1] = la.gather(-1, idx).squeeze(-1)
        lpr_t[c0:c1] = lr.gather(-1, idx).squeeze(-1)
        del za, zr, la, lr
    return kap, lpa_t, lpr_t


@torch.no_grad()
def _exact_seq_kl(worker, td, micro=None):
    """当前参数下，逐位置精确条件 KL 的序列和 D^ex_i = Σ_t KL(π_θ(·|h_t) || π_ref(·|h_t))，同时给出采样版 D_i、Σk3_i、T_i。
    actor 与 ref 各做一次不带梯度的裸前向；用 lm_head 的 forward hook 抓最后隐藏状态与（前向期间已 unshard 的）lm_head 权重，
    在 fp32 里逐块重算 logits——直接用模型输出的 bf16 logits 会有 ~2.7e-3/token 的 KL 噪声地板（2026-09-11 实测）。
    需要 worker._kl_ref_worker（engine_workers 补丁挂上）。返回 dict(D_exact, D, k3, L, kappa, ref_lp_absdiff)；ref 不可用时返回 None。"""
    import numpy as np
    ref_w = getattr(worker, "_kl_ref_worker", None)
    if ref_w is None:
        return None
    if int(getattr(worker.engine, "ulysses_sequence_parallel_size", 1) or 1) > 1:
        # 本函数自己调 engine.module 做裸前向，喂的是完整序列，而序列并行的注意力 all-to-all 期望的是切片，两者不匹配
        print("[ZCPO] 序列并行下不能用独立重算路径（_exact_seq_kl），请用合并 pass（ZCPO_COMBINED=1）", flush=True)
        return None
    micro = int(micro or os.environ.get("ZCPO_KL_MICRO", os.environ.get("KLSTATS_EXACT_MICRO", "2")))
    chunk = int(os.environ.get("ZCPO_KL_CHUNK", os.environ.get("KLSTATS_EXACT_CHUNK", "1024")))
    # 有效长度按 verl 自己的算法取（attention_mask），token 序列用去填充的嵌套 input_ids（有效提示 + 回复）
    P_valid, L_valid = _seq_lens(td)
    if "input_ids" in td.keys() and td["input_ids"].is_nested:
        full = [t.cpu() for t in td["input_ids"].unbind()]
        pos_list = [t.cpu() for t in td["position_ids"].unbind()] if "position_ids" in td.keys() and td["position_ids"].is_nested else None
    else:  # padded 布局：prompts 左填充、responses 右填充
        pr, rs = td["prompts"], td["responses"]
        full = [torch.cat([pr[i].cpu()[pr.shape[1] - P_valid[i]:], rs[i].cpu()[: L_valid[i]]]) for i in range(len(P_valid))]
        pos_list = None
    P_list = [f[: f.numel() - l] for f, l in zip(full, L_valid)]
    R_list = [f[f.numel() - l:] for f, l in zip(full, L_valid)]
    for f, p, l in zip(full, P_valid, L_valid):
        assert f.numel() == p + l, f"length mismatch: input_ids {f.numel()} vs P+L {p}+{l}"
    B = len(P_list)
    dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    D_ex = torch.zeros(B); D_s = torch.zeros(B); K3 = torch.zeros(B); Ls = torch.zeros(B); kappa = {}
    pad_id = 0
    ref_lp_engine = _padded(td, "ref_log_prob")["ref_log_prob"].float() if "ref_log_prob" in td.keys() else None
    absdiff = []

    def _lm_head(module):
        for name in ("lm_head", "output", "embed_out"):
            m = getattr(module, name, None)
            if m is not None:
                return m
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features > 30000:
                return m
        raise RuntimeError("lm_head not found")

    def _base_model(module):
        """lm_head 之前的基座模型（Qwen3ForCausalLM.model 等）；FSDP 包装下属性访问会透传到被包装模块。"""
        for name in ("model", "transformer", "base_model"):
            m = getattr(module, name, None)
            if m is not None and hasattr(m, "layers"):
                return m
        return None

    def _forward_hidden(engine, ids, att, pos, in_training=False):
        """返回 (最后隐藏状态 [M,T,H] bf16, lm_head 权重 [V,H] fp32)。
        in_training=True 用于策略引擎：本函数在 before_train_batch 里被调用，此时策略正处于训练上下文（参数在 GPU、模块 train()）。
        不能再套 engine.eval_mode()——它退出时会按参数卸载配置把参数挪回 CPU、并把 engine.mode 置空，紧接着的训练步报
        "Expects tensor to be on the compute device"（2026-09-22 实测，param_offload=True 时必现）。参考引擎不训练，照常用 eval_mode 装卸参数。
        2026-09-16：思考模式序列到 2 万 token，外层前向的 [M,T,V] logits 单条就 6.7 GB（bf16）。改为钩在基座模型的输出上拿隐状态，
        并传 logits_to_keep=1 让 lm_head 只算最后一个位置；不支持该参数的实现回退到原来的 lm_head 输入钩子。"""
        cap = {}
        lm = _lm_head(engine.module)
        base = _base_model(engine.module)
        def hook_lm(mod, inp, out):
            cap.setdefault("h", inp[0].detach())
            cap["W"] = _local(mod.weight).detach().float().clone()   # 必须在前向期间抓（FSDP 已 unshard）；前向之外读分片参数会非法访存（2026-09-18 冒烟）
        def hook_base(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
            cap["h"] = h.detach()
        handles = [lm.register_forward_hook(hook_lm)]
        if base is not None:
            handles.append(base.register_forward_hook(hook_base))
        def _run():
            try:
                if base is None:
                    raise TypeError("no base model handle")
                engine.module(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=1)
            except TypeError:
                cap.pop("h", None)
                engine.module(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False)
        try:
            if in_training:
                was_training = engine.module.training
                engine.module.eval()
                try:
                    _run()
                finally:
                    engine.module.train(was_training)
            else:
                with engine.eval_mode():
                    _run()
        finally:
            for hd in handles:
                hd.remove()
        return cap["h"], cap["W"]

    tok_budget = int(os.environ.get("ZCPO_KL_TOKENS", os.environ.get("KLSTATS_EXACT_TOKENS", "16384")))   # 每次裸前向的 M×T 上限（2026-09-16：思考模式长序列按 token 预算分组）
    groups = []; cur = []
    for i in range(B):
        T_i = int(P_list[i].numel() + R_list[i].numel())
        if cur and (len(cur) >= micro or (max(int(P_list[j].numel() + R_list[j].numel()) for j in cur + [i]) * (len(cur) + 1)) > tok_budget):
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    for rows in groups:
        seqs = [torch.cat([P_list[i], R_list[i]]) for i in rows]
        T = max(int(t.numel()) for t in seqs)
        ids = torch.full((len(rows), T), pad_id, dtype=torch.long); att = torch.zeros((len(rows), T), dtype=torch.long)
        for r, t in enumerate(seqs):
            ids[r, : t.numel()] = t; att[r, : t.numel()] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        if pos_list is not None:
            for r, i in enumerate(rows):
                pl = pos_list[i]
                if pl.dim() == 1 and pl.numel() == int(seqs[r].numel()):
                    pos[r, : pl.numel()] = pl
        ids, att, pos = ids.to(dev), att.to(dev), pos.to(dev)
        h_a, W_a = _forward_hidden(worker.engine, ids, att, pos, in_training=True)
        h_r, W_r = _forward_hidden(ref_w.engine, ids, att, pos)
        if h_a.dim() == 2:  # 某些实现返回打包布局 [M*T, H]
            h_a = h_a.view(len(rows), T, -1); h_r = h_r.view(len(rows), T, -1)
        for r, i in enumerate(rows):
            P, L = int(P_list[i].numel()), int(R_list[i].numel())
            if L == 0:
                continue
            tgt = R_list[i].to(dev)
            kap, lpa, lpr = _position_kl_and_logps(h_a[r, P - 1 : P - 1 + L], h_r[r, P - 1 : P - 1 + L], tgt, chunk, W_a=W_a, W_r=W_r, temperature=_row_temperature(td, i))
            dlt = lpa - lpr
            D_ex[i] = float(kap.sum()); D_s[i] = float(dlt.sum())
            K3[i] = float(torch.clamp(torch.exp(-dlt) + dlt - 1.0, -10.0, 10.0).sum()); Ls[i] = L
            kappa[i] = kap.cpu().numpy()
            if ref_lp_engine is not None:
                absdiff.append(float((lpr.cpu() - ref_lp_engine[i, :L]).abs().mean()))
        del h_a, h_r, W_a, W_r
        torch.cuda.empty_cache()
    return dict(D_exact=D_ex, D=D_s, k3=K3, L=Ls, kappa=kappa, ref_lp_absdiff=(sum(absdiff) / len(absdiff)) if absdiff else float("nan"))



# ---------------------------------------------------------------------------
# 2b. 合并推理 pass（2026-09-16）：在策略的 old_log_prob 前向回调里，对同一 micro-batch 立刻跑参考基座前向，
#     两侧隐状态按位置分块做 fp32 lm_head 投影 → κ 累加到回复级 D_i，参考 logp 一并缓存；驱动端的 ref pass 直接取缓存。
#     每个 token 仍只有策略一次、参考一次前向；不跨 pass 缓存任何 token×词表张量。开关 ZCPO_COMBINED=1（仅 loss_mode=zcpo 且 ZCPO_KL_KIND=exact）。
# ---------------------------------------------------------------------------
def _lm_head(module):
    """输出头模块（Qwen3ForCausalLM.lm_head 等）；找不到命名属性时取词表大小的最后一个 Linear。"""
    for name in ("lm_head", "output", "embed_out"):
        m = getattr(module, name, None)
        if m is not None:
            return m
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features > 30000:
            return m
    raise RuntimeError("lm_head not found")


def _base_model(module):
    """lm_head 之前的基座模型（Qwen3ForCausalLM.model 等）；FSDP 包装下属性访问透传到被包装模块。"""
    for name in ("model", "transformer", "base_model"):
        m = getattr(module, name, None)
        if m is not None and hasattr(m, "layers"):
            return m
    return None


def _row_key(ids):
    """整条序列（提示+回复）的指纹，用于把合并 pass 的缓存对上 update 阶段的 mini-batch 行。
    2026-09-22 改为全序列哈希：旧版只取长度+首 48+末 16 个 token，同一提示的两条回复若开头、结尾、长度都相同会撞键、D_i 被互换。
    现在只有输入完全相同的两行才同键，而相同输入的 D_i 由同一次确定性前向给出、本就相等。"""
    import hashlib
    t = ids.detach().cpu().to(torch.int64).contiguous()
    return (int(t.numel()), hashlib.blake2b(t.numpy().tobytes(), digest_size=16).hexdigest())


def combined_enabled(hybrid):
    try:
        actor = getattr(hybrid, "actor", None); ref = getattr(hybrid, "ref", None)
        if actor is None or ref is None or not _env_flag("ZCPO_COMBINED", "0"):
            return False
        if not (_loss_mode(_actor_config(actor)) == "zcpo" and os.environ.get("ZCPO_KL_KIND", "exact") == "exact"):
            return False
        # 融合推理按"去填充后打包成一条"的布局切片（隐状态为 [1,total_nnz,H]）。填充布局下隐状态是 [B,T,H]，
        # 按全局偏移会切到批维、取到空张量，D_i 静默为 0（2026-09-22 实测）。此时改用独立重算路径。
        mc = getattr(actor, "model_config", None) or {}
        rmpad = bool(mc.get("use_remove_padding", False)) if hasattr(mc, "get") else bool(getattr(mc, "use_remove_padding", False))
        if not rmpad:
            if not getattr(hybrid, "_zcpo_layout_warned", False):
                print("[ZCPO-COMBINED] 融合推理只支持去填充打包布局（use_remove_padding=True）；当前为填充布局，改用独立重算路径", flush=True)
                hybrid._zcpo_layout_warned = True
            return False
        return True
    except Exception as e:  # noqa
        print(f"[ZCPO-COMBINED] enable check failed: {e!r}", flush=True)
        return False


def _row_temperature(td, i):
    """该行的采样温度：verl 把 rollout.temperature 放在 batch 的非张量键 temperature 里（标量），支持 per-sample 时为 [B] 张量。取不到按 1.0。"""
    try:
        t = tu.get(td, key="temperature", default=1.0)
    except Exception:
        t = td.get("temperature", 1.0) if hasattr(td, "get") else 1.0
    if isinstance(t, torch.Tensor):
        t = t.flatten()
        return float(t[i]) if t.numel() > 1 else float(t[0])
    return float(t)


def _sp_window(g0, g1, base, win_hi):
    """某条回复的全局位置区间 [g0,g1) 与本秩窗口 [base,win_hi) 的交集；返回 (a0,a1)，空交集时 a1<=a0。"""
    return max(g0, base), min(g1, win_hi)


def _sp_info(engine):
    """序列并行的 (sp_size, 进程组, 组内秩)。未启用时返回 (1, None, 0)。
    进程组直接读引擎上的 ulysses_parallel_group（transformer_impl 建 device mesh 时存的，稳定不变），
    不用 verl 的全局变量：那个全局量由 sharding manager 的上下文设置，嵌套调用参考模型前向后会被还原，
    拿到 None 就会退化成全局组，数据并行下两组的集合通信次数对不上，直接死锁（2026-09-22 实测）。"""
    sp = int(getattr(engine, "ulysses_sequence_parallel_size", 1) or 1)
    if sp <= 1:
        return 1, None, 0
    g = getattr(engine, "ulysses_parallel_group", None)
    if g is None:
        mesh = getattr(engine, "ulysses_device_mesh", None)
        g = mesh["sp"].get_group() if mesh is not None else None
    if g is None:
        raise RuntimeError("序列并行已启用但取不到 sp 进程组；拒绝退化到全局组")
    return sp, g, int(dist.get_rank(group=g))


def _sp_allreduce_rows(cache, rid, group, device):
    """把本微批各行的 D_exact / D_k1 / k3 分段和在序列并行组内求和，写回缓存。"""
    idx = [int(r) for r in rid]
    for k in ("D_exact", "D_k1", "k3", "cov"):
        t = torch.tensor([float(cache[k].get(i, 0.0)) for i in idx], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        for n, i in enumerate(idx):
            cache[k][i] = float(t[n])


def _sp_selftest(cache, rid, sp, group, h_a, h_r, W_a, W_r, ids, offs, P_, L_, chunk, data_mb=None):
    """序列并行自检：all-gather 隐状态后按非并行方式重算 D_i，与分片+all-reduce 的结果比对。仅调试用。"""
    ga = [torch.empty_like(h_a) for _ in range(sp)]
    gr = [torch.empty_like(h_r) for _ in range(sp)]
    dist.all_gather(ga, h_a.contiguous(), group=group)
    dist.all_gather(gr, h_r.contiguous(), group=group)
    full_a = torch.cat(ga, dim=0); full_r = torch.cat(gr, dim=0)
    worst = 0.0; worst_row = None; n = 0
    for j, r in enumerate(rid):
        r = int(r); P, L = int(P_[j]), int(L_[j])
        if L <= 0:
            continue
        o0 = int(offs[j]) if offs is not None else j * ids.shape[1]
        seq = ids[j] if offs is not None else ids[j][: P + L]
        tgt = seq[P: P + L].to(full_a.device)
        g0 = o0 + P - 1
        with torch.autocast(device_type="cuda", enabled=False):
            kap, _, _ = _position_kl_and_logps(full_a[g0: g0 + L], full_r[g0: g0 + L], tgt, chunk, W_a=W_a, W_r=W_r, temperature=_row_temperature(data_mb, j) if data_mb is not None else 1.0)
        ref = float(kap.double().sum()); got = float(cache["D_exact"][r]); n += 1
        d = abs(got - ref)
        if d > worst:
            worst, worst_row = d, (r, got, ref, L)
    rel = worst / max(1e-12, abs(worst_row[2])) if worst_row else 0.0
    print(f"[ZCPO-SP-SELFTEST] sp={sp} 比对 {n} 行：D_i 最大绝对差={worst:.3e} 相对差={rel:.3e} 最差行={worst_row}", flush=True)


@torch.no_grad()
def combined_log_prob(hybrid, data):
    """替代 ActorRolloutRefWorker.compute_log_prob 的主体。返回与 actor.infer_batch 相同的输出；副作用：actor._zcpo_combined 缓存。"""
    import numpy as np
    actor, refw = hybrid.actor, hybrid.ref
    B = int(data.batch_size[0])
    old_cl = tu.get(data, key="compute_loss", default=True)
    chunk = int(os.environ.get("ZCPO_KL_CHUNK", os.environ.get("KLSTATS_EXACT_CHUNK", "1024")))
    cache = {"B": B, "D_exact": {}, "D_k1": {}, "k3": {}, "L": {}, "cov": {}, "kappa": {}, "ref_lp": {}, "keys": {}, "absdiff_ref": [], "absdiff_act": [], "t_kl": 0.0}
    cap = {}
    base_a = _base_model(actor.engine.module); base_r = _base_model(refw.engine.module)
    lm_a = _lm_head(actor.engine.module); lm_r = _lm_head(refw.engine.module)
    if base_a is None or base_r is None:
        raise RuntimeError("base model not found for combined pass")
    def _hook_h(name):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
            cap[name] = h.detach()
        return f
    def _hook_w(name):
        def f(mod, inp, out):
            if name not in cap:
                cap[name] = _local(mod.weight).detach().float().clone()   # 前向期间已 unshard
        return f
    handles = [base_a.register_forward_hook(_hook_h("h_a")), base_r.register_forward_hook(_hook_h("h_r")),
               lm_a.register_forward_hook(_hook_w("W_a")), lm_r.register_forward_hook(_hook_w("W_r"))]
    dev0 = next(actor.engine.module.parameters()).device

    def ref_cb(model_output, data=None, dp_group=None, **kw):
        data_mb = data
        lp = model_output["log_probs"]
        rid = data_mb["zcpo_rowid"].tolist()
        for j, r in enumerate(rid):
            cache["ref_lp"][int(r)] = lp[j].detach().float().cpu()
        return torch.tensor(0.0, device=dev0), {}

    def actor_cb(model_output, data=None, dp_group=None, **kw):
        data_mb = data
        t0 = time.time()
        h_a = cap.pop("h_a")
        sp, sp_group, sp_rank = _sp_info(actor.engine)   # 必须在嵌套的参考前向之前取，避免受其上下文影响
        # 参考基座前向：同一 micro-batch、同一打包顺序（forward_step 不重排行）
        refw.engine.forward_step(data_mb, loss_function=ref_cb, forward_only=True)
        h_r = cap.pop("h_r")
        W_a, W_r = cap["W_a"], cap["W_r"]
        if h_a.dim() == 3 and h_a.shape[0] == 1:
            h_a = h_a[0]; h_r = h_r[0]          # 打包布局 [total_nnz, H]
        ids = data_mb["input_ids"]
        offs = ids.offsets().tolist() if ids.is_nested else None
        P_, L_ = _seq_lens(data_mb)
        rid = data_mb["zcpo_rowid"].tolist()
        lp_engine = model_output["log_probs"]
        # 序列并行：verl 把打包序列 [1, total_nnz] 末尾补齐到 sp 的整数倍后按位置连续切片，
        # 本秩只持有 [sp_rank*parts, (sp_rank+1)*parts)（verl/utils/ulysses.py: ulysses_pad + slice_input_tensor）。
        # κ 是逐位置量、D_i 是求和，所以每秩只算落在本窗口内的位置，回合后对 D_i 做一次组内 all-reduce 即可，
        # 不需要 gather 隐状态。引擎给出的 log_probs 已在 SP 组内 all-gather 过，是整条序列，自检时按全局下标取。
        base = sp_rank * int(h_a.shape[0]) if sp > 1 else 0
        win_hi = base + int(h_a.shape[0])
        for j, r in enumerate(rid):
            r = int(r); P, L = int(P_[j]), int(L_[j])
            if offs is not None:
                o0 = int(offs[j]); seq = ids[j]
            else:
                o0 = j * ids.shape[1]; seq = ids[j][: P + L]
            cache["keys"][r] = _row_key(seq); cache["L"][r] = max(L, 0); cache["cov"][r] = 0   # 指纹用整条序列
            if L <= 0:
                cache["D_exact"][r] = 0.0; cache["D_k1"][r] = 0.0; cache["k3"][r] = 0.0; cache["L"][r] = 0
                cache["kappa"][r] = np.zeros(0, np.float32); continue
            g0, g1 = o0 + P - 1, o0 + P - 1 + L          # 该回复在全局打包序列里的位置区间
            a0, a1 = _sp_window(g0, g1, base, win_hi)    # 与本秩窗口求交
            if a1 <= a0:                                  # 本秩不含该回复的任何位置
                cache["D_exact"][r] = 0.0; cache["D_k1"][r] = 0.0; cache["k3"][r] = 0.0
                cache["kappa"][r] = np.zeros(0, np.float32); continue
            tgt = seq[P + (a0 - g0): P + (a1 - g0)].to(h_a.device)
            ha = h_a[a0 - base: a1 - base]; hr = h_r[a0 - base: a1 - base]
            if ha.shape[0] != a1 - a0 or hr.shape[0] != a1 - a0 or int(seq.numel()) != P + L:
                print(f"[ZCPO-COMBINED] SLICE MISMATCH row={r} j={j} o0={o0} P={P} L={L} seq={int(seq.numel())} sp={sp} base={base} win=[{a0},{a1}) h_a={tuple(h_a.shape)}；本步缓存作废，改用独立重算", flush=True)
                cache["slice_bad"] = True
            # 回调跑在引擎 forward_step 的 autocast(bf16) 里（纲要 transformer_impl.py），必须关掉 autocast 才是真 fp32 投影
            with torch.autocast(device_type="cuda", enabled=False):
                kap, lpa, lpr = _position_kl_and_logps(ha, hr, tgt, chunk, W_a=W_a, W_r=W_r, temperature=_row_temperature(data_mb, j))
            dlt = lpa - lpr
            # 用 float64 累加：分片后求和顺序变了，fp32 累加会带来 1e-6 量级的差异；float64 下分片与否的结果一致到 1e-15
            cache["D_exact"][r] = float(kap.double().sum()); cache["D_k1"][r] = float(dlt.double().sum())
            cache["k3"][r] = float(torch.clamp(torch.exp(-dlt) + dlt - 1.0, -10.0, 10.0).double().sum()); cache["cov"][r] = a1 - a0
            cache["kappa"][r] = kap.float().cpu().numpy() if sp == 1 else np.zeros(0, np.float32)   # SP>1 时逐 token κ 分散在各秩，诊断量不再收集
            # 自检：我们的 fp32 logp 与引擎 logp（策略/参考）在本秩覆盖的位置上的平均绝对差；错位会立刻放大
            try:
                lo = (P - 1) + (a0 - g0); hi = lo + (a1 - a0)
                cache["absdiff_act"].append(float((lpa.cpu() - lp_engine[j].detach().float().cpu()[lo:hi]).abs().mean()))
                cache["absdiff_ref"].append(float((lpr.cpu() - cache["ref_lp"][r][lo:hi]).abs().mean()))
            except Exception:
                pass
        if sp > 1:   # 把各秩的分段和加起来，得到整条回复的 D_i / Σk3；cov 用来验证每个位置被且只被一个秩算过
            _sp_allreduce_rows(cache, rid, sp_group, h_a.device)
            if _env_flag("ZCPO_SP_SELFTEST", "0"):
                # 正确性对照：把隐状态在组内 all-gather 成整条序列，完全按 sp=1 的方式重算一遍 D_i，与分片求和的结果逐行比对。
                # 同一次前向、同一批数据，差异只可能来自序列并行的记账，因此应为浮点误差量级。
                _sp_selftest(cache, rid, sp, sp_group, h_a, h_r, W_a, W_r, ids, offs, P_, L_, chunk, data_mb=data_mb)
            bad = [(int(r), cache["cov"][int(r)], cache["L"][int(r)]) for r in rid if cache["cov"][int(r)] != cache["L"][int(r)]]
            if bad:
                print(f"[ZCPO-SP] 覆盖校验失败（行, 各秩合计覆盖位置数, 应为 L）: {bad[:5]}", flush=True)
            else:
                cache["sp_cov_ok"] = cache.get("sp_cov_ok", 0) + 1
        cache["t_kl"] += time.time() - t0
        return torch.tensor(0.0, device=h_a.device), {}

    old_fn = getattr(actor, "loss_fn", None)
    try:
        # batch 标志只在这里改，finally 里恢复：失败回退到普通 actor pass 时不能带着 compute_loss=True（会调 ppo_loss 并要 global_batch_size）
        data["zcpo_rowid"] = torch.arange(B)
        tu.assign_non_tensor(data, compute_loss=True)
        actor.set_loss_fn(actor_cb)
        with refw.engine.eval_mode():
            output = actor.infer_batch(data)
    finally:
        actor.set_loss_fn(old_fn)
        for hd in handles:
            hd.remove()
        cap.clear()
        tu.assign_non_tensor(data, compute_loss=old_cl)
        if "zcpo_rowid" in data.keys():
            del data["zcpo_rowid"]
    n_ok = len(cache["D_exact"])
    cache["complete"] = (n_ok == B and len(cache["ref_lp"]) == B and not cache.get("slice_bad", False))
    if not cache["complete"]:
        print(f"[ZCPO-COMBINED] incomplete cache: D {n_ok}/{B}, ref_lp {len(cache['ref_lp'])}/{B}; ref pass and exact KL will fall back", flush=True)
    else:
        aa = cache["absdiff_act"]; ar = cache["absdiff_ref"]
        sp_note = f" sp覆盖校验通过={cache.get('sp_cov_ok', 0)}次" if cache.get("sp_cov_ok") else ""
        print(f"[ZCPO-COMBINED] rows={B} kl_time={cache['t_kl']:.1f}s |logp_ours-engine| actor={np.mean(aa) if aa else float('nan'):.2e} ref={np.mean(ar) if ar else float('nan'):.2e}{sp_note}", flush=True)
    if _env_flag("ZCPO_SP_DEBUG", "0"):   # 序列并行正确性验证：按回复指纹落盘每行的 D_i，供 sp=1 与 sp>1 两次运行逐行比对
        import json as _json
        rk = dist.get_rank() if dist.is_initialized() else 0
        out = os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), f"sp_debug_rank{rk}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            for r in sorted(cache["D_exact"]):
                k = cache["keys"].get(r)
                f.write(_json.dumps({"key": str(k), "L": int(cache["L"].get(r, 0)), "D_exact": cache["D_exact"][r],
                                     "D_k1": cache["D_k1"][r], "k3": cache["k3"][r]}) + "\n")
    actor._zcpo_combined = cache
    hybrid._zcpo_combined = cache
    return output


def combined_ref_output(hybrid, data):
    """驱动端 ref pass：合并 pass 已算过就直接返回缓存（与 ref.infer_batch 输出同构：嵌套 log_probs + metrics）。"""
    c = getattr(hybrid, "_zcpo_combined", None)
    if not c or not c.get("complete") or int(data.batch_size[0]) != c["B"]:
        return None
    try:
        ids = data["input_ids"]
        lens = [int(t.shape[0]) for t in ids.unbind()] if ids.is_nested else [int(ids.shape[1])] * c["B"]
        rows = []
        for i in range(c["B"]):
            lp = c["ref_lp"][i]
            if int(lp.numel()) != lens[i]:
                print(f"[ZCPO-COMBINED] ref cache length mismatch at row {i}: {lp.numel()} vs {lens[i]}; falling back", flush=True)
                return None
            rows.append(lp)
        nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
        out = tu.get_tensordict(tensor_dict={"log_probs": nested}, non_tensor_dict={"metrics": {"mfu": 0.0, "zcpo_combined": 1.0}})
        return out
    except Exception as e:  # noqa
        print(f"[ZCPO-COMBINED] ref cache assembly failed: {e!r}; falling back", flush=True)
        return None


def _combined_lookup(worker, td):
    """update 阶段：按回复指纹把 mini-batch 的行对到合并 pass 的缓存；全部命中才用。返回 ex dict 或 None。"""
    import numpy as np
    c = getattr(worker, "_zcpo_combined", None)
    if not c or not c.get("complete"):
        return None
    try:
        P_, L_ = _seq_lens(td)
        ids = td["input_ids"]
        seqs = [t for t in ids.unbind()] if ids.is_nested else [ids[i][: P_[i] + L_[i]] for i in range(len(P_))]
        inv = {}
        for r, k in c["keys"].items():
            inv.setdefault(k, []).append(r)
        B = len(seqs); D_ex = torch.zeros(B); D_s = torch.zeros(B); K3 = torch.zeros(B); Ls = torch.zeros(B); kappa = {}
        used = set()
        for i in range(B):
            k = _row_key(seqs[i][: P_[i] + L_[i]])   # 与合并 pass 一致：整条序列（提示+回复）
            cands = [r for r in inv.get(k, []) if r not in used]
            if not cands:
                return None
            r = cands[0]; used.add(r)
            D_ex[i] = c["D_exact"][r]; D_s[i] = c["D_k1"][r]; K3[i] = c["k3"][r]; Ls[i] = c["L"][r]; kappa[i] = c["kappa"][r]
        return dict(D_exact=D_ex, D=D_s, k3=K3, L=Ls, kappa=kappa, ref_lp_absdiff=float(np.mean(c["absdiff_ref"])) if c["absdiff_ref"] else float("nan"), combined=True)
    except Exception as e:  # noqa
        print(f"[ZCPO-COMBINED] lookup failed: {e!r}; recomputing", flush=True)
        return None


@torch.no_grad()
def zcpo_group_coefficients(R, D, uids, beta, L=None):
    """纯函数：R[B] 序列奖励，D[B] 当前参数下的序列 KL 量（ZCPO_KL_KIND=exact 时为 Σ_t κ_t 精确条件 KL，k1 时为采样对数比），L[B] 有效回答 token 数，uids 分组。返回 (A/s_q, -βK, active)。
    论文式(scale)/(C)（2026-09-11 新版，见 新版kl公式.md）：A_i=R_i-mean, s_q=max|A_i|, B_q=1{G>=2, s_q>0}；
    D̃_i=D_i/T_i（先序列内长度平均），K_i=D̃_i-mean_g(D̃)（再组内中心化）；C_i=A_i/s_q-βK_i；非活跃组在除法前跳过。
    L=None 时退化为旧版（不除长度），仅供对照。"""
    B = R.shape[0]
    R = R.double()
    D = D.double()
    if L is not None:
        D = D / L.double().clamp(min=1.0)
    A_over_s = torch.zeros(B, dtype=torch.float64)
    negbetaK = torch.zeros(B, dtype=torch.float64)
    active = torch.zeros(B, dtype=torch.bool)
    for _, rows in group_by_uid(uids).items():
        idx = torch.tensor(rows)
        if len(rows) < 2:
            continue
        Rg = R[idx]
        A = Rg - Rg.mean()
        s = A.abs().max()
        if s <= 0:
            continue  # B_q = 0：整组跳过（在除法之前）
        Dg = D[idx]
        K = Dg - Dg.mean()
        A_over_s[idx] = A / s
        negbetaK[idx] = -beta * K
        active[idx] = True
    return A_over_s.float(), negbetaK.float(), active


def _check_groups_intact(worker, uids):
    """组均值、s_q、K_i 都在组内算，一组 G 条回复必须完整落在同一个 mini-batch 里。
    多卡数据并行按行数把批切块，若块边界落在组中间就会静默算错，所以这里硬校验：
    每个 uid 组的行数必须等于 rollout.n（engine_workers.init_model 挂在 actor._zcpo_group_size 上）；
    拿不到 n 时退而要求各组行数一致。不满足直接报错，不允许带着错的系数继续训练。"""
    sizes = {u: len(rows) for u, rows in group_by_uid(uids).items()}
    if not sizes:
        return
    n = getattr(worker, "_zcpo_group_size", None)
    if n is not None:
        bad = {u: c for u, c in sizes.items() if c != int(n)}
        if bad:
            raise RuntimeError(f"ZCPO 组被拆开：期望每组 {int(n)} 条回复，实际 {dict(list(bad.items())[:4])}（共 {len(bad)} 组）。"
                               "请检查 trainer.balance_batch=False、actor.shuffle=False，且 train_batch_size*n 能被数据并行秩数整除、ppo_mini_batch_size*n 能被 n 整除")
    elif len(set(sizes.values())) > 1:
        raise RuntimeError(f"ZCPO 组大小不一致：{sorted(set(sizes.values()))}，说明组被数据并行切分拆开了")


@torch.no_grad()
def zcpo_coefficients(worker, td, config, uids):
    beta = float(config.kl_loss_coef)
    key = "token_level_rewards" if "token_level_rewards" in td.keys() else "token_level_scores"
    R = _padded(td, key)[key].float().sum(-1)  # 序列总奖励（DAPO：正确性 ±1 + 超长软惩罚）
    kind = os.environ.get("ZCPO_KL_KIND", "exact")   # exact: 逐位置精确条件 KL（新版 kl 公式）；k1: 采样对数比（旧版）
    ex = None
    if kind == "exact":
        ex = _combined_lookup(worker, td)
        if ex is not None and _env_flag("ZCPO_COMBINED_VERIFY", "0"):
            # 合并 pass 的 D_i 是在旧策略（old_log_prob 那次前向的 θ，即本步更新前）算的，整步各 mini-batch 共用；
            # 旧路径在每个 mini-batch 前用当前 θ 重算。两者只在本步第一个 mini-batch（尚未更新）可逐位比对；
            # 之后的 mini-batch 差异是步内参数漂移（第 1 步 lr 预热为 0，所以第 1 步全部一致；2026-09-18 冒烟核实）。
            ex2 = _exact_seq_kl(worker, td)
            if ex2 is not None:
                dmax = float((ex["D_exact"] - ex2["D_exact"]).abs().max()); rel = float(((ex["D_exact"] - ex2["D_exact"]).abs() / ex2["D_exact"].abs().clamp(min=1e-6)).max())
                tag = "VERIFY" if int(getattr(worker, "_zcpo_batch_idx", 0)) == 0 else "DRIFT(within-step, not a check)"
                # 两条路径各自的 fp32 logp 与引擎 logp 的平均绝对差：融合路径复用引擎那次前向（去填充打包布局），
                # 独立重算另起一次裸前向（填充布局）。若后者明显更大，说明差异来自前向布局而非记账。
                print(f"[ZCPO-COMBINED-{tag}] rows={len(ex['L'])} max|D_combined-D_recomputed|={dmax:.3e} max_rel={rel:.3e} "
                      f"L_match={bool((ex['L']==ex2['L']).all())} ref_lp_absdiff 融合={float(ex['ref_lp_absdiff']):.3e} 独立重算={float(ex2['ref_lp_absdiff']):.3e}", flush=True)
                bad = [(i, float(ex["D_exact"][i]), float(ex2["D_exact"][i]), int(ex["L"][i]), int(ex2["L"][i]), float(ex["k3"][i]), float(ex2["k3"][i])) for i in range(len(ex["L"])) if abs(float(ex["D_exact"][i]) - float(ex2["D_exact"][i])) > 1e-3 * max(1.0, abs(float(ex2["D_exact"][i])))]
                if bad:
                    print(f"[ZCPO-COMBINED-VERIFY] mismatched rows (i, D_comb, D_recomp, L_comb, L_recomp, k3_comb, k3_recomp): {bad[:6]}", flush=True)
        if ex is None:
            ex = _exact_seq_kl(worker, td)
    if ex is not None:
        D, k3s, Ls, D_exact = ex["D"], ex["k3"], ex["L"], ex["D_exact"]
        global _LAST_KAPPA
        _LAST_KAPPA = ex["kappa"]
    else:
        if kind == "exact":
            print("[ZCPO] exact KL unavailable (no ref worker handle); falling back to k1", flush=True)
        D, k3s, Ls = _current_seq_logratio(worker, td)
        D_exact = None
    B = R.shape[0]
    D_used = D_exact if D_exact is not None else D
    _check_groups_intact(worker, uids)
    A_over_s, negbetaK, active = zcpo_group_coefficients(R.cpu(), D_used.cpu(), uids, beta, L=Ls.cpu())
    C = A_over_s + negbetaK
    K = -negbetaK / beta if beta > 0 else torch.zeros(B)
    ref = td["response_mask"]
    td["advantages"] = _nested_like(ref, C.tolist())
    td["zcpo_A"] = A_over_s
    td["zcpo_K"] = negbetaK
    td["zcpo_active"] = active
    pos = A_over_s > 0
    def _corr(x, y):
        x = x.double(); y = y.double()
        if x.numel() < 3 or x.std() == 0 or y.std() == 0:
            return 0.0
        return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
    act = active
    Ka = K[act] if act.any() else torch.zeros(1)
    stats = {
        # --- KL 值（整批总和 / 逐 token 均值；当前参数）---
        "kl/k3_sum": float(k3s.sum()),
        "kl/k3_mean": float(k3s.sum() / max(1.0, float(Ls.sum()))),
        "kl/tokens": float(Ls.sum()),
        "kl/len_mean": float(Ls.mean()),
        # --- 序列对数比 D_i（有符号）与中心化 K_i ---
        "zcpo/D_abs_mean": float(D.abs().mean()),
        "zcpo/D_per_token_mean": float((D / Ls.clamp(min=1)).mean()),
        "zcpo/K_abs_mean": float(Ka.abs().mean()),
        "zcpo/K_std": float(Ka.std()) if Ka.numel() > 1 else 0.0,
        "zcpo/K_max_abs": float(Ka.abs().max()),
        "zcpo/K_pos_frac": float((Ka > 0).float().mean()),
        "zcpo/K_sum_over_batch": float(Ka.sum()),
        "zcpo/K_abs_over_sqrtL_mean": float((K.abs() / Ls.clamp(min=1).sqrt())[act].mean()) if act.any() else 0.0,
        "zcpo/K_abs_over_L_mean": float((K.abs() / Ls.clamp(min=1))[act].mean()) if act.any() else 0.0,
        "zcpo/corr_K_len": _corr(K[act], Ls[act]) if act.any() else 0.0,
        "zcpo/corr_absK_len": _corr(K[act].abs(), Ls[act]) if act.any() else 0.0,
        "zcpo/corr_K_A": _corr(K[act], A_over_s[act]) if act.any() else 0.0,
        "zcpo/active_frac": float(active.float().mean()),
        "zcpo/coef_abs_mean": float(C[active].abs().mean()) if active.any() else 0.0,
        "zcpo/betaK_abs_mean": float(negbetaK[active].abs().mean()) if active.any() else 0.0,
        "zcpo/A_abs_mean": float(A_over_s[active].abs().mean()) if active.any() else 0.0,
        "zcpo/reversal_frac_pos": float(((C < 0) & pos).float().sum() / max(1, int(pos.sum()))),
        # 逐行原始值：D_i、K_i、L_i、Σk3_i、A_i/s、uid 序号——用于步内拟合 D 与 L 的关系（线性部分 vs √L 噪声部分）
        "zcpo/Dtilde_abs_mean": float((D / Ls.clamp(min=1)).abs().mean()),
        "zcpo/K_is_per_token": 1.0,
        "zcpo/kl_kind": 1.0 if D_exact is not None else 0.0,   # 1=exact 条件 KL，0=k1 采样对数比
        "zcpo/combined": 1.0 if (ex is not None and ex.get("combined")) else 0.0,   # 1=κ 来自合并推理 pass 的缓存
        "zcpo/ref_lp_absdiff": float(ex["ref_lp_absdiff"]) if ex is not None else float("nan"),   # 裸前向 fp32 ref logp 与引擎 ref_log_prob 的平均绝对差（精度自检）
        "zcpo/Dexact_per_token_mean": float((D_exact / Ls.clamp(min=1)).mean()) if D_exact is not None else float("nan"),
        "rows/D_exact": [round(float(x), 5) for x in D_exact.tolist()] if D_exact is not None else [],
        "rows/D": [round(float(x), 5) for x in D.tolist()],
        "rows/Dtilde": [round(float(x), 6) for x in (D / Ls.clamp(min=1)).tolist()],
        "rows/K": [round(float(x), 6) for x in K.tolist()],
        "rows/L": [int(x) for x in Ls.tolist()],
        "rows/k3_sum": [round(float(x), 5) for x in k3s.tolist()],
        "rows/A": [round(float(x), 5) for x in A_over_s.tolist()],
        "rows/uid": [sorted(set(uids)).index(u) for u in uids],
        "zcpo/frac_betaK_gt_A": float(((negbetaK.abs() > A_over_s.abs()) & active).float().sum() / max(1, int(active.sum()))),
        "zcpo/D_mean": float(D.mean()),
        "zcpo/D_std": float(D.std()) if B > 1 else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
# 3. 分支梯度统计
# ---------------------------------------------------------------------------
def _responses_padded(td):
    """从 no-padding 布局取回复 token id，[B, Lmax]，pad=-1。"""
    resp = td["responses"]
    parts = resp.unbind() if resp.is_nested else list(resp)
    L = max(int(t.shape[0]) for t in parts)
    out = torch.full((len(parts), L), -1, dtype=torch.long)
    for i, t in enumerate(parts):
        out[i, : t.shape[0]] = t.cpu()
    return out


def _vanilla_pg_mat(ratio, adv, low, high, c):
    l1 = -adv * ratio
    l2 = -adv * torch.clamp(ratio, 1.0 - low, 1.0 + high)
    clip1 = torch.maximum(l1, l2)
    l3 = -adv * c
    clip2 = torch.minimum(l3, clip1)
    return torch.where(adv < 0, clip2, clip1)



def _dapo_rho_and_closed(ratio, C, low, high, c):
    """DAPO/PPO 双裁剪下，损失 = -C·ρ(r, C)。ρ 按 C 的符号选：C>0 时 ρ=min(r, clip(r))；C<0 时 ρ=min(max(r, clip(r)), c)。
    门关闭（对 r 的梯度为 0）：C>0 且 r>1+ε_high；C<0 且（r<1−ε_low 或 r>c）。ZCPO 把联合系数 C_i 当作优势喂给 DAPO 默认裁剪，不改任何裁剪规则。"""
    clipped = torch.clamp(ratio, 1.0 - low, 1.0 + high)
    rho_pos = torch.minimum(ratio, clipped)
    rho_neg = torch.minimum(torch.maximum(ratio, clipped), torch.full_like(ratio, float(c)))
    rho = torch.where(C < 0, rho_neg, rho_pos)
    closed = ((C > 0) & (ratio > 1.0 + high)) | ((C < 0) & ((ratio < 1.0 - low) | (ratio > c)))
    return rho, closed

def _make_branch_loss(kind, mode, config, denom, side=None, accountant=None, token_store=None, loss_store=None, last_store=None):
    """kind: pg | kl | pg_shared | kl_shared | pg_clipped | kl_clipped。
    *_shared 只保留全组共享前缀 token；*_clipped 只保留奖励门控已关闭的 token（F1：裁剪后残留的纯 KL）。
    side: 可选 dict，累计 closed/all token 计数。"""
    low, high, c = _clip(config)
    kl_type = str(config.kl_loss_type)
    kl_coef = float(config.kl_loss_coef)

    def fn(model_output, data, dp_group=None):
        lp = no_padding_2_padding(model_output["log_probs"], data)
        keys = ["response_mask", "old_log_probs"]
        if mode == "vanilla":
            keys.append("advantages")
            if kind.startswith("kl") or kind == "pgkl":
                keys.append("ref_log_prob")
        if kind.endswith("shared"):
            keys.append("klstats_shared")
        d = _padded(data, *keys)
        dev = lp.device
        mask = d["response_mask"].to(dev).to(torch.bool)
        nak = torch.clamp(lp - d["old_log_probs"].to(dev), -20.0, 20.0)
        ratio = torch.exp(nak)
        if accountant is not None:
            P_, L_ = _seq_lens(data)
            accountant.begin(data["klstats_rowid"].tolist(), P_, L_)
        if token_store is not None:
            for j, rid in enumerate(data["klstats_rowid"].tolist()):
                n = int(mask[j].sum().item())
                token_store.setdefault(int(rid), {})["cur_logp"] = lp[j, :n].detach().float().cpu().numpy()
                token_store[int(rid)]["ratio"] = ratio[j, :n].detach().float().cpu().numpy()
        if mode == "vanilla":
            adv = d["advantages"].to(dev)
            closed = ((adv > 0) & (ratio > 1.0 + high)) | ((adv < 0) & (ratio < 1.0 - low))
            if last_store is not None and kind == "kl":
                # 2026-09-15 用户定：末 mini-batch 逐 token 记录 (ratio, A, Δ, 是否裁剪)，验证"裁剪 token 是否落在 k3 系数的重尾（π_θ≪π_ref）上"。只记录，不影响损失。
                try:
                    ref_ = d["ref_log_prob"].to(dev); old_ = d["old_log_probs"].to(dev)
                    for j, rid in enumerate(data["klstats_rowid"].tolist()):
                        n = int(mask[j].sum().item())
                        last_store[int(rid)] = {
                            "ratio": ratio[j, :n].detach().float().cpu().numpy(), "adv": adv[j, :n].detach().float().cpu().numpy(),
                            "delta_cur": (lp[j, :n] - ref_[j, :n]).detach().float().cpu().numpy(), "delta_old": (old_[j, :n] - ref_[j, :n]).detach().float().cpu().numpy(),
                            "closed": closed[j, :n].detach().cpu().numpy(),
                        }
                except Exception as e:  # noqa
                    print(f"[KLSTATS-LAST] token store failed: {e!r}", flush=True)
            if kind == "pgkl":
                mat = _vanilla_pg_mat(ratio, adv, low, high, c) + kl_penalty(logprob=lp, ref_logprob=d["ref_log_prob"].to(dev), kl_penalty=kl_type) * kl_coef
            elif kind.startswith("pg"):
                mat = _vanilla_pg_mat(ratio, adv, low, high, c)
            else:
                mat = kl_penalty(logprob=lp, ref_logprob=d["ref_log_prob"].to(dev), kl_penalty=kl_type) * kl_coef
        else:
            Cfull = (data["zcpo_A"] + data["zcpo_K"]).to(dev).float().unsqueeze(-1)
            rho, closed = _dapo_rho_and_closed(ratio, Cfull, low, high, c)   # 门与乘子都由联合系数 C 决定（DAPO 默认裁剪）
            if kind == "pgkl":
                coef = Cfull
            else:
                coef = (data["zcpo_A"] if kind.startswith("pg") else data["zcpo_K"]).to(dev).float().unsqueeze(-1)
            mat = -coef * rho        # pg 分支 + kl 分支 = -C·ρ = DAPO 双裁剪损失
        if side is not None:
            side["closed"] = side.get("closed", 0) + int((closed & mask).sum().item())
            side["all"] = side.get("all", 0) + int(mask.sum().item())
        if kind.endswith("shared"):
            mask = mask & d["klstats_shared"].to(dev).to(torch.bool)
        if kind.endswith("clipped"):
            mask = mask & closed
        if loss_store is not None and kind in ("pg", "kl"):
            # 逐 token 的 loss 项本身（未除分母）：pg = -A·clip(ratio)（或 zcpo 的 -A/s·clip），kl = β·k3（或 zcpo 的 βK·clip）
            vals = (mat * mask).detach().float()
            rids = data["klstats_rowid"].tolist() if "klstats_rowid" in data.keys() else list(range(mask.shape[0]))
            for j, rid in enumerate(rids):
                n = int(mask[j].sum().item()) if not kind.endswith("clipped") else mask.shape[1]
                loss_store.setdefault(kind, {})[int(rid)] = vals[j, :n].cpu().numpy() if not kind.endswith("clipped") else vals[j].cpu().numpy()
        loss = (mat * mask).sum() / denom
        return loss, {}

    return fn


def _loss_level_stats(loss_store, prefix, denom):
    """loss 层面的分支占比：Σ|pg 项|、Σ kl 项（整批与逐 token 分位数）。与梯度占比并列记录。"""
    import numpy as np

    out = {}
    pg = loss_store.get("pg", {}); kl = loss_store.get("kl", {})
    if not pg:
        return out
    pg_all = np.concatenate([np.asarray(v).reshape(-1) for v in pg.values()]) if pg else np.zeros(0)
    kl_all = np.concatenate([np.asarray(kl[r]).reshape(-1)[: len(np.asarray(pg[r]).reshape(-1))] for r in pg if r in kl]) if kl else np.zeros(0)
    pg_abs = np.abs(pg_all)
    out[f"{prefix}pg_sum"] = float(pg_all.sum())
    out[f"{prefix}pg_abs_sum"] = float(pg_abs.sum())
    out[f"{prefix}pg_loss_normalized"] = float(pg_all.sum() / denom)
    q = lambda x, p: float(np.quantile(x, p)) if x.size else 0.0
    out[f"{prefix}pg_abs_token_mean"] = float(pg_abs.mean()) if pg_abs.size else 0.0
    out[f"{prefix}pg_abs_token_p50"] = q(pg_abs, 0.5); out[f"{prefix}pg_abs_token_p90"] = q(pg_abs, 0.9); out[f"{prefix}pg_abs_token_max"] = float(pg_abs.max()) if pg_abs.size else 0.0
    if kl_all.size and kl_all.size == pg_all.size:
        kl_abs = np.abs(kl_all)
        out[f"{prefix}kl_sum"] = float(kl_all.sum())
        out[f"{prefix}kl_abs_sum"] = float(kl_abs.sum())
        out[f"{prefix}kl_loss_normalized"] = float(kl_all.sum() / denom)
        out[f"{prefix}kl_token_mean"] = float(kl_all.mean())
        out[f"{prefix}kl_abs_token_p50"] = q(kl_abs, 0.5); out[f"{prefix}kl_abs_token_p90"] = q(kl_abs, 0.9)
        out[f"{prefix}kl_abs_token_p99"] = q(kl_abs, 0.99); out[f"{prefix}kl_abs_token_max"] = float(kl_abs.max())
        # 整批占比：Σ|kl| / (Σ|pg| + Σ|kl|)，以及 Σ|kl| / Σ|pg|
        out[f"{prefix}kl_share_of_loss"] = float(kl_abs.sum() / max(pg_abs.sum() + kl_abs.sum(), 1e-12))
        out[f"{prefix}kl_over_pg_abs"] = float(kl_abs.sum() / max(pg_abs.sum(), 1e-12))
        # 逐 token 占比：|kl_t| / (|pg_t| + |kl_t|) 的分位数；以及 |kl_t| > |pg_t| 的 token 比例
        tot = pg_abs + kl_abs
        ok = tot > 0
        sh = np.where(ok, kl_abs / np.maximum(tot, 1e-12), 0.0)[ok]
        out[f"{prefix}kl_token_share_mean"] = float(sh.mean()) if sh.size else 0.0
        out[f"{prefix}kl_token_share_p50"] = q(sh, 0.5); out[f"{prefix}kl_token_share_p90"] = q(sh, 0.9); out[f"{prefix}kl_token_share_p99"] = q(sh, 0.99)
        out[f"{prefix}frac_tokens_kl_gt_pg"] = float((kl_abs > pg_abs).mean()) if kl_abs.size else 0.0
        # 奖励为 0 的 token（组内全对/全错，A=0）上的 KL：这些 token 只受 KL 驱动
        zero = pg_abs == 0
        out[f"{prefix}frac_tokens_pg_zero"] = float(zero.mean())
        out[f"{prefix}kl_abs_sum_on_pg_zero"] = float(kl_abs[zero].sum())
        out[f"{prefix}kl_on_pg_zero_over_kl"] = float(kl_abs[zero].sum() / max(kl_abs.sum(), 1e-12))
    return out


def _dump_last_tokens(store, step, mode, config):
    """末 mini-batch 的逐 token 记录落盘：<KLSTATS_OUT 所在目录>/last_dump/step_XXXX.npz，
    字段 tok_off（行偏移）、row、ratio、adv、delta_cur=logπθ−logπref、delta_old=logπold−logπref、closed（奖励门控关闭=被裁剪）、meta。约 1～2 MB/步。"""
    import numpy as np
    rids = sorted(store)
    arrays = {"row": np.asarray(rids, dtype=np.int32), "tok_off": np.cumsum([0] + [len(store[r]["ratio"]) for r in rids]).astype(np.int64)}
    for key in ("ratio", "adv", "delta_cur", "delta_old"):
        arrays[key] = np.concatenate([np.asarray(store[r][key], dtype=np.float32) for r in rids]) if rids else np.zeros(0, np.float32)
    arrays["closed"] = np.concatenate([np.asarray(store[r]["closed"], dtype=np.bool_) for r in rids]) if rids else np.zeros(0, np.bool_)
    arrays["meta"] = np.array(json.dumps({"step": step, "mode": mode, "beta": float(config.kl_loss_coef), "clip": list(_clip(config)),
                                          "note": "closed = (A>0 & ratio>1+eps_high) | (A<0 & ratio<1-eps_low)，与 last/clipped_token_frac 同口径；k3 系数 = beta*(1-exp(-delta_cur))"}))
    out_dir = os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), "last_dump")
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"step_{step:04d}.npz.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(out_dir, f"step_{step:04d}.npz"))


def instrument_last(worker, td, config, uids, mode, step):
    """每步最后一个 mini-batch（策略已更新过几次，ratio≠1）：F1 统计——奖励门控关闭 token 上残留的 KL 梯度。"""
    t0 = time.time()
    B = td.batch_size[0]
    m = _padded(td, "response_mask")["response_mask"].to(torch.bool)
    n_tok = float(m.sum().item())
    denom = n_tok   # 两种模式共用基础算法的 token-mean 聚合（2026-09-22 起 ZCPO 不再用 |B|·T⋆）
    kl_coef = float(config.kl_loss_coef)
    if "ref_log_prob" not in td.keys() or kl_coef <= 0:
        return
    _assign_engine_defaults(worker, td)
    td["klstats_rowid"] = torch.arange(B)
    side = {}
    lstore = {}
    last_tok = {}
    _, pg_acc = _run_pass(worker, [td], _make_branch_loss("pg", mode, config, denom, side, loss_store=lstore))
    _, kl_acc = _run_pass(worker, [td], _make_branch_loss("kl", mode, config, denom, loss_store=lstore, last_store=last_tok))
    _, pgc_acc = _run_pass(worker, [td], _make_branch_loss("pg_clipped", mode, config, denom))
    _, klc_acc = _run_pass(worker, [td], _make_branch_loss("kl_clipped", mode, config, denom))
    pg, kl, pgc, klc = pg_acc.norm(), kl_acc.norm(), pgc_acc.norm(), klc_acc.norm()
    dot = kl_acc.dot(pg_acc)
    tot = max(pg**2 + kl**2 + 2 * dot, 0.0) ** 0.5
    rec = {
        "step": step, "mode": mode, "which": "last_mini_batch", "mini_batch_rows": B, "tokens": n_tok,
        "last/pg_norm": pg, "last/kl_norm": kl, "last/total_norm": tot,
        "last/kl_share_of_total": kl / max(tot, 1e-12), "last/cos_pg_kl": dot / max(pg * kl, 1e-12),
        "last/clipped_token_frac": side.get("closed", 0) / max(1, side.get("all", 1)),
        "last/pg_clipped_norm": pgc, "last/kl_clipped_norm": klc,
        "last/kl_clipped_over_kl": klc / max(kl, 1e-12), "last/kl_clipped_over_total": klc / max(tot, 1e-12),
        "last/pg_sum_over_sqrt_tokens": pg * denom / max(n_tok, 1.0) ** 0.5,
        "last/kl_sum_over_tokens": kl * denom / max(n_tok, 1.0),
        "klstats_time_s": time.time() - t0,
    }
    try:
        rec.update(_loss_level_stats(lstore, "last/loss_", denom))
    except Exception as e:
        print(f"[KLSTATS-LAST] loss-level stats failed: {e!r}", flush=True)
    if last_tok and _env_flag("KLSTATS_LAST_TOKEN_DUMP", "1"):
        try:
            _dump_last_tokens(last_tok, step, mode, config)
        except Exception as e:
            print(f"[KLSTATS-LAST] token dump failed: {e!r}", flush=True)
    # 更新投影：把两支梯度与更新前参数快照留到 CPU，优化器步之后由 after_train_batch 计算实际参数更新 Δθ 在两支方向上的投影
    if _env_flag("KLSTATS_UPDATE_PROJ", "1"):
        try:
            mod = worker.engine.module
            worker._klstats_pending = {
                "step": step, "mode": mode,
                "pg": [None if b is None else b.detach().float().cpu() for b in pg_acc.buf],
                "kl": [None if b is None else b.detach().float().cpu() for b in kl_acc.buf],
                "theta": [_local(q).detach().float().cpu() for q in mod.parameters()],
                "pg_norm": pg, "kl_norm": kl, "tot_norm": tot,
            }
        except Exception as e:
            print(f"[KLSTATS-LAST] snapshot for update projection failed: {e!r}", flush=True)
    del pg_acc, kl_acc, pgc_acc, klc_acc
    torch.cuda.empty_cache()
    JsonlWriter(os.environ.get("KLSTATS_LAST_OUT", os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), "kl_grad_stats_last.jsonl"))).write(rec)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"[KLSTATS-LAST] step={step} mode={mode} pg={pg:.4g} kl={kl:.4g} kl_share={rec['last/kl_share_of_total']:.3g} "
            f"clipped_tok_frac={rec['last/clipped_token_frac']:.3g} kl_on_clipped/kl={rec['last/kl_clipped_over_kl']:.3g} "
            f"pg_on_clipped={pgc:.3g} t={rec['klstats_time_s']:.0f}s",
            flush=True,
        )
    worker.engine.optimizer_zero_grad()
    grad_zero(worker.engine.module)


class PositionGradAccountant:
    """一次反传里按位置记每层的 ‖a_s‖²·‖δ_s‖²，位置 s 对整体梯度的贡献范数 = sqrt(Σ_层 ‖a_s‖²‖δ_s‖²)。
    只挂 nn.Linear（前向记输入范数、反向记输出梯度范数）与 nn.Embedding（反向记输出梯度范数，输入是 one-hot）。
    用法：acc = PositionGradAccountant(module)；每个微批的损失函数里 acc.begin(rowids, P, L)（内部先 flush 上一微批）；
    forward_backward_batch 返回后 acc.flush()；最后 acc.remove()。结果 acc.result: {rowid: np.array[P+L]}。
    RMSNorm 权重不计（参数量不到千分之一）；绑定的 lm_head/embedding 两处贡献分开计（近似）。"""

    def __init__(self, module):
        self.handles = []
        self.a2 = {}
        self.acc = None
        self.meta = None
        self.result = {}
        self.n_linear = 0
        self.n_embed = 0
        # 只用 forward hook + 输出张量的 tensor hook 取 grad_output。不用 register_full_backward_hook：
        # 它会把模块输出包进自定义 Function，verl 的 rmpad 路径随后对 lm_head 输出做 squeeze(0).div_()，
        # 触发 "view ... modified inplace" 错误（flash-attn 冒烟即因此失败）。tensor hook 不改变输出张量。
        for name, mod in module.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.handles.append(mod.register_forward_hook(self._fwd(name, False)))
                self.n_linear += 1
            elif isinstance(mod, torch.nn.Embedding):
                self.handles.append(mod.register_forward_hook(self._fwd(name, True)))
                self.n_embed += 1

    def _fwd(self, name, is_embed):
        def hook(mod, inputs, output):
            if not is_embed:
                x = inputs[0]
                if torch.is_tensor(x) and x.dim() >= 2:
                    self.a2[name] = x.detach().float().pow(2).sum(-1)
            if torch.is_tensor(output) and output.requires_grad:
                output.register_hook(self._bwd(name, is_embed))
        return hook

    def _bwd(self, name, is_embed):
        def hook(g):
            if g is None or self.meta is None:
                return
            d2 = g.detach().float().pow(2).sum(-1)
            if is_embed:
                c = d2
            else:
                a2 = self.a2.get(name)
                if a2 is None or a2.shape != d2.shape:
                    return
                c = a2 * d2
            # rmpad 打包布局是 (1, total_nnz)：按 1-D 处理，flush 里按各行 P+L 依次切；padded 布局 (B, T) 按行切
            c = c.reshape(-1) if (c.dim() == 1 or c.shape[0] == 1) else c.reshape(c.shape[0], -1)
            if self.acc is None:
                self.acc = torch.zeros_like(c, dtype=torch.float64)
            if self.acc.shape == c.shape:
                self.acc += c.double()
        return hook

    def begin(self, rowids, prompt_lens, resp_lens):
        self.flush()
        self.meta = (list(rowids), list(prompt_lens), list(resp_lens))
        self.acc = None

    def flush(self):
        if self.meta is None or self.acc is None:
            self.meta = None
            return
        rowids, P, L = self.meta
        acc = self.acc.sqrt().float().cpu()
        if acc.dim() == 1:
            off = 0
            for r, p_, l_ in zip(rowids, P, L):
                n = int(p_ + l_)
                self.result[int(r)] = acc[off : off + n].numpy()
                off += n
        else:
            for j, (r, p_, l_) in enumerate(zip(rowids, P, L)):
                self.result[int(r)] = acc[j, : int(p_ + l_)].numpy()
        self.meta = None
        self.acc = None

    def remove(self):
        self.flush()
        for h in self.handles:
            h.remove()
        self.handles = []
        self.a2 = {}


def _seq_lens(data):
    """该微批每行的【有效】提示长度与回复长度，与 verl no_padding_2_padding 的算法一致（verl/workers/utils/padding.py）：
    no-padding 布局里 `prompts` 仍是左填充到 max_prompt_length 的二维张量、`responses` 是右填充到 max_response_length 的二维张量，
    只有 input_ids/position_ids/loss_mask 是去填充的嵌套张量；有效长度来自 attention_mask（前 prompt 宽度列之和 = 提示长，其余列之和 = 回复长）。
    2026-09-11 之前的版本用了张量宽度（2048/3072），导致逐位置记账（contrib_*）在 rmpad 运行里全部错位，相关结论已作废。"""
    pr, rs = data["prompts"], data["responses"]
    if pr.is_nested and rs.is_nested:
        return [int(t.shape[0]) for t in pr.unbind()], [int(t.shape[0]) for t in rs.unbind()]
    if "attention_mask" in data.keys() and not data["attention_mask"].is_nested:
        am = data["attention_mask"]
        pw = pr.shape[1] if not pr.is_nested else int(am.shape[1] - rs.shape[1])
        P = [int(x) for x in am[:, :pw].sum(-1).tolist()]
        L = [int(x) for x in am[:, pw:].sum(-1).tolist()]
        return P, L
    # 兜底：response_mask 给回复长，嵌套 input_ids 给总长
    rm = data["response_mask"]
    L = [int(t.sum().item()) for t in (rm.unbind() if rm.is_nested else rm)]
    if "input_ids" in data.keys() and data["input_ids"].is_nested:
        tot = [int(t.shape[0]) for t in data["input_ids"].unbind()]
        return [t - l for t, l in zip(tot, L)], L
    return [int(pr.shape[1])] * len(L), L


def _async_savez(path, **arrays):
    import threading

    import numpy as np

    def _save():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    t = threading.Thread(target=_save, daemon=True)
    t.start()
    return t


def _coherence(total_norm, part_norms):
    """‖Σg_i‖² = Σ‖g_i‖² + Σ_{i≠j}<g_i,g_j>。返回 (‖Σg‖² − Σ‖g_i‖²) / ((Σ‖g_i‖)² − Σ‖g_i‖²)：完全同向=1，互相正交=0，反向<0。"""
    s2 = sum(n * n for n in part_norms)
    s1 = sum(part_norms)
    den = s1 * s1 - s2
    return (total_norm * total_norm - s2) / den if den > 1e-30 else 0.0


def _sample_token_grad_norms(worker, td, mask, coef_pg, coef_kl, K, seed):
    """均匀抽 K 个有效 token，逐个做单 token 反传，返回 [(‖∇logπ_t‖, |c_pg,t|, |c_kl,t|), ...]。"""
    module = worker.engine.module
    valid = mask.nonzero()
    if valid.shape[0] == 0 or K <= 0:
        return []
    g = torch.Generator().manual_seed(int(seed))
    pick = valid[torch.randperm(valid.shape[0], generator=g)[:K]].tolist()
    out = []
    for row, t in pick:
        sub = tu.index_select_tensor_dict(td, torch.tensor([row]))

        def fn(model_output, data, dp_group=None, _t=t):
            lp = no_padding_2_padding(model_output["log_probs"], data)
            return lp[0, _t], {}

        worker.engine.optimizer_zero_grad()
        grad_zero(module)
        worker.engine.forward_backward_batch(sub, loss_function=fn, forward_only=False)
        gn = grad_norm(module)
        out.append((gn, abs(float(coef_pg[row, t])), abs(float(coef_kl[row, t]))))
    worker.engine.optimizer_zero_grad()
    grad_zero(module)
    return out


def _run_pass(worker, td_rows_list, fn, accountant=None):
    """对每个行集合各自 backward，返回逐集合梯度范数和 CPU 累加器（= 全部集合之和）。accountant 可选：按位置记梯度贡献。"""
    module = worker.engine.module
    acc = _CpuGradAccumulator(module)
    norms = []
    for sub in td_rows_list:
        worker.engine.optimizer_zero_grad()
        grad_zero(module)
        worker.engine.forward_backward_batch(sub, loss_function=fn, forward_only=False)
        if accountant is not None:
            accountant.flush()
        norms.append(grad_norm(module))
        acc.add_current()
    worker.engine.optimizer_zero_grad()
    grad_zero(module)
    return norms, acc


_LAST_KAPPA = {}


def _dump_tokens(td, d, m, a, k, uids, mode, step, tok_store, contrib_pg, contrib_kl, config, contrib_tot=None):
    """每步第一个 mini-batch 的全量逐 token / 逐位置记录，npz 异步落盘。
    行拼接：tok_off[r]..tok_off[r+1] 是第 r 行的响应 token 段；pos_off 是提示+响应位置段（contrib_*）。"""
    import numpy as np

    B = td.batch_size[0]
    responses = _responses_padded(td)
    P_, L_ = _seq_lens(td)
    lens = [int(m[i].sum().item()) for i in range(B)]
    old_lp = d["old_log_probs"].float().cpu().numpy()
    ref_lp = d["ref_log_prob"].float().cpu().numpy() if "ref_log_prob" in d.keys() else None
    adv = d["advantages"].float().cpu().numpy()
    uid_list = sorted(set(uids))
    uid_idx = {u: i for i, u in enumerate(uid_list)}
    rows = {"row": [], "uid": [], "prompt_len": [], "resp_len": [], "tok_off": [0], "pos_off": [0]}
    tok = {"token_id": [], "old_logp": [], "ref_logp": [], "cur_logp": [], "ratio": [], "adv": [], "coef_pg": [], "coef_kl": [], "k3": [], "loss_pg": [], "loss_kl": [], "kappa": []}
    pos = {"contrib_pg": [], "contrib_kl": [], "contrib_tot": []}
    for i in range(B):
        n = lens[i]
        rows["row"].append(i)
        rows["uid"].append(uid_idx[uids[i]])
        rows["prompt_len"].append(P_[i])
        rows["resp_len"].append(n)
        tok["token_id"].append(responses[i, :n].numpy())
        tok["old_logp"].append(old_lp[i, :n])
        tok["ref_logp"].append(ref_lp[i, :n] if ref_lp is not None else np.full(n, np.nan, dtype=np.float32))
        ts = tok_store.get(i, {})
        tok["cur_logp"].append(np.asarray(ts.get("cur_logp", np.full(n, np.nan, dtype=np.float32)))[:n])
        tok["ratio"].append(np.asarray(ts.get("ratio", np.full(n, np.nan, dtype=np.float32)))[:n])
        tok["loss_pg"].append(np.asarray(ts.get("loss_pg", np.full(n, np.nan, dtype=np.float32)))[:n])
        tok["loss_kl"].append(np.asarray(ts.get("loss_kl", np.full(n, np.nan, dtype=np.float32)))[:n])
        kp = _LAST_KAPPA.get(i) if mode == "zcpo" else None
        tok["kappa"].append(np.asarray(kp, dtype=np.float32)[:n] if kp is not None and len(kp) >= n else np.full(n, np.nan, dtype=np.float32))
        tok["adv"].append(adv[i, :n])
        tok["coef_pg"].append(a[i, :n].float().cpu().numpy())
        tok["coef_kl"].append(k[i, :n].float().cpu().numpy())
        if ref_lp is not None:
            dv = ref_lp[i, :n] - old_lp[i, :n]
            tok["k3"].append(np.clip(np.exp(dv) - dv - 1.0, -10, 10).astype(np.float32))
        else:
            tok["k3"].append(np.full(n, np.nan, dtype=np.float32))
        rows["tok_off"].append(rows["tok_off"][-1] + n)
        cp = np.asarray(contrib_pg.get(i, np.full(P_[i] + n, np.nan)), dtype=np.float32)
        ck = np.asarray(contrib_kl.get(i, np.full(P_[i] + n, np.nan)), dtype=np.float32)
        pos["contrib_pg"].append(cp)
        pos["contrib_kl"].append(ck)
        pos["contrib_tot"].append(np.asarray((contrib_tot or {}).get(i, np.full(P_[i] + n, np.nan)), dtype=np.float32))
        rows["pos_off"].append(rows["pos_off"][-1] + len(cp))
    arrays = {key: np.asarray(v) for key, v in rows.items()}
    for key, v in tok.items():
        arrays[key] = np.concatenate(v).astype(np.int32 if key == "token_id" else np.float32)
    for key, v in pos.items():
        arrays[key] = np.concatenate(v).astype(np.float32)
    if mode == "zcpo":
        for key in ("zcpo_A", "zcpo_K"):
            if key in td.keys():
                arrays[key] = td[key].float().cpu().numpy()
    arrays["meta"] = np.array(json.dumps({
        "step": step, "mode": mode, "beta": float(config.kl_loss_coef), "clip": list(_clip(config)), "uids": uid_list,
        "note": "kappa 是逐位置精确条件 KL(zcpo 模式)；contrib_tot 是奖励+KL 合并反传的逐位置贡献范数，<pg_s,kl_s>=(tot²−pg²−kl²)/2；loss_pg/loss_kl 是逐 token 的 loss 项(未除分母)；contrib_* 是位置 s 的激活对该分支整体梯度的贡献范数(含提示位置，位置 s 预测 token s+1)；tok_* 只含响应 token；coef_kl 为 β(1-π_ref/π_θ)",
    }))
    out_dir = os.environ.get("KLSTATS_DUMP_DIR", os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), "token_dump"))
    _async_savez(os.path.join(out_dir, f"step_{step:04d}.npz"), **arrays)
    # 逐位置"奖励分支抵消后被 KL 反转"的统计（只看响应位置：位置 s 预测 token s+1，故取 [P-1, P+L-1)）
    out = {}
    try:
        cp, ck, ct = arrays["contrib_pg"], arrays["contrib_kl"], arrays["contrib_tot"]
        sel = np.zeros(len(cp), dtype=bool)
        for i in range(B):
            o = int(arrays["pos_off"][i]); sel[o + max(P_[i] - 1, 0): o + P_[i] + lens[i] - 1] = True
        ok = sel & np.isfinite(cp) & np.isfinite(ck) & np.isfinite(ct)
        cp, ck, ct = cp[ok], ck[ok], ct[ok]
        dot = (ct ** 2 - cp ** 2 - ck ** 2) / 2.0
        cosv = dot / np.maximum(cp * ck, 1e-30)
        kl_dom = ck > cp                                   # 该位置 KL 贡献大于奖励贡献
        rev = kl_dom & (dot < 0)                           # 且方向相反：合成方向由 KL 决定并与奖励方向相反
        pg_ref = np.median(cp[cp > 0]) if np.any(cp > 0) else 1.0
        weak = cp < 0.1 * pg_ref                           # 奖励贡献已抵消到中位数的 1/10 以下
        out.update({
            "pos/n": int(ok.sum()),
            "pos/frac_kl_dominates": float(kl_dom.mean()),
            "pos/frac_reversal": float(rev.mean()),
            "pos/frac_pg_weak": float(weak.mean()),
            "pos/frac_reversal_given_pg_weak": float(rev[weak].mean()) if weak.any() else 0.0,
            "pos/frac_kl_dominates_given_pg_weak": float(kl_dom[weak].mean()) if weak.any() else 0.0,
            "pos/cos_mean": float(cosv.mean()), "pos/cos_p10": float(np.quantile(cosv, 0.1)), "pos/cos_p90": float(np.quantile(cosv, 0.9)),
            "pos/kl_over_pg_median": float(np.median(ck / np.maximum(cp, 1e-30))),
            "pos/kl_over_pg_p90": float(np.quantile(ck / np.maximum(cp, 1e-30), 0.9)),
            "pos/pg_contrib_median": float(pg_ref), "pos/kl_contrib_median": float(np.median(ck)),
        })
    except Exception as e:
        print(f"[KLSTATS] position reversal stats failed: {e!r}", flush=True)
    return out



@torch.no_grad()
def _param_concentration(module, pg_acc, kl_acc):
    """参数层面的集中度：两支梯度（整 mini-batch 累加）在参数坐标上的分布。
    (a) 按模块类型 / 层分组的 ‖g‖² 份额；(b) 全部坐标按 |g| 排序后 top 0.1%/1%/10% 坐标占 ‖g‖² 的份额与 Gini；
    (c) pg 与 kl 各自 top-1% 坐标集合的重合度（Jaccard）。只依赖累加器 buffer，不依赖逐位置记账。"""
    out = {}
    names = [n for n, _ in module.named_parameters()]
    def gtype(n):
        if "embed_tokens" in n or "lm_head" in n: return "embed/lm_head"
        if "self_attn" in n: return "attn"
        if "mlp" in n: return "mlp"
        if "norm" in n: return "norm"
        return "other"
    def layer(n):
        return int(n.split(".layers.")[1].split(".")[0]) if ".layers." in n else -1
    for tag, acc in (("pg", pg_acc), ("kl", kl_acc)):
        if acc is None or acc.buf is None: continue
        bytype, bylayer, tot = {}, {}, 0.0
        flat = []
        for n, b in zip(names, acc.buf):
            if b is None: continue
            e = float(b.double().pow(2).sum()); tot += e
            bytype[gtype(n)] = bytype.get(gtype(n), 0.0) + e
            li = layer(n); bylayer[li] = bylayer.get(li, 0.0) + e
            flat.append(b.detach().float().abs().flatten())
        if tot <= 0: continue
        for k, v in bytype.items(): out[f"param/{tag}_share_{k}"] = v / tot
        L = [k for k in bylayer if k >= 0]
        if L:
            nl = max(L) + 1; thirds = [sum(v for k, v in bylayer.items() if 0 <= k < nl // 3), sum(v for k, v in bylayer.items() if nl // 3 <= k < 2 * nl // 3), sum(v for k, v in bylayer.items() if k >= 2 * nl // 3)]
            out[f"param/{tag}_share_layers_early"] = thirds[0] / tot; out[f"param/{tag}_share_layers_mid"] = thirds[1] / tot; out[f"param/{tag}_share_layers_late"] = thirds[2] / tot
        g = torch.cat(flat)
        g2 = g.double().pow(2); N = g2.numel()
        srt, idx = torch.sort(g, descending=True)
        cs = torch.cumsum(srt.double().pow(2), 0); tot2 = float(cs[-1])
        for fr in (0.001, 0.01, 0.1):
            k = max(1, int(N * fr)); out[f"param/{tag}_top{fr:g}_share"] = float(cs[k - 1]) / tot2
        # Gini（按 |g| 排序的累计份额）
        asc = torch.flip(srt.double(), [0]); cum = torch.cumsum(asc, 0); out[f"param/{tag}_gini_abs"] = float(1 - 2 * cum.sum() / (cum[-1] * N) + 1.0 / N)
        out[f"param/{tag}_frac_zero"] = float((g == 0).sum()) / N
        k1p = max(1, int(N * 0.01)); out[f"_top_{tag}"] = idx[:k1p]
        del g, g2, srt, cs, asc, cum
    if "_top_pg" in out and "_top_kl" in out:
        a, b = out.pop("_top_pg"), out.pop("_top_kl")
        inter = len(set(a.tolist()) & set(b.tolist())) if a.numel() <= 6_000_000 else float(torch.isin(a, b).sum())
        out["param/top1pct_jaccard_pg_kl"] = float(inter) / float(a.numel() + b.numel() - inter)
    else:
        out.pop("_top_pg", None); out.pop("_top_kl", None)
    return out

def instrument(worker, td, config, uids, mode, step):
    t0 = time.time()
    B = td.batch_size[0]
    low, high, c = _clip(config)
    kl_coef = float(config.kl_loss_coef)
    kl_type = str(config.kl_loss_type)

    # --- 共享前缀 / 相似度 ---
    responses = _responses_padded(td)
    resp_mask = _padded(td, "response_mask")["response_mask"].cpu()
    if resp_mask.shape[1] != responses.shape[1]:
        L = max(resp_mask.shape[1], responses.shape[1])
        rm = torch.zeros(B, L, dtype=resp_mask.dtype); rm[:, : resp_mask.shape[1]] = resp_mask; resp_mask = rm
        rs = torch.full((B, L), -1, dtype=torch.long); rs[:, : responses.shape[1]] = responses; responses = rs
    sp_stats, shared_pad, per_group_info = shared_prefix_stats(responses, resp_mask, uids)
    td["klstats_shared"] = _nested_from_padded_bool(td["response_mask"], shared_pad)

    # --- 系数级统计（无梯度；首个 mini-batch old≈current，用 old_log_probs）---
    d = _padded(td, "response_mask", "old_log_probs", "advantages", *(["ref_log_prob"] if "ref_log_prob" in td.keys() else []))
    m = d["response_mask"].to(torch.bool)
    n_tok = float(m.sum().item())
    coef = {}
    if "ref_log_prob" in d.keys():  # 参考策略与 rollout 时策略的逐 token 差异（诊断：ref 是否真的冻结）
        dref = (d["old_log_probs"].float() - d["ref_log_prob"].float()).abs()
        coef["diag/abs_old_minus_ref_mean"] = float((dref * m).sum() / max(1.0, n_tok))
        coef["diag/abs_old_minus_ref_max"] = float(dref[m].max()) if bool(m.any()) else 0.0
    if mode == "vanilla" and "ref_log_prob" in d.keys():
        a = d["advantages"].float()  # ratio=1 => 门控全开，a_it = A_i
        k = kl_token_coef(d["old_log_probs"].float(), d["ref_log_prob"].float(), kl_coef, kl_type)
    else:
        a = td["zcpo_A"].float().unsqueeze(-1).expand_as(m).clone()
        k = td["zcpo_K"].float().unsqueeze(-1).expand_as(m).clone()
    sm = shared_pad & m
    net = gross = knet = kgross = 0.0
    n_pos = 0
    for _, rows in group_by_uid(uids).items():
        ii = torch.tensor(rows)
        g_sm = sm[ii]
        if not bool(g_sm.any()):
            continue
        cols = g_sm.any(0)
        net += float((a[ii] * g_sm).sum(0)[cols].abs().sum())
        gross += float((a[ii].abs() * g_sm).sum(0)[cols].sum())
        knet += float((k[ii] * g_sm).sum(0)[cols].abs().sum())
        kgross += float((k[ii].abs() * g_sm).sum(0)[cols].sum())
        n_pos += int(cols.sum())
    tot_a = float((a.abs() * m).sum()); tot_k = float((k.abs() * m).sum())
    coef.update({
        "shared_prefix/coef_reward_cancel_ratio": 1.0 - net / max(gross, 1e-12),
        "shared_prefix/coef_kl_cancel_ratio": 1.0 - knet / max(kgross, 1e-12),
        "shared_prefix/coef_reward_gross_frac_of_batch": gross / max(tot_a, 1e-12),
        "shared_prefix/coef_kl_gross_frac_of_batch": kgross / max(tot_k, 1e-12),
        "shared_prefix/positions": n_pos,
        "coef/reward_abs_mean_per_token": tot_a / max(1.0, n_tok),
        "coef/kl_abs_mean_per_token": tot_k / max(1.0, n_tok),
        "coef/frac_tokens_kl_gt_reward": float(((k.abs() > a.abs()).float() * m).sum() / max(1.0, n_tok)),
    })

    # --- 分支反向传播 ---
    denom = n_tok   # 两种模式共用基础算法的 token-mean 聚合
    _assign_engine_defaults(worker, td)
    dump = _env_flag("KLSTATS_DUMP") and step % int(os.environ.get("KLSTATS_DUMP_EVERY", "1")) == 0
    td["klstats_rowid"] = torch.arange(B)
    lstore = {}
    groups = group_by_uid(uids)
    per_group = _env_flag("KLSTATS_PER_GROUP", "1") and (not dist.is_initialized() or dist.get_world_size() == 1)
    if per_group:
        subs = [tu.index_select_tensor_dict(td, torch.tensor(rows)) for rows in groups.values()]
    else:
        subs = [td]
    contrib_pg, contrib_kl, contrib_tot, tok_store = {}, {}, {}, {}
    if dump:
        acct = PositionGradAccountant(worker.engine.module)
        pg_norms, pg_acc = _run_pass(worker, subs, _make_branch_loss("pg", mode, config, denom, accountant=acct, token_store=tok_store, loss_store=lstore), accountant=acct)
        acct.remove()
        contrib_pg = acct.result
    else:
        pg_norms, pg_acc = _run_pass(worker, subs, _make_branch_loss("pg", mode, config, denom, loss_store=lstore))
    pg_total = pg_acc.norm()
    rec = {"step": step, "mode": mode, "mini_batch_rows": B, "tokens": n_tok, "grad/pg_norm": pg_total}
    has_ref = "ref_log_prob" in td.keys()
    if has_ref and kl_coef > 0:
        if dump:
            acct = PositionGradAccountant(worker.engine.module)
            kl_norms, kl_acc = _run_pass(worker, subs, _make_branch_loss("kl", mode, config, denom, accountant=acct, loss_store=lstore), accountant=acct)
            acct.remove()
            contrib_kl = acct.result
            # 第三次：奖励+KL 合并反传的逐位置贡献范数 → 逐位置符号内积 <pg_s,kl_s> = (tot_s² − pg_s² − kl_s²)/2（极化恒等式）
            acct = PositionGradAccountant(worker.engine.module)
            _run_pass(worker, subs, _make_branch_loss("pgkl", mode, config, denom, accountant=acct), accountant=acct)
            acct.remove()
            contrib_tot = acct.result
        else:
            kl_norms, kl_acc = _run_pass(worker, subs, _make_branch_loss("kl", mode, config, denom, loss_store=lstore))
        kl_total = kl_acc.norm()
        dot = kl_acc.dot(pg_acc)
        tot = max(pg_total**2 + kl_total**2 + 2 * dot, 0.0) ** 0.5
        rec.update({
            "grad/kl_norm": kl_total, "grad/total_norm": tot,
            "grad/kl_share_of_total": kl_total / max(tot, 1e-12),
            "grad/kl_over_pg": kl_total / max(pg_total, 1e-12),
            "grad/cos_pg_kl": dot / max(pg_total * kl_total, 1e-12),
        })
        if _env_flag("KLSTATS_PARAM_CONC", "1") and step % int(os.environ.get("KLSTATS_PARAM_CONC_EVERY", "5")) == 0:
            try:
                rec.update(_param_concentration(worker.engine.module, pg_acc, kl_acc))
            except Exception as e:
                print(f"[KLSTATS] param concentration failed: {e!r}", flush=True)
        del kl_acc
        _, pgs_acc = _run_pass(worker, [td], _make_branch_loss("pg_shared", mode, config, denom))
        _, kls_acc = _run_pass(worker, [td], _make_branch_loss("kl_shared", mode, config, denom))
        pgs, kls = pgs_acc.norm(), kls_acc.norm()
        rec.update({
            "grad/pg_shared_norm": pgs, "grad/kl_shared_norm": kls,
            "grad/pg_shared_over_pg": pgs / max(pg_total, 1e-12),
            "grad/kl_shared_over_kl": kls / max(kl_total, 1e-12),
        })
        del pgs_acc, kls_acc
    else:
        kl_norms = [0.0] * len(subs)
        kl_total = 0.0
    del pg_acc
    try:
        rec.update(_loss_level_stats(lstore, "loss/", denom))
    except Exception as e:
        print(f"[KLSTATS] loss-level stats failed: {e!r}", flush=True)
    if dump:
        try:
            for kind_, key_ in (("pg", "loss_pg"), ("kl", "loss_kl")):
                for rid_, v_ in lstore.get(kind_, {}).items():
                    tok_store.setdefault(int(rid_), {})[key_] = v_
            rec.update(_dump_tokens(td, d, m, a, k, uids, mode, step, tok_store, contrib_pg, contrib_kl, config, contrib_tot=contrib_tot))
        except Exception as e:  # 落盘失败不影响训练
            print(f"[KLSTATS] token dump failed: {e!r}", flush=True)
    rec.update(sp_stats)
    rec.update(coef)
    if "ref_log_prob" in d.keys():
        _Lr = m.sum(-1).float()
        _Dr = ((d["old_log_probs"].float() - d["ref_log_prob"].float()) * m).sum(-1)
        _dv = d["ref_log_prob"].float() - d["old_log_probs"].float()
        _k3r = (torch.clamp(torch.exp(_dv) - _dv - 1.0, -10.0, 10.0) * m).sum(-1)
        _Ar = (d["advantages"].float() * m).sum(-1) / _Lr.clamp(min=1)
        _ul = sorted(set(uids))
        rec["rows/L"] = [int(x) for x in _Lr.tolist()]
        rec["rows/D"] = [round(float(x), 5) for x in _Dr.tolist()]
        rec["rows/k3_sum"] = [round(float(x), 5) for x in _k3r.tolist()]
        rec["rows/A"] = [round(float(x), 5) for x in _Ar.tolist()]
        rec["rows/uid"] = [_ul.index(u) for u in uids]

    # --- 真实相加（不除以归一化分母）与 KL 值总和 ---
    rec["grad/denominator"] = denom
    rec["grad/pg_sum_norm"] = pg_total * denom
    rec["grad/kl_sum_norm"] = kl_total * denom
    # token 数与占比的关系：奖励分支各 token 梯度近似正交，Σ 随 √N 增长；KL 同向，Σ 随 N 增长。
    # 记 Σ/√N 与 Σ/N，若假设成立则 pg 的 Σ/√N、kl 的 Σ/N 随 N 基本不变。
    rec["scale/tokens"] = n_tok
    rec["scale/pg_sum_over_sqrt_tokens"] = rec["grad/pg_sum_norm"] / max(n_tok, 1.0) ** 0.5
    rec["scale/pg_sum_over_tokens"] = rec["grad/pg_sum_norm"] / max(n_tok, 1.0)
    rec["scale/kl_sum_over_sqrt_tokens"] = rec["grad/kl_sum_norm"] / max(n_tok, 1.0) ** 0.5
    rec["scale/kl_sum_over_tokens"] = rec["grad/kl_sum_norm"] / max(n_tok, 1.0)
    rec["scale/kl_share_of_total"] = rec.get("grad/kl_share_of_total", 0.0)
    if "ref_log_prob" in d.keys():
        dv = (d["ref_log_prob"].float() - d["old_log_probs"].float())
        k3 = torch.clamp(torch.exp(dv) - dv - 1.0, -10.0, 10.0)
        rec["kl/k3_sum"] = float((k3 * m).sum())
        rec["kl/k3_mean"] = rec["kl/k3_sum"] / max(1.0, n_tok)
        rec["kl/beta"] = kl_coef
        rec["kl/beta_k3_sum"] = kl_coef * rec["kl/k3_sum"]
    # --- 组间一致性（用已有的逐组范数，精确）---
    if per_group and len(subs) > 1:
        rec["coh/pg_group"] = _coherence(pg_total, pg_norms)
        rec["coh/kl_group"] = _coherence(kl_total, kl_norms) if kl_total > 0 else 0.0
    # --- token 间一致性（抽样单 token 反传）---
    K = int(os.environ.get("KLSTATS_TOKEN_SAMPLES", "24"))
    if K > 0:
        # 逐 token 系数：vanilla 首个 mini-batch ratio=1 => |A_t|；KL => β|1−π_ref/π_θ|；zcpo => 行常数
        samp = _sample_token_grad_norms(worker, td, m, a, k, K, seed=step)
        if samp:
            N = n_tok   # 两种模式共用基础算法的 token-mean 聚合
            gns = [x[0] for x in samp]
            rec["tok/samples"] = len(samp)
            rec["tok/grad_norm_mean"] = sum(gns) / len(gns)
            rec["tok/grad_norm_max"] = max(gns)
            for name, j, S in (("pg", 1, rec["grad/pg_sum_norm"]), ("kl", 2, rec["grad/kl_sum_norm"])):
                v = [x[0] * x[j] for x in samp]
                e1 = sum(v) / len(v)
                e2 = sum(x * x for x in v) / len(v)
                rec[f"tok/{name}_coefgrad_norm_mean"] = e1
                rand = (n_tok * e2) ** 0.5
                aligned = n_tok * e1
                rec[f"coh/{name}_sum_over_random"] = S / rand if rand > 0 else 0.0
                rec[f"coh/{name}_sum_over_aligned"] = S / aligned if aligned > 0 else 0.0
                den = n_tok * n_tok * e1 * e1 - n_tok * e2
                rec[f"coh/{name}_token"] = (S * S - n_tok * e2) / den if den > 0 else 0.0
    if per_group:
        A_seq = (d["advantages"].float() * m).sum(-1) / m.sum(-1).clamp(min=1) if mode == "vanilla" else td["zcpo_A"].float()
        glist = []
        for info, pn, kn in zip(per_group_info, pg_norms, kl_norms):
            gd = {k_: v_ for k_, v_ in info.items() if k_ != "rows"}
            ii = torch.tensor(info["rows"])
            n_tok_g = float(m[ii].sum().item())
            gd.update({"pg_norm": pn, "kl_norm": kn, "kl_over_pg": kn / max(pn, 1e-12),
                       "n_tokens": n_tok_g, "kl_share_approx": kn / max(pn + kn, 1e-12),
                       "pg_sum_over_sqrt_tokens": pn * denom / max(n_tok_g, 1.0) ** 0.5,
                       "kl_sum_over_tokens": kn * denom / max(n_tok_g, 1.0),
                       "adv_max": float(A_seq[ii].max()), "adv_min": float(A_seq[ii].min()), "n_pos": int((A_seq[ii] > 0).sum())})
            glist.append(gd)
        rec["groups"] = glist
        js = sorted(g["pair_jaccard"] for g in glist)
        med = js[len(js) // 2] if js else 0.0
        hi = [g["pg_norm"] for g in glist if g["pair_jaccard"] >= med]
        lo = [g["pg_norm"] for g in glist if g["pair_jaccard"] < med]
        rec["grad/pg_norm_group_mean_highsim"] = sum(hi) / max(1, len(hi))
        rec["grad/pg_norm_group_mean_lowsim"] = sum(lo) / max(1, len(lo))
    rec["klstats_time_s"] = time.time() - t0
    JsonlWriter().write(rec)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"[KLSTATS] step={step} mode={mode} pg={rec.get('grad/pg_norm', 0):.4g} kl={rec.get('grad/kl_norm', 0):.4g} "
            f"kl_share={rec.get('grad/kl_share_of_total', 0):.3f} cos={rec.get('grad/cos_pg_kl', 0):.3f} "
            f"shared_tok_frac={rec['shared_prefix/token_frac_group']:.3f} jaccard={rec['shared_prefix/pair_jaccard_mean']:.3f} "
            f"pg_shared/pg={rec.get('grad/pg_shared_over_pg', 0):.3g} kl_shared/kl={rec.get('grad/kl_shared_over_kl', 0):.3g} "
            f"|old-ref|={rec.get('diag/abs_old_minus_ref_mean', 0):.3g} "
            f"sum_pg={rec.get('grad/pg_sum_norm', 0):.4g} sum_kl={rec.get('grad/kl_sum_norm', 0):.4g} "
            f"coh_tok(pg,kl)=({rec.get('coh/pg_token', 0):.3g},{rec.get('coh/kl_token', 0):.3g}) "
            f"coh_grp(pg,kl)=({rec.get('coh/pg_group', 0):.3g},{rec.get('coh/kl_group', 0):.3g}) "
            f"sum/random(pg,kl)=({rec.get('coh/pg_sum_over_random', 0):.3g},{rec.get('coh/kl_sum_over_random', 0):.3g}) "
            f"t={rec['klstats_time_s']:.0f}s",
            flush=True,
        )
    td.pop("klstats_shared", None)
    td.pop("klstats_rowid", None)
    worker.engine.optimizer_zero_grad()
    grad_zero(worker.engine.module)
    torch.cuda.empty_cache()   # 诊断用的梯度累加 buffer 已 del，但缓存分配器仍保留着；不还回去 vLLM 唤醒时会 OOM（24GB 卡实测）


# ---------------------------------------------------------------------------
# 4. 入口
# ---------------------------------------------------------------------------
def before_train_batch(worker, mini_batch_td, batch_idx, total_num_iterations=None):
    worker._zcpo_batch_idx = batch_idx
    config = _actor_config(worker)
    mode = _loss_mode(config)
    if batch_idx == 0:
        worker._klstats_step = getattr(worker, "_klstats_step", 0) + 1
    # 断点续训：启动脚本把已完成的全局步数放进 KLSTATS_STEP_OFFSET，保证 jsonl 里的 step 与 trainer 的 global_step 一致
    step = getattr(worker, "_klstats_step", 0) + int(os.environ.get("KLSTATS_STEP_OFFSET", "0"))
    uids = _uids(mini_batch_td)
    if mode == "zcpo":
        stats = zcpo_coefficients(worker, mini_batch_td, config, uids)
        JsonlWriter(os.environ.get("ZCPO_COEF_OUT", os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), "zcpo_coef.jsonl"))).write(
            {"step": step, "batch_idx": batch_idx, **stats}
        )
        if batch_idx == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[ZCPO] step={step} " + " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in stats.items() if isinstance(v, (int, float))), flush=True)
    if _env_flag("KLSTATS_ENABLE") and batch_idx == 0:
        every = int(os.environ.get("KLSTATS_EVERY", "1"))
        if step % every == 0:
            instrument(worker, mini_batch_td, config, uids, mode, step)
    if (
        _env_flag("KLSTATS_ENABLE")
        and _env_flag("KLSTATS_LAST_MB", "1")
        and total_num_iterations is not None
        and total_num_iterations > 1
        and batch_idx == total_num_iterations - 1
        and step % int(os.environ.get("KLSTATS_LAST_EVERY", "1")) == 0
    ):
        instrument_last(worker, mini_batch_td, config, uids, mode, step)


def after_train_batch(worker, mini_batch_td, batch_idx, total_num_iterations=None):
    """最后一个 mini-batch 的优化器步之后：实际参数更新 Δθ = θ_new − θ_old 与两支梯度的关系。
    记录 ‖Δθ‖、cos(Δθ, g_pg)、cos(Δθ, g_kl)、cos(Δθ, g_pg+g_kl)，以及 Δθ 在两支单位方向上的投影之比。
    AdamW 逐坐标归一化，梯度整体缩小不会让步长缩小；这里直接量"更新里有多少沿 KL 方向"。"""
    pend = getattr(worker, "_klstats_pending", None)
    if pend is None or total_num_iterations is None or batch_idx != total_num_iterations - 1:
        return
    worker._klstats_pending = None
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        mod = worker.engine.module
        d_norm2 = d_pg = d_kl = d_th = th_norm2 = 0.0
        pg2 = kl2 = pgkl = 0.0
        n_params = 0
        for q, g_pg, g_kl, th in zip(mod.parameters(), pend["pg"], pend["kl"], pend["theta"]):
            new = _local(q).detach().float().cpu()
            dlt = (new - th).double()
            n_params += dlt.numel()
            d_norm2 += float((dlt * dlt).sum()); d_th += float((dlt * th.double()).sum()); th_norm2 += float((th.double() ** 2).sum())
            if g_pg is not None:
                gp = g_pg.double(); d_pg += float((dlt * gp).sum()); pg2 += float((gp * gp).sum())
            if g_kl is not None:
                gk = g_kl.double(); d_kl += float((dlt * gk).sum()); kl2 += float((gk * gk).sum())
            if g_pg is not None and g_kl is not None:
                pgkl += float((g_pg.double() * g_kl.double()).sum())
        dn = d_norm2 ** 0.5; pgn = pg2 ** 0.5; kln = kl2 ** 0.5
        tot2 = pg2 + kl2 + 2 * pgkl; totn = max(tot2, 0.0) ** 0.5
        d_tot = d_pg + d_kl
        lr = None
        try:
            lr = float(worker.engine.optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        rec = {
            "step": pend["step"], "mode": pend["mode"], "which": "update_projection",
            "upd/norm": dn, "upd/n_params": n_params, "upd/lr": lr,
            "upd/rms_per_param_over_lr": (dn / max(n_params, 1) ** 0.5 / lr) if lr else None,   # ≈ 平均 |m/√v|
            "upd/cos_pg": d_pg / max(dn * pgn, 1e-30), "upd/cos_kl": d_kl / max(dn * kln, 1e-30), "upd/cos_total": d_tot / max(dn * totn, 1e-30),
            "upd/proj_pg": -d_pg / max(pgn, 1e-30), "upd/proj_kl": -d_kl / max(kln, 1e-30),   # 沿"负梯度"方向的投影长度（下降为正）
            "upd/proj_kl_over_proj_pg": (d_kl / max(kln, 1e-30)) / (d_pg / max(pgn, 1e-30)) if abs(d_pg) > 0 else None,
            "upd/grad_kl_over_pg": kln / max(pgn, 1e-30), "upd/cos_pg_kl": pgkl / max(pgn * kln, 1e-30),
            "upd/cos_theta": d_th / max(dn * th_norm2 ** 0.5, 1e-30),   # 权重衰减分量：Δθ 与 θ 的夹角
            "klstats_time_s": time.time() - t0,
        }
        JsonlWriter(os.environ.get("KLSTATS_LAST_OUT", os.path.join(os.path.dirname(os.environ.get("KLSTATS_OUT", "./x")), "kl_grad_stats_last.jsonl"))).write(rec)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[KLSTATS-UPD] step={rec['step']} |dθ|={dn:.3g} cos(dθ,pg)={rec['upd/cos_pg']:.3f} cos(dθ,kl)={rec['upd/cos_kl']:.3f} "
                  f"cos(dθ,pg+kl)={rec['upd/cos_total']:.3f} proj_kl/proj_pg={rec['upd/proj_kl_over_proj_pg']} grad kl/pg={rec['upd/grad_kl_over_pg']:.3g} t={rec['klstats_time_s']:.0f}s", flush=True)
    except Exception as e:
        print(f"[KLSTATS-UPD] failed: {e!r}", flush=True)
