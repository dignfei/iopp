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
"""Validate real MHA/GQA/MQA head configurations before model patching."""

import pytest
import torch
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from verl.models.transformers.monkey_patch import apply_monkey_patch


def test_opt_mha_does_not_require_a_grouped_attention_configuration():
    config = OPTConfig(vocab_size=32, hidden_size=24, num_attention_heads=6, num_hidden_layers=1, ffn_dim=48)
    assert not hasattr(config, "num_key_value_heads")
    assert not hasattr(config, "text_config")
    model = OPTForCausalLM(config)
    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)
    with torch.no_grad():
        logits = model(torch.tensor([[5, 6, 7]])).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()
    with pytest.raises(AssertionError, match="num_attention_heads 6 must be divisible"):
        apply_monkey_patch(model, ulysses_sp_size=4, use_remove_padding=False)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_explicit_qwen_mqa_and_gqa_head_counts_remain_authoritative(kv_heads):
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=12,
        num_key_value_heads=kv_heads,
    )
    model = Qwen2ForCausalLM(config)
    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)
    assert model.model.layers[0].self_attn.k_proj.out_features == kv_heads * 8
    if kv_heads == 4:
        with pytest.raises(AssertionError, match="num_key_value_heads 4 must be divisible"):
            apply_monkey_patch(model, ulysses_sp_size=3, use_remove_padding=False)


@pytest.mark.parametrize("heads,kv_heads,error", [(4, 2, "num_attention_heads 4"), (12, 4, "num_key_value_heads 4")])
def test_nested_multimodal_text_configuration_preserves_sequence_parallel_guards(heads, kv_heads, error):
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": heads * 8,
            "head_dim": 8,
            "intermediate_size": heads * 16,
            "num_hidden_layers": 1,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": heads * 8,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    assert not hasattr(config, "num_attention_heads")
    model = Qwen3VLForConditionalGeneration(config)
    with pytest.raises(AssertionError, match=error):
        apply_monkey_patch(model, ulysses_sp_size=3, use_remove_padding=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enabled", [False, True])
def test_qwen3next_router_keeps_rank_and_gradient_without_changing_weights(dtype, enabled):
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    config = Qwen3NextConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["full_attention"],
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )
    config.eqlen_router_fp32 = enabled
    model = Qwen3NextForCausalLM(config).to(dtype)
    apply_monkey_patch(model, use_remove_padding=False)
    gate = model.model.layers[0].mlp.gate
    first_forward = gate.forward.__func__
    apply_monkey_patch(model, use_remove_padding=False)
    assert gate.forward.__func__ is first_forward
    assert gate.weight.dtype == dtype
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[:, 0] = 1
        gate.weight[1, 1] = torch.finfo(dtype).eps / 4
    hidden = torch.zeros(1, 128, dtype=dtype)
    hidden[:, :2] = 1
    hidden.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = gate(hidden)
        logits = output[0] if isinstance(output, tuple) else output
    assert logits.dtype == (torch.float32 if enabled else torch.bfloat16)
    assert logits.argmax(-1).item() == (1 if enabled else 0)
    logits.sum().backward()
    for gradient in (gate.weight.grad, hidden.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
