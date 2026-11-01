# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Padding helpers with an optional FlashAttention implementation."""

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Resolve padding helpers; the fallback provides no attention kernels."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if missing != "flash_attn" and not missing.startswith(("flash_attn.", "flash_attn_")):
                raise
            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Select rows with autograd support using available padding helpers.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Restore selected tokens to padded batch and sequence dimensions.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Rearrange tensor dimensions using the padding helper implementation.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Pack selected tokens and return indices, offsets, and length metadata.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
