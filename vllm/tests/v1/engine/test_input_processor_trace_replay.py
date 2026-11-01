# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.equal_length import (
    NativePairConfig,
    with_native_pair_config,
)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.skip_global_cleanup


def _make_request(params: SamplingParams) -> Request:
    return Request(
        request_id="req-0",
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
    )


def _normalize(params: SamplingParams, prompt_len: int, max_model_len: int) -> None:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len)
    )
    InputProcessor._normalize_trace_replay_params(processor, params, prompt_len)


def test_normalize_trace_replay_params():
    params = SamplingParams(
        max_tokens=16,
        min_tokens=10,
        stop_token_ids=[20],
        trace_decode_token_ids=[10, 20, 30],
    )
    params.update_from_generation_config({}, eos_token_id=20)

    _normalize(params, prompt_len=3, max_model_len=128)

    assert params.max_tokens == 3
    assert params.min_tokens == 0
    assert params.ignore_eos is True
    assert params.eos_token_id is None
    assert params.stop_token_ids == []
    assert params.all_stop_token_ids == set()

    request = _make_request(params)
    assert request.max_tokens == 3
    request.append_output_token_ids([10, 20, 30])
    assert check_stop(request, max_model_len=128)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_trace_longer_than_remaining_context_is_truncated():
    """The trace is staged into a max_model_len-wide row, so it must be cut.

    An explicitly set max_tokens is never clamped to max_model_len, so it cannot
    bound the staged write.
    """
    params = SamplingParams(max_tokens=100, trace_decode_token_ids=list(range(20)))

    _normalize(params, prompt_len=6, max_model_len=10)

    assert params.trace_decode_token_ids == [0, 1, 2, 3]
    assert params.max_tokens == 4


def _validate(enable_trace_replay: bool) -> None:
    """Run _validate_params' trace gating with the rest of verify() stubbed."""
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            return_sampling_mask=False,
            enable_trace_replay=enable_trace_replay,
            is_diffusion=False,
        ),
        vllm_config=SimpleNamespace(
            reasoning_config=None,
            # Engine has no watermark configuration.
            _check_supports_watermarking=lambda params: False,
        ),
        resolve_watermarking=lambda params: False,
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=None,
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
    )
    params = SamplingParams(trace_decode_token_ids=[1, 2, 3])
    with patch.object(SamplingParams, "verify"):
        InputProcessor._validate_params(processor, params, ("generate",))


def test_trace_request_rejected_when_feature_disabled():
    with pytest.raises(VLLMValidationError, match="--enable-trace-replay"):
        _validate(enable_trace_replay=False)


def test_trace_request_accepted_when_feature_enabled():
    _validate(enable_trace_replay=True)


def test_trace_replay_requires_v2_model_runner():
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_trace_replay=True),
        use_v2_model_runner=False,
    )

    with pytest.raises(ValueError, match="trace replay requires Model Runner V2"):
        VllmConfig._verify_trace_replay_config(config)


def _native_input():
    params = with_native_pair_config(
        SamplingParams(max_tokens=16, output_kind=RequestOutputKind.DELTA),
        NativePairConfig("pair", 0, 0),
    )
    request = EngineCoreRequest(
        request_id="pair",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor = SimpleNamespace(
        native_pair_supported=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=6),
    )
    return processor, request


def test_native_pair_input_accepts_exact_budget_without_mutating_params():
    processor, request = _native_input()
    before = request.sampling_params.clone()
    InputProcessor.validate_native_pair_request(processor, request)
    assert request.sampling_params == before
    # The same check also works after ID assignment for direct callers.
    request.external_req_id = request.request_id
    request.request_id += "-internal"
    InputProcessor.validate_native_pair_request(processor, request)


@pytest.mark.parametrize(
    "invalid",
    [
        "unsupported",
        "group",
        "renewal",
        "n",
        "output_kind",
        "empty",
        "embeds",
        "multimodal",
        "resumable",
        "prompt_logprobs",
        "prompt_logprob_token_ids",
        "watermarking",
        "history",
        "step_budget",
    ],
)
def test_native_pair_input_rejects_before_submission(invalid):
    processor, request = _native_input()
    params = request.sampling_params
    if invalid == "unsupported":
        processor.native_pair_supported = False
    elif invalid == "group":
        request.request_id = "another-pair"
    elif invalid == "renewal":
        request.sampling_params = with_native_pair_config(
            params, NativePairConfig("pair", 0, 2)
        )
    elif invalid == "n":
        params.n = 2
    elif invalid == "output_kind":
        params.output_kind = RequestOutputKind.FINAL_ONLY
    elif invalid == "empty":
        request.prompt_token_ids = []
    elif invalid == "embeds":
        import torch

        request.prompt_embeds = torch.zeros(3, 2)
    elif invalid == "multimodal":
        request.mm_features = [object()]
    elif invalid == "resumable":
        request.resumable = True
    elif invalid == "history":
        params.presence_penalty = 1
    elif invalid == "step_budget":
        processor.scheduler_config.max_num_batched_tokens = 5
    else:
        setattr(params, invalid, True if invalid == "watermarking" else [])
    with pytest.raises((ValueError, VLLMValidationError)):
        InputProcessor.validate_native_pair_request(processor, request)


def test_native_pair_input_leaves_ordinary_requests_unchanged():
    _, request = _native_input()
    request.sampling_params.extra_args = {"user_option": True}
    request.resumable = True
    # No capability/config attributes are needed for the ordinary path.
    InputProcessor.validate_native_pair_request(SimpleNamespace(), request)


def test_native_pair_input_requires_unique_internal_ids(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
    processor, request = _native_input()
    with pytest.raises(VLLMValidationError, match="randomization"):
        InputProcessor.validate_native_pair_request(processor, request)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_raw_request_cannot_bypass_frontend_capability(asynchronous):
    """Deprecated CoreRequest inputs must also reject before reserving slots."""
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    processor, request = _native_input()
    processor.native_pair_supported = False
    processor.apply_watermarking = lambda params, watermarking: None
    processor.resolve_watermarking = lambda params: False
    engine = (AsyncLLM if asynchronous else LLMEngine).__new__(
        AsyncLLM if asynchronous else LLMEngine
    )
    engine.input_processor = processor
    engine.output_processor = MagicMock()
    engine.engine_core = MagicMock()
    engine.engine_core.resources.engine_dead = False
    engine.output_handler = None
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    with pytest.raises(VLLMValidationError, match="does not support"):
        if asynchronous:
            await engine.add_request("pair", request, request.params)
        else:
            engine.add_request("pair", request, request.params)
    assert request.external_req_id is None
    engine.output_processor.add_request.assert_not_called()
    engine.engine_core.add_request.assert_not_called()
    engine.engine_core.add_request_async.assert_not_called()
