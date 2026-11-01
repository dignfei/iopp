# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.equal_length import NATIVE_PAIR_KEY
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    core_client,
)
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import InprocClient, MPClient
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.skip_global_cleanup


def _request(request_id, client_index=0, params=None):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=params or SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        external_req_id="pair",
        client_index=client_index,
    )


def _ready_response(capability):
    return EngineCoreReadyResponse(
        max_model_len=128,
        num_gpu_blocks=16,
        block_size=16,
        dp_stats_address=None,
        dtype="float32",
        vllm_version="test",
        world_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_rank=0,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        instance_id="test",
        supports_lora=False,
        max_loras=0,
        native_equal_length_supported=bool(capability),
    )


def test_next_request_is_trailing_and_legacy_output_decodes():
    fields = msgspec.structs.fields(EngineCoreOutput)
    assert fields[-1].name == "eqlen_next_request"
    legacy_type = msgspec.defstruct(
        "LegacyEngineCoreOutput",
        [(field.name, field.type) for field in fields[:-1]],
        array_like=True,
    )
    output = EngineCoreOutput("old", [3], finish_reason=FinishReason.STOP)
    current_array = msgspec.msgpack.decode(msgspec.msgpack.encode(output))
    legacy_payload = msgspec.msgpack.encode(current_array[:-1])
    decoded = msgspec.msgpack.decode(legacy_payload, type=EngineCoreOutput)
    assert decoded == output
    assert decoded.eqlen_next_request is None
    legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(output), type=legacy_type)
    assert legacy.request_id == "old" and legacy.new_token_ids == [3]
    assert legacy.finish_reason == FinishReason.STOP


def test_next_request_round_trips_with_real_engine_codec():
    next_request = _request(
        "next", client_index=3, params=SamplingParams(max_tokens=2, seed=42)
    )
    outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                "old",
                [3],
                finish_reason=FinishReason.STOP,
                eqlen_next_request=next_request,
            )
        ]
    )
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(outputs))
    replacement = decoded.outputs[0].eqlen_next_request
    assert replacement is not None
    assert replacement.request_id == "next"
    assert replacement.prompt_token_ids == [1, 2]
    assert replacement.client_index == 3 and replacement.external_req_id == "pair"
    assert replacement.sampling_params.seed == 42
    assert replacement.sampling_params.max_tokens == 2


@pytest.mark.parametrize(
    "capabilities, supported",
    [
        ((True, True), True),
        ((True, False, True), False),
        ((False, True), False),
        ((True, None), False),
        ((None, True), False),
        ((True, "empty"), False),
        (("empty", True), False),
    ],
)
def test_ready_capability_requires_every_responder(capabilities, supported):
    client = object.__new__(MPClient)
    client._effective_attention_block_sizes = set()
    client._kv_event_sources = {}
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=0),
        model_config=SimpleNamespace(max_model_len=128),
    )
    client.stats_update_address = None
    assert client.supports_native_equal_length_pairs is False

    for capability in capabilities:
        payload = b""
        if capability != "empty":
            fields = msgspec.msgpack.decode(
                msgspec.msgpack.encode(_ready_response(capability))
            )
            if capability is None:
                del fields["native_equal_length_supported"]
            payload = msgspec.msgpack.encode(fields)
        client._apply_ready_response(payload)
    assert client.supports_native_equal_length_pairs is supported
    assert client.vllm_config.cache_config.block_size == 16


@pytest.mark.parametrize("capability", [True, False, None])
def test_inproc_capability_comes_from_scheduler(monkeypatch, capability):
    scheduler = SimpleNamespace()
    if capability is not None:
        scheduler.supports_native_equal_length_pairs = capability
    engine = SimpleNamespace(scheduler=scheduler, shutdown=Mock())
    monkeypatch.setattr(core_client, "EngineCore", lambda *args, **kwargs: engine)
    client = InprocClient(SimpleNamespace(), object, False)
    try:
        assert client.supports_native_equal_length_pairs is (capability is True)
    finally:
        client.shutdown()


@pytest.mark.parametrize("capability", [True, False, None])
def test_core_ready_reports_scheduler_capability(capability):
    ready = _ready_response(capability)
    core = object.__new__(EngineCoreProc)
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=1,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=128, dtype="float32", enable_sleep_mode=False
        ),
        cache_config=SimpleNamespace(
            **{
                name: getattr(ready, name)
                for name in (
                    "num_gpu_blocks",
                    "block_size",
                    "mamba_block_size",
                    "effective_attention_block_size",
                    "kv_cache_size_tokens",
                    "kv_cache_max_concurrency",
                )
            }
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=128),
        instance_id="test",
        lora_config=None,
        weight_transfer_config=None,
    )
    core.scheduler = SimpleNamespace(get_kv_event_publisher_config=lambda: None)
    if capability is not None:
        core.scheduler.supports_native_equal_length_pairs = capability
    core.frontend_stats_publish_address = None
    core.engine_index = 0
    assert core._make_ready_response().native_equal_length_supported is (
        capability is True
    )


def _core(peer=None, ordinary=None):
    core = object.__new__(EngineCoreProc)
    requests = {req.request_id: req for req in (peer, ordinary) if req is not None}

    def reject(request, error):
        rejected = [peer, request] if peer is not None else [request]
        for req in rejected:
            requests.pop(req.request_id, None)
            req.status = RequestStatus.FINISHED_ERROR
        return rejected

    core.scheduler = SimpleNamespace(
        add_request=Mock(side_effect=ValueError("invalid native descriptor")),
        reject_native_request=Mock(side_effect=reject),
        has_requests=lambda: bool(requests),
        requests=requests,
    )
    core.model_executor = Mock()
    core.engines_running = False
    core.batch_queue = deque()
    core.batch_queue_size = 2
    core.engine_index = 7
    core._pending_native_errors = {}
    core.output_queue = queue.Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    core.post_step = Mock()
    return core


@pytest.mark.parametrize("step", ["step", "step_with_batch_queue"])
@pytest.mark.parametrize("with_ordinary", [False, True])
def test_invalid_native_add_returns_pair_errors_without_model_work(step, with_ordinary):
    peer = Request.from_engine_core_request(_request("peer", client_index=1), None)
    ordinary = (
        Request.from_engine_core_request(_request("ordinary"), None)
        if with_ordinary
        else None
    )
    incoming = Request.from_engine_core_request(
        _request("incoming", client_index=2), None
    )
    incoming.sampling_params.extra_args = {NATIVE_PAIR_KEY: "malformed"}
    core = _core(peer, ordinary)
    core.step_fn = getattr(core, step)

    core._handle_client_request(EngineCoreRequestType.ADD, (incoming, 0))
    core.scheduler.reject_native_request.assert_called_once()
    assert core.scheduler.requests == ({"ordinary": ordinary} if with_ordinary else {})
    assert core.has_work()
    assert core._process_engine_step() is False
    routed = dict(core.output_queue.get_nowait() for _ in range(2))
    assert set(routed) == {1, 2}
    for client_index, request_id in ((1, "peer"), (2, "incoming")):
        outputs = routed[client_index]
        assert outputs.engine_index == 7
        assert outputs.finished_requests == {request_id}
        [output] = outputs.outputs
        assert output.request_id == request_id and output.new_token_ids == []
        assert output.finish_reason == FinishReason.ERROR
        assert output.stop_reason == "invalid native descriptor"
    assert not core._pending_native_errors
    assert core.has_work() is with_ordinary
    core.model_executor.execute_model.assert_not_called()
    core.model_executor.sample_tokens.assert_not_called()
    core.post_step.assert_called_once_with(False)


@pytest.mark.parametrize(
    "native, error",
    [(False, ValueError("ordinary error")), (True, RuntimeError("unexpected error"))],
)
def test_other_add_errors_keep_existing_exception_behavior(native, error):
    request = Request.from_engine_core_request(_request("request"), None)
    if native:
        request.sampling_params.extra_args = {NATIVE_PAIR_KEY: "malformed"}
    core = _core()
    core.scheduler.add_request.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        core.add_request(request)
    core.scheduler.reject_native_request.assert_not_called()
    assert not core._pending_native_errors


def _full_attention_spec(**changes):
    return FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32, **changes
    )


def _gdn_spec(**changes):
    return replace(
        MambaSpec(
            block_size=16,
            shapes=((2, 2), (2, 2)),
            dtypes=(torch.float32, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
        ),
        **changes,
    )


def _kv_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=0),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True, async_scheduling=False
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            kv_cache_layout="LBNHC",
            num_gpu_blocks=0,
            block_size=16,
        ),
        model_config=SimpleNamespace(
            max_model_len=128, is_diffusion=False, runner_type="generate"
        ),
        compilation_config=SimpleNamespace(
            compilation_time=0, encoder_compilation_time=0
        ),
        validate_block_size=Mock(),
        max_concurrent_batches=1,
        is_mm_encoder_only=False,
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "second_spec, wrapped, second_worker, max_model_len, supported",
    [
        pytest.param(_full_attention_spec(), False, False, 128, True, id="causal-fa"),
        pytest.param(
            _full_attention_spec(non_causal=True),
            True,
            False,
            128,
            False,
            id="uniform-hidden-non-causal",
        ),
        pytest.param(
            _full_attention_spec(sliding_window=8),
            True,
            False,
            128,
            False,
            id="uniform-hidden-sliding",
        ),
        pytest.param(
            _full_attention_spec(attention_chunk_size=8),
            True,
            False,
            128,
            False,
            id="uniform-hidden-chunked",
        ),
        pytest.param(
            _full_attention_spec(non_causal=True),
            False,
            True,
            128,
            False,
            id="unsafe-second-worker",
        ),
        pytest.param(_gdn_spec(), False, False, 128, True, id="causal-fa-gdn"),
        pytest.param(
            _gdn_spec(shapes=((2,),) * 3, dtypes=(torch.float32,) * 3),
            False,
            False,
            128,
            False,
            id="auxiliary-state",
        ),
        pytest.param(
            _gdn_spec(mamba_type=MambaAttentionBackendEnum.MAMBA2),
            False,
            False,
            128,
            False,
            id="unsupported-mamba",
        ),
        pytest.param(
            _gdn_spec(num_speculative_blocks=1),
            False,
            False,
            128,
            False,
            id="speculative-state",
        ),
        pytest.param(
            _gdn_spec(mamba_cache_mode="all"),
            False,
            False,
            128,
            False,
            id="unsupported-cache-mode",
        ),
        pytest.param(
            _gdn_spec(mamba_cache_mode="none"),
            False,
            False,
            128,
            False,
            id="none-state-block-too-short",
        ),
        pytest.param(
            _gdn_spec(mamba_cache_mode="none", block_size=128),
            False,
            False,
            128,
            True,
            id="none-state-covers-context",
        ),
        pytest.param(
            _gdn_spec(mamba_cache_mode="none", block_size=64),
            False,
            False,
            64,
            True,
            id="none-state-after-auto-fit",
        ),
    ],
)
def test_core_checks_original_worker_layers_before_simplification(
    monkeypatch, second_spec, wrapped, second_worker, max_model_len, supported
):
    causal = _full_attention_spec()
    layers = {"first": causal, "second": second_spec}
    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=layers)
    worker_specs = [{"group": uniform}] if wrapped else [layers]
    if second_worker:
        worker_specs = [{"first": causal}, {"second": second_spec}]
    worker_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(layers), uniform)],
    )
    config = _kv_config()
    core = object.__new__(EngineCore)
    core.model_executor = Mock()
    core.model_executor.get_kv_cache_specs.return_value = worker_specs
    core.model_executor.determine_available_memory.return_value = [1024] * len(
        worker_specs
    )
    core.collective_rpc = Mock()

    def group_specs(vllm_config, specs, memory):
        # Grouping can promote/replace the original dictionary entries.
        for worker in specs:
            worker.clear()
            worker["first"] = causal
        vllm_config.model_config.max_model_len = max_model_len
        return [worker_config] * len(specs)

    monkeypatch.setattr(core_module, "register_all_kvcache_specs", lambda _: None)
    monkeypatch.setattr(
        core_module,
        "resolve_kv_cache_layout",
        lambda *args: SimpleNamespace(name="LBNHC"),
    )
    monkeypatch.setattr(core_module, "get_kv_cache_configs", group_specs)
    monkeypatch.setattr(core_module, "update_kv_cache_capacity", lambda *args: None)
    monkeypatch.setattr(core_module.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", False)
    scheduler_config = core._initialize_kv_caches(config)

    # The real scheduler-config generator retains only the first (safe) layer.
    assert scheduler_config.kv_cache_groups[0].kv_cache_spec == causal
    assert core._native_exact_prefix_worker_specs_supported is supported
    assert not hasattr(config, "_native_exact_prefix_worker_specs_supported")
    assert not hasattr(
        config.scheduler_config, "_native_exact_prefix_worker_specs_supported"
    )
    if max_model_len < 128:
        core.collective_rpc.assert_called_once_with(
            "update_max_model_len", args=(max_model_len,)
        )


@pytest.mark.parametrize("supported", [True, False, None])
@pytest.mark.parametrize("custom_scheduler", [False, True])
def test_core_transfers_original_spec_gate_without_config_mutation(
    monkeypatch, supported, custom_scheduler
):
    class Scheduler:
        __slots__ = (
            () if custom_scheduler else ("_native_exact_prefix_worker_specs_supported",)
        )
        connector = ec_connector = None

        def __init__(self, **kwargs):
            pass

        @property
        def supports_native_equal_length_pairs(self):
            return not custom_scheduler and getattr(
                self, "_native_exact_prefix_worker_specs_supported", True
            )

        def get_kv_connector(self):
            return None

    config = _kv_config()
    config.scheduler_config.get_scheduler_cls = lambda: Scheduler

    def initialize(core, _):
        if supported is not None:
            core._native_exact_prefix_worker_specs_supported = supported
        return KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[])

    monkeypatch.setattr("vllm.plugins.load_general_plugins", lambda: None)
    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", initialize)
    monkeypatch.setattr(core_module, "StructuredOutputManager", lambda _: None)
    monkeypatch.setattr(
        core_module, "resolve_kv_cache_block_sizes", lambda *args: (16, 16)
    )
    monkeypatch.setattr(
        core_module, "engine_receiver_cache_from_config", lambda _: None
    )
    monkeypatch.setattr(core_module, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(core_module, "maybe_attach_gc_debug_callback", lambda: None)
    monkeypatch.setattr(core_module, "enable_envs_cache", lambda: None)
    monkeypatch.setattr(core_module.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", False)
    core = EngineCore(config, lambda _: Mock(), log_stats=False)
    if custom_scheduler:
        assert not hasattr(
            core.scheduler, "_native_exact_prefix_worker_specs_supported"
        )
    else:
        assert core.scheduler._native_exact_prefix_worker_specs_supported is (
            supported is True
        )
    assert core.scheduler.supports_native_equal_length_pairs is (
        supported is True and not custom_scheduler
    )
    assert not hasattr(config, "_native_exact_prefix_worker_specs_supported")
