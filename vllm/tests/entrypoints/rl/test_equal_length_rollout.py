# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-length paired rollouts, driven by a deterministic fake engine."""

import asyncio
import gc
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.entrypoints.rl.equal_length_rollout import (
    BranchRequest,
    PairedRollout,
    SegmentPair,
    _branch_params,
    generate_equal_length_pairs,
    generate_equal_length_pairs_async,
)
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.core.equal_length import (
    NATIVE_PAIR_KEY,
    EqualLengthPairScheduler,
    NativePairConfig,
    get_native_pair_config,
    with_native_pair_config,
)
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate

PROMPT = [1, 2, 3]


def test_native_pair_descriptor_preserves_user_parameters():
    params = SamplingParams(max_tokens=8, extra_args={"user": [1]})
    config = NativePairConfig("pair", 1, 3)
    native = with_native_pair_config(params, config)

    assert get_native_pair_config(native) == config
    assert get_native_pair_config(params) is None
    assert params.extra_args == {"user": [1]}
    assert native.extra_args is not None
    native.extra_args["user"].append(2)
    assert params.extra_args == {"user": [1]}


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {},
        {"group_id": "pair", "slot": 0},
        {"group_id": "pair", "slot": 0, "branch_index": 0, "extra": 1},
        {"group_id": "", "slot": 0, "branch_index": 0},
        {"group_id": 1, "slot": 0, "branch_index": 0},
        {"group_id": "pair", "slot": True, "branch_index": 0},
        {"group_id": "pair", "slot": 2, "branch_index": 0},
        {"group_id": "pair", "slot": 0, "branch_index": True},
        {"group_id": "pair", "slot": 0, "branch_index": -1},
    ],
)
def test_native_pair_descriptor_rejects_invalid_schema(descriptor):
    params = SamplingParams(max_tokens=8, extra_args={NATIVE_PAIR_KEY: descriptor})
    with pytest.raises(ValueError, match="descriptor"):
        get_native_pair_config(params)


class FakeEngine:
    """Branch ``b`` emits tokens ``1000 * (b + 1) + t`` and hits EOS after
    ``lengths[b]`` tokens unless ``max_tokens`` stops it first. Core branches
    advance together; ``speeds[b]`` controls frontend packet coalescing."""

    def __init__(self, lengths, default_length=100, speeds=None):
        self.lengths = lengths
        self.default_length = default_length
        self.speeds = speeds or {}
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True)
        )
        self.engine_core = SimpleNamespace(supports_native_equal_length_pairs=True)
        self.active: dict[str, dict] = {}
        self.submitted: list[tuple[str, list[int], SamplingParams]] = []
        self.branches: list[tuple[str, list[int], SamplingParams]] = []
        self.groups: dict[str, dict] = {}
        self.queues: dict[tuple[str, int], list[RequestOutput]] = {}
        self.delivered: list[RequestOutput] = []

    def add_request(self, request_id, prompt, params):
        self.submitted.append((request_id, prompt["prompt_token_ids"], params))
        config = get_native_pair_config(params)
        if config is None:
            request = BranchRequest(
                request_id, prompt["prompt_token_ids"], params.max_tokens, 99
            )
            self._add_branch(request_id, 0, request, params, native=False)
            return
        assert request_id == config.group_id
        assert config.branch_index == config.slot
        if request_id not in self.groups:
            scheduler = EqualLengthPairScheduler(
                prompt["prompt_token_ids"], params.max_tokens, request_id
            )
            self.groups[request_id] = {
                "scheduler": scheduler,
                "initial": scheduler.start(),
                "params": params,
            }
        request = self.groups[request_id]["initial"][config.slot]
        self._add_branch(request_id, config.slot, request, params)

    def _add_branch(self, group_id, slot, request, params, native=True):
        branch = request.branch_index
        eos_at = self.lengths.get(branch, self.default_length)
        self.active[request.request_id] = {
            "branch": branch,
            "length": min(eos_at, params.max_tokens),
            "reason": "stop" if eos_at <= params.max_tokens else "length",
            "emitted": 0,
            "group_id": group_id,
            "slot": slot,
            "native": native,
        }
        self.queues.setdefault((group_id, slot), [])
        self.branches.append((request.request_id, request.prompt_token_ids, params))

    def has_unfinished_requests(self):
        return bool(self.active) or any(self.queues.values())

    def advance_core(self):
        outputs = []
        for request_id, state in list(self.active.items()):
            start = state["emitted"]
            token_ids = tokens(state["branch"], start, start + 1)
            state["emitted"] += 1
            finished = state["emitted"] == state["length"]
            if finished:
                del self.active[request_id]
            reason = state["reason"] if finished else None
            completion = CompletionOutput(
                index=state["branch"] if state["native"] else 0,
                text="",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=[{t: Logprob(logprob=-t / 1000)} for t in token_ids],
                finish_reason=reason,
            )
            output = RequestOutput(
                state["group_id"], None, [-999], None, [completion], False
            )
            outputs.append((state, output))
            if not state["native"]:
                output.finished = finished
                continue
            group = self.groups[state["group_id"]]
            for request in group["scheduler"].on_output(
                request_id, token_ids, None, finished, reason
            ):
                occupied = {
                    s["slot"]
                    for s in self.active.values()
                    if s["group_id"] == state["group_id"]
                }
                [slot] = set((0, 1)) - occupied
                params = with_native_pair_config(
                    _branch_params(group["params"], request),
                    NativePairConfig(state["group_id"], slot, request.branch_index),
                )
                self._add_branch(state["group_id"], slot, request, params)
        for state, output in outputs:
            if state["native"]:
                output.finished = self.groups[state["group_id"]]["scheduler"].done
            self.queues[state["group_id"], state["slot"]].append(output)

    def pop_output(self, group_id, slot):
        queue = self.queues[group_id, slot]
        if not queue:
            return None
        if len(queue) < self.speeds.get(queue[-1].outputs[-1].index, 1) and not any(
            output.finished for output in queue
        ):
            return None
        output, *following = queue
        queue.clear()
        for delta in following:
            output.add(delta, aggregate=True)
        self.delivered.append(output)
        return output

    def step(self):
        self.advance_core()
        return [
            output
            for group_id, slot in self.queues
            if (output := self.pop_output(group_id, slot)) is not None
        ]

    def abort_request(self, request_ids):
        for request_id, state in list(self.active.items()):
            if state["group_id"] in request_ids:
                self.active.pop(request_id)
        for (group_id, _), queue in self.queues.items():
            if group_id in request_ids:
                queue.clear()


class FakeAsyncEngine:
    def __init__(self, engine: FakeEngine):
        self.engine = engine
        self.vllm_config = engine.vllm_config
        self.engine_core = engine.engine_core

    async def generate(self, prompt, params, request_id):
        self.engine.add_request(request_id, prompt, params)
        config = get_native_pair_config(params)
        assert config is not None
        finished = False
        try:
            while True:
                await asyncio.sleep(0)
                self.engine.advance_core()
                output = self.engine.pop_output(request_id, config.slot)
                if output is None:
                    continue
                finished = output.finished
                yield output
                if finished:
                    return
        finally:
            if not finished:
                self.engine.abort_request([request_id])


def tokens(branch: int, start: int, end: int) -> list[int]:
    return [1000 * (branch + 1) + t for t in range(start, end)]


# Branches 0 and 1 start from the prompt; each later branch starts where the
# previous pair was harvested. With max_tokens=12 the run is:
#   pair 0: b0 stops at 3                  vs b1[0:3]
#   pair 1: b2 stops at 4                  vs b1[3:7]
#   pair 2: b3 stops at 2                  vs b1[7:9]
#   pair 3: b1 stops (10 tokens, 1 left)   vs b4[0:1]
#   pair 4: b4 and b5 both reach the cap   (12 = 10 inherited + 2)
RELAY_LENGTHS = {0: 3, 1: 10, 2: 4, 3: 2}
RELAY_PAIRS = [
    (0, tokens(0, 0, 3), tokens(1, 0, 3), "stop", None),
    (3, tokens(2, 0, 4), tokens(1, 3, 7), "stop", None),
    (7, tokens(3, 0, 2), tokens(1, 7, 9), "stop", None),
    (9, tokens(1, 9, 10), tokens(4, 0, 1), "stop", None),
    (10, tokens(4, 1, 3), tokens(5, 0, 2), "length", "length"),
]


def summarize(rollout: PairedRollout):
    return [
        (
            p.prefix_len,
            p.plus_token_ids,
            p.minus_token_ids,
            p.plus_finish_reason,
            p.minus_finish_reason,
        )
        for p in rollout.pairs
    ]


@pytest.mark.parametrize("speeds", [{}, {1: 3}, {0: 5, 2: 3}, {4: 4, 5: 1}], ids=str)
def test_relay_harvests_equal_length_pairs_regardless_of_speed(speeds):
    engine = FakeEngine(RELAY_LENGTHS, speeds=speeds)
    [[rollout]] = generate_equal_length_pairs(
        engine, [PROMPT], SamplingParams(max_tokens=12)
    )

    assert summarize(rollout) == RELAY_PAIRS
    assert [len(t.response_token_ids) for t in rollout.trajectories()] == [
        3,
        7,
        9,
        10,
        12,
        12,
    ]
    assert len(engine.submitted) == 2
    assert len({request_id for request_id, _, _ in engine.submitted}) == 1


def test_new_branches_inherit_prefix_budget_and_distinct_seeds():
    engine = FakeEngine(RELAY_LENGTHS)
    params = SamplingParams(
        max_tokens=12,
        seed=7,
        repetition_penalty=1.2,
        repetition_detection=RepetitionDetectionParams(),
        extra_args={"test": True},
    )
    [[rollout]] = generate_equal_length_pairs(engine, [PROMPT], params)

    contexts = [PROMPT, PROMPT] + [
        PROMPT + sum((p.minus_token_ids for p in rollout.pairs[: i + 1]), [])
        for i in range(len(rollout.pairs) - 1)
    ]
    assert [prompt for _, prompt, _ in engine.branches] == contexts
    for _, prompt, branch_params in engine.branches:
        assert branch_params.n == 1
        assert branch_params.output_kind == RequestOutputKind.DELTA
        assert branch_params.max_tokens == 12 - (len(prompt) - len(PROMPT))
        assert branch_params.repetition_penalty == 1.2
        assert branch_params.extra_args is not None
        assert branch_params.extra_args["test"] is True
        assert get_native_pair_config(branch_params) is not None
    seeds = [branch_params.seed for _, _, branch_params in engine.branches]
    assert len(set(seeds)) == len(seeds)
    assert params.seed == 7 and params.output_kind == RequestOutputKind.CUMULATIVE


@pytest.mark.parametrize(
    "lengths, max_tokens, reason",
    [({0: 5, 1: 5}, 12, "stop"), ({}, 6, "length")],
    ids=["same_eos", "both_capped"],
)
def test_branches_ending_together_form_a_single_pair(lengths, max_tokens, reason):
    [[rollout]] = generate_equal_length_pairs(
        FakeEngine(lengths), [PROMPT], SamplingParams(max_tokens=max_tokens)
    )

    [pair] = rollout.pairs
    assert len(pair.plus_token_ids) == len(pair.minus_token_ids)
    assert (pair.plus_finish_reason, pair.minus_finish_reason) == (reason, reason)
    assert len(rollout.trajectories()) == 2


def test_logprobs_follow_segments():
    [[rollout]] = generate_equal_length_pairs(
        FakeEngine(RELAY_LENGTHS), [PROMPT], SamplingParams(max_tokens=12, logprobs=0)
    )

    for pair in rollout.pairs:
        assert pair.plus_logprobs == [-t / 1000 for t in pair.plus_token_ids]
        assert pair.minus_logprobs == [-t / 1000 for t in pair.minus_token_ids]
    for trajectory in rollout.trajectories():
        assert trajectory.logprobs == [-t / 1000 for t in trajectory.response_token_ids]


def test_async_driver_matches_sync_driver():
    params = SamplingParams(max_tokens=12)
    expected = generate_equal_length_pairs(
        FakeEngine(RELAY_LENGTHS), [PROMPT], params, num_pairs=2
    )[0]

    actual = asyncio.run(
        generate_equal_length_pairs_async(
            FakeAsyncEngine(FakeEngine(RELAY_LENGTHS, speeds={1: 2})),
            PROMPT,
            params,
            num_pairs=2,
        )
    )

    assert [summarize(r) for r in actual] == [summarize(r) for r in expected]


@pytest.mark.parametrize("driver", ["sync", "async"])
def test_native_pairs_submit_only_initial_slots(driver):
    engine = FakeEngine(RELAY_LENGTHS)
    params = SamplingParams(max_tokens=12, seed=7)
    if driver == "sync":
        results = generate_equal_length_pairs(engine, [PROMPT, [4, 5]], params, 2)
        expected_groups = 4
        assert [r.prompt_token_ids for row in results for r in row] == [
            PROMPT,
            PROMPT,
            [4, 5],
            [4, 5],
        ]
    else:
        results = asyncio.run(
            generate_equal_length_pairs_async(
                FakeAsyncEngine(engine), PROMPT, params, 2
            )
        )
        expected_groups = 2
        assert all(summarize(r) == RELAY_PAIRS for r in results)
    assert len(engine.submitted) == 2 * expected_groups
    groups: dict[str, set[int]] = {}
    for request_id, prompt, branch_params in engine.submitted:
        config = get_native_pair_config(branch_params)
        assert config is not None
        assert config.group_id == request_id
        assert config.branch_index == config.slot
        assert branch_params.output_kind == RequestOutputKind.DELTA
        groups.setdefault(request_id, set()).add(config.slot)
    assert len(groups) == expected_groups
    assert all(slots == {0, 1} for slots in groups.values())
    seeds = [params.seed for _, _, params in engine.submitted]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("driver", ["sync", "async"])
def test_coalesced_native_branches_use_completion_indexes(driver):
    engine = FakeEngine(RELAY_LENGTHS, speeds=dict.fromkeys(range(6), 100))
    params = SamplingParams(max_tokens=12, logprobs=0)
    if driver == "sync":
        [[rollout]] = generate_equal_length_pairs(engine, [PROMPT], params)
    else:
        [rollout] = asyncio.run(
            generate_equal_length_pairs_async(FakeAsyncEngine(engine), PROMPT, params)
        )
    assert len(engine.delivered) == 2
    assert all(output.finished for output in engine.delivered)
    assert any(len(output.outputs) > 1 for output in engine.delivered)
    assert all(output.prompt_token_ids == [-999] for output in engine.delivered)
    assert rollout.prompt_token_ids == PROMPT
    assert summarize(rollout) == RELAY_PAIRS
    assert len(engine.submitted) == 2
    for pair in rollout.pairs:
        assert pair.plus_logprobs == [-t / 1000 for t in pair.plus_token_ids]
        assert pair.minus_logprobs == [-t / 1000 for t in pair.minus_token_ids]


@pytest.mark.parametrize("driver", ["sync", "async"])
@pytest.mark.parametrize(
    "malformed, error, message",
    [
        ("index", RuntimeError, "branch index"),
        ("duplicate_eos", RuntimeError, "after EOS"),
        ("missing_branch", RuntimeError, "Incomplete native"),
        ("logprobs", ValueError, "Missing logprobs"),
    ],
)
def test_malformed_native_packets_fail_instead_of_returning_a_rollout(
    driver, malformed, error, message
):
    class MalformedEngine(FakeEngine):
        def pop_output(self, group_id, slot):
            output = super().pop_output(group_id, slot)
            if output is not None and slot == 0:
                if malformed == "index":
                    output.outputs[0].index = -1
                elif malformed == "duplicate_eos":
                    output.outputs.append(output.outputs[0])
                elif malformed == "missing_branch":
                    output.outputs.pop()
                else:
                    output.outputs[0].logprobs = []
            return output

    engine = MalformedEngine(RELAY_LENGTHS, speeds=dict.fromkeys(range(6), 100))
    params = SamplingParams(max_tokens=12, logprobs=0)
    with pytest.raises(error, match=message):
        if driver == "sync":
            generate_equal_length_pairs(engine, [PROMPT], params)
        else:
            asyncio.run(
                generate_equal_length_pairs_async(
                    FakeAsyncEngine(engine), PROMPT, params
                )
            )
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize("driver", ["sync", "async"])
def test_native_pairs_require_engine_capability_before_submitting(driver):
    engine = FakeEngine({})
    engine.engine_core.supports_native_equal_length_pairs = False
    with pytest.raises(ValueError, match="does not support native"):
        if driver == "sync":
            generate_equal_length_pairs(engine, [PROMPT], SamplingParams(max_tokens=8))
        else:
            asyncio.run(
                generate_equal_length_pairs_async(
                    FakeAsyncEngine(engine), PROMPT, SamplingParams(max_tokens=8)
                )
            )
    assert not engine.submitted


@pytest.mark.parametrize("driver", ["sync", "async"])
@pytest.mark.parametrize("reason", ["abort", "error"])
def test_native_branch_failure_aborts_the_pair(driver, reason):
    class AbortedEngine(FakeEngine):
        def pop_output(self, group_id, slot):
            output = super().pop_output(group_id, slot)
            if output is not None:
                output.outputs[0].finish_reason = reason
            return output

    engine = AbortedEngine({})
    with pytest.raises(RuntimeError, match=reason):
        if driver == "sync":
            generate_equal_length_pairs(engine, [PROMPT], SamplingParams(max_tokens=8))
        else:
            asyncio.run(
                generate_equal_length_pairs_async(
                    FakeAsyncEngine(engine), PROMPT, SamplingParams(max_tokens=8)
                )
            )
    assert not engine.has_unfinished_requests()


def test_async_native_slot_cannot_end_without_its_final_output():
    class TruncatedEngine(FakeAsyncEngine):
        async def generate(self, prompt, params, request_id):
            async for output in super().generate(prompt, params, request_id):
                yield output
                return

    engine = FakeEngine({})
    with pytest.raises(RuntimeError, match="before its final output"):
        asyncio.run(
            generate_equal_length_pairs_async(
                TruncatedEngine(engine), PROMPT, SamplingParams(max_tokens=8)
            )
        )
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize("exit_kind", ["cancel", "double_cancel", "branch_error"])
def test_async_exit_waits_for_child_cleanup_and_preserves_error(exit_kind):
    async def run():
        failure = RuntimeError("injected branch failure")
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: unhandled.append(context))

        class BlockingEngine(FakeAsyncEngine):
            def __init__(self):
                super().__init__(FakeEngine({}))
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.cleaning = asyncio.Event()
                self.allow_cleanup = asyncio.Event()

            async def generate(self, prompt, params, request_id):
                self.engine.add_request(request_id, prompt, params)
                if len(self.engine.active) == 2:
                    self.started.set()
                config = get_native_pair_config(params)
                assert config is not None
                failing_branch = exit_kind == "branch_error" and config.slot == 0
                try:
                    await self.release.wait()
                    if failing_branch:
                        raise failure
                    await asyncio.Event().wait()
                    yield
                finally:
                    if not failing_branch:
                        self.cleaning.set()
                        await self.allow_cleanup.wait()
                    self.engine.active.pop(f"{request_id}-{config.branch_index}", None)

        engine = BlockingEngine()
        task = asyncio.create_task(
            generate_equal_length_pairs_async(
                engine, PROMPT, SamplingParams(max_tokens=8)
            )
        )
        try:
            await engine.started.wait()
            if exit_kind == "branch_error":
                engine.release.set()
            else:
                task.cancel()
            await engine.cleaning.wait()
            if exit_kind == "double_cancel":
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert engine.engine.active
            engine.allow_cleanup.set()
            expected = (
                RuntimeError if exit_kind == "branch_error" else asyncio.CancelledError
            )
            with pytest.raises(expected) as caught:
                await task
            if exit_kind == "branch_error":
                assert caught.value is failure
            assert not engine.engine.active
            caught.value.__traceback__ = None
            await asyncio.sleep(0)
            gc.collect()
            assert not unhandled
        finally:
            engine.allow_cleanup.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            loop.set_exception_handler(previous_handler)

    asyncio.run(run())


@pytest.mark.parametrize("driver", ["sync", "async"])
@pytest.mark.parametrize(
    "params, num_pairs, error",
    [
        pytest.param(SamplingParams(n=2, max_tokens=8), 1, "n=1", id="n>1"),
        pytest.param(
            SamplingParams(max_tokens=None), 1, "max_tokens", id="no_max_tokens"
        ),
        pytest.param(SamplingParams(max_tokens=8), 0, "num_pairs", id="no_pairs"),
        *[
            pytest.param(
                SamplingParams(max_tokens=8, **{name: value}), 1, name, id=name
            )
            for name, value in (
                ("presence_penalty", 0.5),
                ("frequency_penalty", -0.5),
                ("stop", ["AB"]),
                ("structured_outputs", StructuredOutputsParams(regex="a*")),
                ("bad_words", ["AB"]),
                ("_bad_words_token_ids", [[1, 2]]),
                ("thinking_token_budget", 0),
                (
                    "repetition_detection",
                    RepetitionDetectionParams(max_pattern_size=2, min_count=2),
                ),
                ("trace_decode_token_ids", [1, 2]),
            )
        ],
    ],
)
def test_rejects_invalid_requests(params, num_pairs, error, driver):
    engine = FakeEngine({})
    engine.engine_core.supports_native_equal_length_pairs = False
    with pytest.raises(ValueError, match=error):
        if driver == "sync":
            generate_equal_length_pairs(engine, [PROMPT], params, num_pairs)
        else:
            asyncio.run(
                generate_equal_length_pairs_async(
                    FakeAsyncEngine(engine), PROMPT, params, num_pairs
                )
            )
    assert not engine.submitted


@pytest.mark.parametrize("inherited_tokens", [2, 6, 7])
def test_inherited_prefix_preserves_min_tokens_stop_mask(inherited_tokens):
    params = SamplingParams(max_tokens=8, min_tokens=6, stop_token_ids=[0])
    request = BranchRequest(
        "restart", PROMPT + [1] * inherited_tokens, 8 - inherited_tokens, 2
    )
    restarted = _branch_params(params, request, 0)
    processor = MinTokensLogitsProcessor(VllmConfig(), torch.device("cpu"), False)

    for global_position in range(inherited_tokens, 8):
        processor.update_state(
            BatchUpdate(
                batch_size=2,
                removed=(),
                added=[
                    (0, params, PROMPT, [1] * global_position),
                    (
                        1,
                        restarted,
                        request.prompt_token_ids,
                        [1] * (global_position - inherited_tokens),
                    ),
                ],
                moved=(),
            )
        )
        logits = processor.apply(torch.zeros((2, 2)))
        assert torch.isneginf(logits[0, 0]).item() == (global_position < 6)
        assert torch.equal(logits[0], logits[1])
    assert restarted.max_tokens is not None
    assert restarted.min_tokens <= restarted.max_tokens
    assert params.min_tokens == 6 and params.max_tokens == 8


@pytest.mark.parametrize(
    "failure_stage, cleanup_fails",
    [("submit", False), ("partial_submit", False), ("step", False), ("step", True)],
    ids=["submit", "partial_submit", "step", "abort_error"],
)
def test_sync_failure_aborts_own_requests_and_preserves_error(
    failure_stage, cleanup_fails
):
    failure = RuntimeError("injected engine failure")

    class FailingEngine(FakeEngine):
        def __init__(self):
            super().__init__({})
            self.aborted = []

        def add_request(self, request_id, prompt, params):
            invalid = prompt["prompt_token_ids"] == [-1]
            if invalid and failure_stage == "submit":
                raise failure
            super().add_request(request_id, prompt, params)
            if invalid:
                raise failure
            return "internal-" + request_id

        def step(self):
            raise failure

        def abort_request(self, request_ids, internal=False):
            assert not internal
            self.aborted.extend(request_ids)
            super().abort_request(request_ids)
            if cleanup_fails:
                raise RuntimeError("injected abort failure")

    engine = FailingEngine()
    params = SamplingParams(max_tokens=8)
    engine.add_request("unrelated-99", {"prompt_token_ids": PROMPT}, params)
    prompts = [PROMPT, [-1]] if "submit" in failure_stage else [PROMPT]
    with pytest.raises(RuntimeError) as caught:
        generate_equal_length_pairs(engine, prompts, params)
    assert caught.value is failure
    assert set(engine.active) == {"unrelated-99"}
    assert engine.aborted
    assert all(request_id.startswith("eqlen-") for request_id in engine.aborted)


def _example_rollout() -> PairedRollout:
    # Four pairs whose segments have lengths 1, 2, 1, 2.
    pairs, prefix = [], 0
    for i, length in enumerate([1, 2, 1, 2]):
        last = i == 3
        pairs.append(
            SegmentPair(
                prefix_len=prefix,
                plus_token_ids=[10 * i + 1] * length,
                minus_token_ids=[10 * i + 2] * length,
                plus_finish_reason="stop",
                minus_finish_reason="stop" if last else None,
            )
        )
        prefix += length
    return PairedRollout(PROMPT, pairs)


# Only the fourth trajectory (o1- o2- o3- o4+) is correct: its reward reaches
# o1-, o2- and o3- through max aggregation, while every other trajectory only
# determines the reward of its own final segment.
EXAMPLE_REWARDS = [0, 0, 0, 1, 0]


def test_minus_rewards_aggregate_over_extensions():
    rollout = _example_rollout()

    assert rollout.segment_rewards(EXAMPLE_REWARDS) == [
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 0),
    ]
    assert rollout.segment_rewards(EXAMPLE_REWARDS, "mean") == pytest.approx(
        [(0, 0.25), (0, 1 / 3), (0, 0.5), (1, 0)]
    )
    with pytest.raises(ValueError):
        rollout.segment_rewards(EXAMPLE_REWARDS[:-1])


def test_each_segment_enters_the_loss_once_with_its_pair_advantage():
    samples = _example_rollout().training_samples(EXAMPLE_REWARDS)

    assert [s.response_token_ids for s in samples] == [
        [1],
        [2, 11, 11],
        [2, 12, 12, 21],
        [2, 12, 12, 22, 31, 31],
        [2, 12, 12, 22, 32, 32],
    ]
    assert [s.loss_mask for s in samples] == [
        [True],
        [True, True, True],
        [False, True, True, True],
        [False, False, False, True, True, True],
        [False, False, False, False, True, True],
    ]
    assert [s.advantages for s in samples] == [
        [-1.0],
        [1.0, -1.0, -1.0],
        [0.0, 1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
    ]
    assert [s.segment_ids for s in samples] == [
        [0],
        [1, 2, 2],
        [-1, 3, 3, 4],
        [-1, -1, -1, 5, 6, 6],
        [-1, -1, -1, -1, 7, 7],
    ]
    assert [s.loss_weights for s in samples] == [
        [0.125],
        [0.125, 0.0625, 0.0625],
        [0.0, 0.0625, 0.0625, 0.125],
        [0.0, 0.0, 0.0, 0.125, 0.0625, 0.0625],
        [0.0, 0.0, 0.0, 0.0, 0.0625, 0.0625],
    ]


def test_pairs_with_equal_rewards_are_skipped():
    samples = _example_rollout().training_samples([1, 1, 1, 1, 1])

    assert not any(any(s.loss_mask) for s in samples)
    assert not any(any(s.advantages) for s in samples)
    assert all(all(segment_id == -1 for segment_id in s.segment_ids) for s in samples)
    assert not any(any(s.loss_weights) for s in samples)


def test_rloo_advantage_is_the_reward_difference():
    pair = SegmentPair(0, [5], [6], "stop", "stop")
    [plus, minus] = PairedRollout(PROMPT, [pair]).training_samples(
        [0.2, 0.5], advantage="rloo"
    )

    assert plus.advantages == pytest.approx([-0.3])
    assert minus.advantages == pytest.approx([0.3])


def test_training_metadata_matches_pair_objective_and_segment_ratios():
    short = PairedRollout(PROMPT, _example_rollout().pairs[:2])
    short.pairs[-1].minus_finish_reason = "stop"
    subgroups = [(_example_rollout(), EXAMPLE_REWARDS), (short, [1, 1, 0])]
    weighted_loss = pair_loss = 0.0

    def token_loss(token, advantage):
        ratio = math.exp(token / 100)
        clipped_ratio = min(1.1, max(0.9, ratio))
        return -min(ratio * advantage, clipped_ratio * advantage)

    for rollout, rewards in subgroups:
        segment_log_ratios: dict[int, list[float]] = {}
        for sample in rollout.training_samples(rewards):
            for token, advantage, segment_id, weight in zip(
                sample.response_token_ids,
                sample.advantages,
                sample.segment_ids,
                sample.loss_weights,
                strict=True,
            ):
                weighted_loss += weight * token_loss(token, advantage)
                if segment_id >= 0:
                    segment_log_ratios.setdefault(segment_id, []).append(token / 100)

        expected_ratios = {}
        subgroup_loss = 0.0
        for i, (pair, (plus_reward, minus_reward)) in enumerate(
            zip(rollout.pairs, rollout.segment_rewards(rewards), strict=True)
        ):
            if plus_reward == minus_reward:
                continue
            advantage = 1.0 if plus_reward > minus_reward else -1.0
            for side, tokens in enumerate((pair.plus_token_ids, pair.minus_token_ids)):
                subgroup_loss += sum(
                    token_loss(token, advantage if side == 0 else -advantage)
                    for token in tokens
                ) / (2 * len(tokens))
                expected_ratios[2 * i + side] = math.exp(
                    sum(token / 100 for token in tokens) / len(tokens)
                )
        pair_loss += subgroup_loss / len(rollout.pairs)
        assert {
            segment_id: math.exp(sum(values) / len(values))
            for segment_id, values in segment_log_ratios.items()
        } == pytest.approx(expected_ratios)

    assert weighted_loss / len(subgroups) == pytest.approx(pair_loss / len(subgroups))
    empty = PairedRollout(PROMPT, [SegmentPair(0, [], [], "stop", "stop")])
    assert all(
        not sample.loss_weights and not sample.segment_ids
        for sample in empty.training_samples([1, 0])
    )
