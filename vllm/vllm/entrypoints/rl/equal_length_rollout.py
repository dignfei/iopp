# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-length paired rollouts for group-relative RL.

Two branches decode from a shared context. When the shorter branch terminates
after ``T`` tokens, its tokens and the first ``T`` tokens of the other branch
form an equal-length segment pair ``(o+, o-)``. The ``T`` tokens of ``o-`` are
appended to the context, and the freed slot starts a new branch from it while
the other branch keeps decoding. This repeats until both branches terminate at
the same length, so every pair compares segments of identical length.

Each pair submits two initial requests. The engine renews a terminated branch
in its freed slot, reusing the continuing branch's computed KV prefix.

Sampling options whose output history cannot be restored across inherited
prefixes are rejected before any requests are submitted.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.inputs.llm import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.core.equal_length import (
    AdvantageEstimator as AdvantageEstimator,
)
from vllm.v1.core.equal_length import (
    Aggregation as Aggregation,
)
from vllm.v1.core.equal_length import (
    BranchRequest as BranchRequest,
)
from vllm.v1.core.equal_length import (
    EqualLengthPairScheduler as EqualLengthPairScheduler,
)
from vllm.v1.core.equal_length import NativePairConfig, with_native_pair_config
from vllm.v1.core.equal_length import (
    PairedRollout as PairedRollout,
)
from vllm.v1.core.equal_length import (
    SegmentPair as SegmentPair,
)
from vllm.v1.core.equal_length import (
    TrainingSample as TrainingSample,
)
from vllm.v1.core.equal_length import (
    Trajectory as Trajectory,
)
from vllm.v1.core.equal_length import (
    branch_sampling_params as _branch_params,
)
from vllm.v1.core.equal_length import (
    validate_equal_length_sampling_params as _validate,
)

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)

PromptTokens = list[int] | TokensPrompt

# Spacing between seeds of different pairs so their branches never collide.
_SEED_STRIDE = 1 << 20


def _prompt_token_ids(prompt: PromptTokens) -> list[int]:
    if isinstance(prompt, dict):
        return list(prompt["prompt_token_ids"])
    return list(prompt)


def _sampled_logprobs(
    completion: CompletionOutput,
) -> tuple[list[int], list[float] | None]:
    token_ids = list(completion.token_ids)
    if completion.logprobs is None:
        return token_ids, None
    logprobs = []
    for token_id, position in zip(token_ids, completion.logprobs):
        if position is None:
            return token_ids, None
        logprobs.append(position[token_id].logprob)
    return token_ids, logprobs


def _require_native_pairs(engine) -> None:
    if not getattr(
        getattr(engine, "engine_core", None),
        "supports_native_equal_length_pairs",
        False,
    ):
        raise ValueError("This engine does not support native equal-length pairs.")


@dataclass
class _BranchOutput:
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None
    finish_reason: str | None = None


class _NativePair:
    def __init__(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        group_id: str,
        track_logprobs: bool,
    ):
        self.group_id = group_id
        self.scheduler = EqualLengthPairScheduler(
            prompt_token_ids, max_tokens, group_id, track_logprobs
        )
        self.initial_requests = self.scheduler.start()
        self.branches: dict[int, _BranchOutput] = {}
        self.closed_slots = 0

    def receive(self, output: RequestOutput) -> None:
        if output.request_id != self.group_id:
            raise RuntimeError("Native equal-length output has the wrong group ID.")
        for completion in output.outputs:
            index = completion.index
            if type(index) is not int or index < 0:
                raise RuntimeError("Invalid native equal-length branch index.")
            branch = self.branches.setdefault(
                index,
                _BranchOutput(logprobs=[] if self.scheduler.track_logprobs else None),
            )
            if branch.finish_reason is not None:
                raise RuntimeError(
                    "Native equal-length branch produced output after EOS."
                )
            token_ids, logprobs = _sampled_logprobs(completion)
            branch.token_ids.extend(token_ids)
            if branch.logprobs is not None:
                if logprobs is None or len(logprobs) != len(token_ids):
                    raise ValueError(f"Missing logprobs for branch {index}.")
                branch.logprobs.extend(logprobs)
            reason = completion.finish_reason
            if reason is not None and reason not in ("stop", "length"):
                raise RuntimeError(f"Native equal-length branch ended with {reason!r}.")
            branch.finish_reason = reason

    def close_slot(self) -> bool:
        self.closed_slots += 1
        if self.closed_slots > 2:
            raise RuntimeError("Native equal-length pair closed too many slots.")
        return self.closed_slots == 2

    def rollout(self) -> PairedRollout:
        requests = {r.branch_index: r for r in self.initial_requests}
        for index, branch in sorted(self.branches.items()):
            request = requests.pop(index, None)
            if request is None or branch.finish_reason is None:
                raise RuntimeError("Incomplete native equal-length branch outputs.")
            for new in self.scheduler.on_output(
                request.request_id,
                branch.token_ids,
                branch.logprobs,
                True,
                branch.finish_reason,
            ):
                requests[new.branch_index] = new
        if not self.scheduler.done or requests:
            raise RuntimeError(
                "Native equal-length pair ended before all branches finished."
            )
        return self.scheduler.rollout

    def params(
        self,
        sampling_params: SamplingParams,
        request: BranchRequest,
        seed_offset: int,
    ) -> SamplingParams:
        return with_native_pair_config(
            _branch_params(sampling_params, request, seed_offset),
            NativePairConfig(self.group_id, request.branch_index, request.branch_index),
        )


def generate_equal_length_pairs(
    engine: "LLMEngine",
    prompts: Sequence[PromptTokens],
    sampling_params: SamplingParams,
    num_pairs: int = 1,
) -> list[list[PairedRollout]]:
    """Generate ``num_pairs`` equal-length pair rollouts for every prompt.

    Drives ``engine`` until all rollouts complete, so it must not be serving
    other requests concurrently.

    Returns:
        ``result[p][k]`` is the ``k``-th pair rollout of ``prompts[p]``.

    """
    max_tokens = _validate(sampling_params, num_pairs)
    _require_native_pairs(engine)
    track_logprobs = sampling_params.logprobs is not None
    batch_id = random_uuid()

    pairs: list[list[_NativePair]] = []
    owners: dict[str, _NativePair] = {}

    try:
        for p, prompt in enumerate(prompts):
            row = []
            for k in range(num_pairs):
                pair = _NativePair(
                    _prompt_token_ids(prompt),
                    max_tokens,
                    f"eqlen-{batch_id}-{p}-{k}",
                    track_logprobs,
                )
                seed_offset = (p * num_pairs + k) * _SEED_STRIDE
                owners[pair.group_id] = pair
                for request in pair.initial_requests:
                    engine.add_request(
                        pair.group_id,
                        TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                        pair.params(sampling_params, request, seed_offset),
                    )
                row.append(pair)
            pairs.append(row)

        pending = sum(len(row) for row in pairs)
        while pending:
            if not engine.has_unfinished_requests():
                raise RuntimeError("Engine went idle before all pairs completed.")
            for output in engine.step():
                if not isinstance(output, RequestOutput):
                    continue
                current_pair = owners.get(output.request_id)
                if current_pair is None:
                    continue
                current_pair.receive(output)
                if output.finished and current_pair.close_slot():
                    pending -= 1
        return [[pair.rollout() for pair in row] for row in pairs]
    except BaseException:
        try:
            engine.abort_request(list(owners))
        except Exception:
            logger.exception("Failed to abort equal-length rollout requests.")
        raise


async def generate_equal_length_pairs_async(
    engine: "EngineClient",
    prompt: PromptTokens,
    sampling_params: SamplingParams,
    num_pairs: int = 1,
    request_id_prefix: str | None = None,
) -> list[PairedRollout]:
    """Generate ``num_pairs`` equal-length pair rollouts for one prompt."""
    max_tokens = _validate(sampling_params, num_pairs)
    _require_native_pairs(engine)
    track_logprobs = sampling_params.logprobs is not None
    prefix = request_id_prefix or f"eqlen-{random_uuid()}"
    prompt_token_ids = _prompt_token_ids(prompt)

    queue: asyncio.Queue[tuple[int, RequestOutput | None] | BaseException] = (
        asyncio.Queue()
    )
    tasks: set[asyncio.Task] = set()

    async def consume(k: int, pair: _NativePair, request: BranchRequest) -> None:
        try:
            finished = False
            async for output in engine.generate(
                TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                pair.params(sampling_params, request, k * _SEED_STRIDE),
                pair.group_id,
            ):
                finished = output.finished
                queue.put_nowait((k, output))
            if not finished:
                raise RuntimeError(
                    "Native equal-length slot ended before its final output."
                )
            queue.put_nowait((k, None))
        except BaseException as e:
            queue.put_nowait(e)
            raise

    pairs = [
        _NativePair(prompt_token_ids, max_tokens, f"{prefix}-{k}", track_logprobs)
        for k in range(num_pairs)
    ]
    try:
        for k, pair in enumerate(pairs):
            for request in pair.initial_requests:
                tasks.add(asyncio.create_task(consume(k, pair, request)))
        pending = num_pairs
        while pending:
            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            k, output = item
            pair = pairs[k]
            if output is None:
                if pair.close_slot():
                    pending -= 1
            else:
                pair.receive(output)
        return [pair.rollout() for pair in pairs]
    finally:
        for task in tasks:
            task.cancel()
        cleanup = asyncio.gather(*tasks, return_exceptions=True)
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
