# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-length rollout data and state shared by the engine and frontends."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from vllm.sampling_params import RequestOutputKind, SamplingParams

Aggregation = Literal["max", "mean"]
AdvantageEstimator = Literal["grpo", "rloo"]
NATIVE_PAIR_KEY = "_vllm_equal_length_pair"


@dataclass
class SegmentPair:
    """An equal-length pair of segments generated from the same context.

    ``prefix_len`` response tokens (the ``minus`` segments of earlier pairs) sit
    between the prompt and both segments. ``plus`` is the segment whose branch
    terminated; ``minus`` is the equally long part of the branch that went on,
    and carries a finish reason only in the last pair, where both terminate.
    """

    prefix_len: int
    plus_token_ids: list[int]
    minus_token_ids: list[int]
    plus_finish_reason: str | None
    minus_finish_reason: str | None = None
    plus_logprobs: list[float] | None = None
    minus_logprobs: list[float] | None = None

    def __len__(self) -> int:
        return len(self.plus_token_ids)


@dataclass
class Trajectory:
    """A complete response: every earlier ``minus`` segment plus a final one."""

    response_token_ids: list[int]
    finish_reason: str | None
    logprobs: list[float] | None = None


@dataclass
class TrainingSample:
    """One trajectory with per-token loss mask and advantages.

    Only segments that first appear in this trajectory are unmasked; inherited
    prefixes and segments of pairs with equal rewards are masked out.
    ``segment_ids`` identify active segments within one rollout; masked tokens
    use ``-1``. Sum token losses multiplied by ``loss_weights`` across all samples
    of a rollout, then average these sums across subgroups.
    """

    prompt_token_ids: list[int]
    response_token_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    logprobs: list[float] | None = None
    segment_ids: list[int] = field(default_factory=list)
    loss_weights: list[float] = field(default_factory=list)


@dataclass
class PairedRollout:
    """All segment pairs produced by one pair of branches for one prompt."""

    prompt_token_ids: list[int]
    pairs: list[SegmentPair] = field(default_factory=list)

    def trajectories(self) -> list[Trajectory]:
        """Return the complete responses, in the order they terminated.

        Trajectory ``i < len(pairs)`` ends with ``pairs[i].plus``; the last one
        consists of all ``minus`` segments.
        """
        trajectories = []
        prefix: list[int] = []
        prefix_logprobs: list[float] | None = []
        for pair in self.pairs:
            trajectories.append(
                Trajectory(
                    prefix + pair.plus_token_ids,
                    pair.plus_finish_reason,
                    _concat_logprobs(prefix_logprobs, pair.plus_logprobs),
                )
            )
            prefix = prefix + pair.minus_token_ids
            prefix_logprobs = _concat_logprobs(prefix_logprobs, pair.minus_logprobs)
        if self.pairs:
            trajectories.append(
                Trajectory(prefix, self.pairs[-1].minus_finish_reason, prefix_logprobs)
            )
        return trajectories

    def segment_rewards(
        self,
        trajectory_rewards: Sequence[float],
        aggregation: Aggregation = "max",
    ) -> list[tuple[float, float]]:
        """Return ``(plus_reward, minus_reward)`` for every pair.

        ``pairs[i].plus`` takes the reward of trajectory ``i``; ``pairs[i].minus``
        aggregates the rewards of every trajectory extending it, i.e. trajectories
        ``i + 1`` onwards.
        """
        if len(trajectory_rewards) != len(self.pairs) + 1:
            raise ValueError(
                f"Expected {len(self.pairs) + 1} trajectory rewards, "
                f"got {len(trajectory_rewards)}."
            )
        rewards = []
        for i in range(len(self.pairs)):
            extensions = trajectory_rewards[i + 1 :]
            if aggregation == "max":
                minus_reward = max(extensions)
            elif aggregation == "mean":
                minus_reward = sum(extensions) / len(extensions)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation!r}")
            rewards.append((float(trajectory_rewards[i]), float(minus_reward)))
        return rewards

    def training_samples(
        self,
        trajectory_rewards: Sequence[float],
        aggregation: Aggregation = "max",
        advantage: AdvantageEstimator = "grpo",
    ) -> list[TrainingSample]:
        """Expand the rollout into per-trajectory samples for the policy loss.

        Each pair is its own comparison group of size two. ``"grpo"`` normalizes
        by the pair's standard deviation (``+1``/``-1``); ``"rloo"`` uses the
        leave-one-out difference ``r - r_other``. An active token in a segment
        of length ``L`` receives loss weight ``1 / (2 * len(pairs) * L)``;
        skipped pairs remain in the subgroup denominator. Segment ``i`` has
        ID ``2 * i`` for plus and ``2 * i + 1`` for minus.
        """
        pair_advantages: list[tuple[float, float] | None] = []
        for plus_reward, minus_reward in self.segment_rewards(
            trajectory_rewards, aggregation
        ):
            diff = plus_reward - minus_reward
            if diff == 0:
                pair_advantages.append(None)
            elif advantage == "grpo":
                pair_advantages.append((1.0, -1.0) if diff > 0 else (-1.0, 1.0))
            elif advantage == "rloo":
                pair_advantages.append((diff, -diff))
            else:
                raise ValueError(f"Unknown advantage estimator: {advantage!r}")

        samples = []
        for i, trajectory in enumerate(self.trajectories()):
            loss_mask: list[bool] = []
            advantages: list[float] = []
            segment_ids: list[int] = []
            loss_weights: list[float] = []
            # Trajectory i inherits minus segments 0..i-2 as masked context; the
            # minus segment of pair i-1 and the plus segment of pair i are new.
            for j in range(i):
                adv = pair_advantages[j] if j == i - 1 else None
                length = len(self.pairs[j])
                loss_mask += [adv is not None] * length
                advantages += [0.0 if adv is None else adv[1]] * length
                segment_ids += [-1 if adv is None else 2 * j + 1] * length
                weight = (
                    1.0 / (2 * len(self.pairs) * length)
                    if adv is not None and length
                    else 0.0
                )
                loss_weights += [weight] * length
            if i < len(self.pairs):
                adv = pair_advantages[i]
                length = len(self.pairs[i])
                loss_mask += [adv is not None] * length
                advantages += [0.0 if adv is None else adv[0]] * length
                segment_ids += [-1 if adv is None else 2 * i] * length
                weight = (
                    1.0 / (2 * len(self.pairs) * length)
                    if adv is not None and length
                    else 0.0
                )
                loss_weights += [weight] * length
            samples.append(
                TrainingSample(
                    prompt_token_ids=self.prompt_token_ids,
                    response_token_ids=trajectory.response_token_ids,
                    loss_mask=loss_mask,
                    advantages=advantages,
                    logprobs=trajectory.logprobs,
                    segment_ids=segment_ids,
                    loss_weights=loss_weights,
                )
            )
        return samples


def _concat_logprobs(
    a: list[float] | None, b: list[float] | None
) -> list[float] | None:
    return None if a is None or b is None else a + b


@dataclass
class BranchRequest:
    """A request the driver must submit to start a new branch."""

    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    branch_index: int


@dataclass
class _Branch:
    request_id: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None
    consumed: int = 0
    finished: bool = False
    finish_reason: str | None = None

    @property
    def pending(self) -> int:
        """Tokens generated since the current pair's context."""
        return len(self.token_ids) - self.consumed


class EqualLengthPairScheduler:
    """Pure state machine for one pair of branches; drivers feed it outputs."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        request_id_prefix: str,
        track_logprobs: bool = False,
    ) -> None:
        self.rollout = PairedRollout(prompt_token_ids)
        self.max_tokens = max_tokens
        self.request_id_prefix = request_id_prefix
        self.track_logprobs = track_logprobs
        self.done = False
        self._context: list[int] = []
        self._branches: list[_Branch] = []
        self._num_branches = 0

    def start(self) -> list[BranchRequest]:
        return [self._new_branch(), self._new_branch()]

    def on_output(
        self,
        request_id: str,
        token_ids: Sequence[int],
        logprobs: Sequence[float] | None,
        finished: bool,
        finish_reason: str | None,
    ) -> list[BranchRequest]:
        """Record new tokens of a branch; return the branches to start next."""
        branch = next(b for b in self._branches if b.request_id == request_id)
        branch.token_ids.extend(token_ids)
        if branch.logprobs is not None:
            if logprobs is None:
                raise ValueError(f"Missing logprobs for request {request_id}.")
            branch.logprobs.extend(logprobs)
        if finished:
            branch.finished = True
            branch.finish_reason = finish_reason

        new_requests = []
        while not self.done and (new := self._try_harvest()) is not None:
            new_requests.append(new)
        return new_requests

    def _try_harvest(self) -> BranchRequest | None:
        a, b = self._branches
        if a.finished and (b.finished or b.pending >= a.pending):
            plus, minus = (b, a) if b.finished and b.pending < a.pending else (a, b)
        elif b.finished and a.pending >= b.pending:
            plus, minus = b, a
        else:
            return None

        length = plus.pending
        start, end = minus.consumed, minus.consumed + length
        both_done = minus.finished and minus.pending == length
        self.rollout.pairs.append(
            SegmentPair(
                prefix_len=len(self._context),
                plus_token_ids=plus.token_ids[plus.consumed :],
                minus_token_ids=minus.token_ids[start:end],
                plus_finish_reason=plus.finish_reason,
                minus_finish_reason=minus.finish_reason if both_done else None,
                plus_logprobs=(
                    None if plus.logprobs is None else plus.logprobs[plus.consumed :]
                ),
                minus_logprobs=(
                    None if minus.logprobs is None else minus.logprobs[start:end]
                ),
            )
        )
        if both_done:
            self.done = True
            return None

        self._context += minus.token_ids[start:end]
        minus.consumed = end
        self._branches.remove(plus)
        new = self._new_branch()
        # A branch that already finished beyond the boundary stays in place and
        # is harvested against the new branch once that one catches up.
        return new

    def _new_branch(self) -> BranchRequest:
        index = self._num_branches
        self._num_branches += 1
        branch = _Branch(
            request_id=f"{self.request_id_prefix}-{index}",
            logprobs=[] if self.track_logprobs else None,
        )
        self._branches.append(branch)
        return BranchRequest(
            request_id=branch.request_id,
            prompt_token_ids=self.rollout.prompt_token_ids + self._context,
            max_tokens=self.max_tokens - len(self._context),
            branch_index=index,
        )


def validate_equal_length_sampling_params(
    sampling_params: SamplingParams, num_pairs: int
) -> int:
    if num_pairs < 1:
        raise ValueError(f"num_pairs must be at least 1, got {num_pairs}.")
    if sampling_params.n != 1:
        raise ValueError(
            "Equal-length pairs are sampled one branch per request; set n=1 and "
            "use num_pairs to choose the group size (2 * num_pairs)."
        )
    if sampling_params.max_tokens is None:
        raise ValueError("Equal-length pairs require an explicit max_tokens.")
    for name, enabled in (
        ("presence_penalty", sampling_params.presence_penalty != 0),
        ("frequency_penalty", sampling_params.frequency_penalty != 0),
        ("stop", bool(sampling_params.stop)),
        ("structured_outputs", sampling_params.structured_outputs is not None),
        ("bad_words", bool(sampling_params.bad_words)),
        ("_bad_words_token_ids", bool(sampling_params._bad_words_token_ids)),
        ("thinking_token_budget", sampling_params.thinking_token_budget is not None),
        (
            "repetition_detection",
            sampling_params.repetition_detection is not None
            and sampling_params.repetition_detection.max_pattern_size > 0,
        ),
        ("trace_decode_token_ids", bool(sampling_params.trace_decode_token_ids)),
    ):
        if enabled:
            raise ValueError(
                f"Equal-length pairs do not support {name}: restarting branches "
                "cannot preserve its generation history."
            )
    return sampling_params.max_tokens


def branch_sampling_params(
    base: SamplingParams, request: BranchRequest, seed_offset: int = 0
) -> SamplingParams:
    assert base.max_tokens is not None
    params = base.clone()
    params.n = 1
    params.output_kind = RequestOutputKind.DELTA
    params.max_tokens = request.max_tokens
    inherited_tokens = base.max_tokens - request.max_tokens
    params.min_tokens = max(0, base.min_tokens - inherited_tokens)
    if base.seed is not None:
        # Branches sharing a context must not share a seed, or they would
        # decode identical tokens.
        params.seed = base.seed + seed_offset + request.branch_index
    return params


@dataclass(frozen=True)
class NativePairConfig:
    group_id: str
    slot: int
    branch_index: int


def get_native_pair_config(params: SamplingParams | None) -> NativePairConfig | None:
    if (
        params is None
        or not params.extra_args
        or NATIVE_PAIR_KEY not in params.extra_args
    ):
        return None
    config = params.extra_args[NATIVE_PAIR_KEY]
    if not isinstance(config, dict) or set(config) != {
        "group_id",
        "slot",
        "branch_index",
    }:
        raise ValueError("Invalid native equal-length pair descriptor.")
    group_id, slot, branch_index = (
        config[k] for k in ("group_id", "slot", "branch_index")
    )
    if (
        not isinstance(group_id, str)
        or not group_id
        or type(slot) is not int
        or slot not in (0, 1)
        or type(branch_index) is not int
        or branch_index < 0
    ):
        raise ValueError("Invalid native equal-length pair descriptor.")
    return NativePairConfig(group_id, slot, branch_index)


def with_native_pair_config(
    params: SamplingParams, config: NativePairConfig
) -> SamplingParams:
    result = params.clone()
    result.extra_args = {
        **(result.extra_args or {}),
        NATIVE_PAIR_KEY: {
            "group_id": config.group_id,
            "slot": config.slot,
            "branch_index": config.branch_index,
        },
    }
    get_native_pair_config(result)
    return result
