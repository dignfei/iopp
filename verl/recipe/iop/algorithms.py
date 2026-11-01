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
"""IOP edit alignment, hard repair eligibility and group advantages."""

import json
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from verl.trainer.ppo.core_algos import register_adv_est


@dataclass(frozen=True)
class EditAlignment:
    """A deterministic shortest unit-cost edit script and its original coordinates."""

    original: tuple[int, ...]
    repaired: tuple[int, ...]
    edits: tuple[tuple[str, int, int], ...]

    @property
    def distance(self):
        return len(self.edits)

    @property
    def normalized_distance(self):
        return self.distance / max(len(self.original), len(self.repaired), 1)

    def masks(self, limit=None):
        """Activate the first limit edits on both sides, without inventing insertion coordinates."""
        count = self._count(limit)
        negative, positive = [False] * len(self.original), [False] * len(self.repaired)
        for tag, source, target in self.edits[:count]:
            if tag != "insert":
                negative[source] = True
            if tag != "delete":
                positive[target] = True
        return negative, positive

    def graft_prefix(self, limit):
        """Apply edit columns up to the chosen operation; a full script reuses the full repair."""
        count = self._count(limit)
        if count == len(self.edits):
            return list(self.repaired)
        if count == 0:
            return list(self.original[: self.edits[0][1]])
        prefix, source_cursor, target_cursor = [], 0, 0
        for tag, source, target in self.edits[:count]:
            assert self.original[source_cursor:source] == self.repaired[target_cursor:target]
            prefix.extend(self.original[source_cursor:source])
            if tag != "delete":
                prefix.append(self.repaired[target])
            source_cursor = source + (tag != "insert")
            target_cursor = target + (tag != "delete")
        return prefix

    def _count(self, limit):
        if limit is None:
            return len(self.edits)
        if type(limit) is not int or limit < 0:
            raise ValueError("Edit limit must be a nonnegative integer or None")
        return min(limit, len(self.edits))


def align_tokens(original, repaired):
    """Compute exact Levenshtein alignment using the existing native distance dependency."""
    for tokens in (original, repaired):
        if any(isinstance(t, bool) or not isinstance(t, Integral) or t < 0 for t in tokens):
            raise ValueError("IOP alignment requires nonnegative integer token IDs")
    original, repaired = tuple(original), tuple(repaired)
    edits = tuple((op.tag, op.src_pos, op.dest_pos) for op in Levenshtein.editops(original, repaired))
    return EditAlignment(original, repaired, edits)


def repair_messages(question, failed, reference):
    """Put all repair conditions in user context, leaving only the repair as an SFT target."""
    if not isinstance(question, list) or not question:
        raise ValueError("IOP question must be a nonempty chat-message list")
    if any(not isinstance(message.get("content"), str) for message in question):
        raise ValueError("IOP supports text-only question messages")
    return [
        {
            "role": "user",
            "content": (
                "Repair the failed reasoning with the fewest necessary edits, using the verified correct reference. "
                "Preserve correct reasoning and output only the complete repaired response.\n\n"
                f"Original conversation:\n{json.dumps(question, ensure_ascii=False)}\n\n"
                f"Failed reasoning:\n{failed}\n\nVerified correct reference:\n{reference}"
            ),
        }
    ]


def _binary(values):
    values = [float(value) for value in values]
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("IOP task rewards and audit decisions must be binary {0,1}")
    return values


def repair_scores(original, candidates, task_rewards, audits, edit_weight=0.3):
    """Score the full repair group; hard eligibility is a separate operation."""
    if not isfinite(edit_weight) or edit_weight < 0:
        raise ValueError("IOP edit_weight must be finite and nonnegative")
    rewards, audits = _binary(task_rewards), _binary(audits)
    if not len(candidates) == len(rewards) == len(audits):
        raise ValueError("IOP repair group lengths differ")
    return [
        h * (r - edit_weight * align_tokens(original, tokens).normalized_distance)
        for tokens, r, h in zip(candidates, rewards, audits, strict=True)
    ]


def select_repair(scores, task_rewards, audits):
    """Select the best h=1 AND task-correct candidate, even if an ineligible score is larger."""
    rewards, audits = _binary(task_rewards), _binary(audits)
    if not len(scores) == len(rewards) == len(audits) or any(not isfinite(s) for s in scores):
        raise ValueError("Invalid IOP candidate scores")
    eligible = [i for i, (r, h) in enumerate(zip(rewards, audits, strict=True)) if r == h == 1]
    return max(eligible, key=lambda i: scores[i]) if eligible else None


@register_adv_est("iop")
def compute_iop_advantage(token_level_rewards, response_mask, index, config=None, **kwargs):
    """Population-standardize policy pairs and complete repair groups separately by uid."""
    del config, kwargs
    scores = token_level_rewards.detach().double().sum(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("IOP rewards must be finite")
    active = response_mask.bool().any(-1)
    if len(index) != len(scores):
        raise ValueError("IOP uid count differs from batch size")
    advantage = torch.zeros_like(scores)
    labels = np.asarray(index)
    for uid in dict.fromkeys(labels.tolist()):
        rows = torch.as_tensor(np.flatnonzero(labels == uid), device=scores.device)
        rows = rows[active[rows]]
        if len(rows) > 1:
            group = scores[rows]
            advantage[rows] = (group - group.mean()) / (group.std(unbiased=False) + 1e-8)
    advantages = advantage.to(token_level_rewards.dtype).unsqueeze(-1) * response_mask
    return advantages, advantages.clone()
