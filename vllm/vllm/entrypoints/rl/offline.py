# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.entrypoints.common.offline import OfflineInferenceMixin
from vllm.entrypoints.rl.equal_length_rollout import (
    PairedRollout,
    PromptTokens,
    generate_equal_length_pairs,
)
from vllm.sampling_params import SamplingParams


class RLOfflineMixin(OfflineInferenceMixin):
    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info

        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def start_weight_update(self) -> None:
        """Start a new weight update."""
        self.llm_engine.collective_rpc("start_weight_update")

    def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        self.llm_engine.collective_rpc("start_draft_weight_update")

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info

        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        self.llm_engine.collective_rpc("finish_weight_update")
        if weight_version is not None:
            self.llm_engine.set_weight_version(weight_version)

    def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        self.llm_engine.set_weight_version(new_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.llm_engine.get_weight_version()

    def generate_equal_length_pairs(
        self,
        prompts: Sequence[PromptTokens],
        sampling_params: SamplingParams,
        num_pairs: int = 1,
    ) -> list[list[PairedRollout]]:
        """Generate equal-length segment pairs for group-relative RL.

        Each of the ``num_pairs`` pairs per prompt relays two branches: whenever
        the shorter one terminates, the equally long prefix of the other is
        inherited as context and a new branch starts from it.

        Args:
            prompts: Tokenized prompts.
            sampling_params: Sampling parameters for every branch; ``n`` must be
                1 and ``max_tokens`` caps the full response length.
            num_pairs: Pairs of branches per prompt (group size / 2).

        Returns:
            ``result[p][k]`` is the ``k``-th pair rollout of ``prompts[p]``.

        """
        return generate_equal_length_pairs(
            self.llm_engine, prompts, sampling_params, num_pairs
        )
