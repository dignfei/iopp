# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

pytestmark = pytest.mark.skip_global_cleanup


def _runner(discard: list[bool]) -> CPUModelRunner:
    count = len(discard)
    generators = {i: torch.Generator().manual_seed(17 + i) for i in range(count)}
    metadata = SamplingMetadata(
        temperature=torch.ones(count),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(count),
        presence_penalties=torch.zeros(count),
        repetition_penalties=torch.ones(count),
        output_token_ids=[[] for _ in discard],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    runner = CPUModelRunner.__new__(CPUModelRunner)
    runner.sampler = Sampler(use_fp64_gumbel=True)
    runner.input_batch = SimpleNamespace(
        num_reqs=count,
        generators=generators,
        sampling_metadata=metadata,
        update_async_output_token_ids=lambda: None,
        req_ids=[str(i) for i in range(count)],
        req_id_to_index={str(i): i for i in range(count)},
        num_tokens_no_spec=np.ones(count, dtype=np.int32),
        token_ids_cpu=np.zeros((count, 8), dtype=np.int32),
        is_token_ids=np.zeros((count, 8), dtype=bool),
        vocab_size=17,
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array(discard, dtype=bool))
    runner.use_async_scheduling = False
    runner.max_model_len = 8
    runner.requests = {
        str(i): SimpleNamespace(output_token_ids=[]) for i in range(count)
    }
    runner.num_prompt_logprobs = {}
    return runner


def _sample(runner):
    return runner._sample(torch.zeros(runner.input_batch.num_reqs, 17), None)


def _bookkeep(runner, output):
    return runner._bookkeeping_sync(
        SimpleNamespace(num_scheduled_tokens={}), output, None, torch.empty(0), 0
    )


def test_partial_prefill_preserves_seeded_decode_stream():
    runner = _runner([True])
    generator = runner.input_batch.generators[0]
    before = generator.get_state()
    sampled = _sample(runner)
    assert torch.equal(generator.get_state(), before)
    assert _bookkeep(runner, sampled)[3] == [[]]
    runner.discard_request_mask.np[0] = False
    actual = _sample(runner)
    reference = _runner([False])
    expected = _sample(reference)
    assert torch.equal(actual.sampled_token_ids, expected.sampled_token_ids)
    assert torch.equal(
        generator.get_state(), reference.input_batch.generators[0].get_state()
    )


def test_mixed_prefill_decode_advances_only_decode_rng():
    runner = _runner([True, False])
    generators = runner.input_batch.generators.copy()
    before = {i: gen.get_state() for i, gen in generators.items()}
    output = _sample(runner)
    assert torch.equal(generators[0].get_state(), before[0])
    assert not torch.equal(generators[1].get_state(), before[1])
    result = _bookkeep(runner, output)
    assert result[3] == [[], [output.sampled_token_ids[1, 0].item()]]
    assert runner.requests["0"].output_token_ids == []
    assert runner.requests["1"].output_token_ids == result[3][1]
    assert all(runner.input_batch.generators[i] is gen for i, gen in generators.items())


@pytest.mark.parametrize("stage", ["sampler", "bookkeeping"])
def test_discarded_generator_restored_after_failure(stage):
    runner = _runner([True, False])
    generators = runner.input_batch.generators.copy()
    before = {i: gen.get_state() for i, gen in generators.items()}
    if stage == "sampler":
        original_forward = runner.sampler.forward

        def fail_after_sampling(*args, **kwargs):
            original_forward(*args, **kwargs)
            raise RuntimeError("injected after actual sampling")

        runner.sampler.forward = fail_after_sampling
        with pytest.raises(RuntimeError, match="injected"):
            _sample(runner)
    else:
        output = _sample(runner)

        def fail_prompt_logprobs(*args):
            raise RuntimeError("injected during bookkeeping")

        runner._get_prompt_logprobs_dict = fail_prompt_logprobs
        with pytest.raises(RuntimeError, match="injected"):
            _bookkeep(runner, output)
    assert torch.equal(generators[0].get_state(), before[0])
    assert not torch.equal(generators[1].get_state(), before[1])
    assert all(runner.input_batch.generators[i] is gen for i, gen in generators.items())
