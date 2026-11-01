# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    kl_penalty,
    register_adv_est,
)
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Isolate registry mutations from other algorithm tests."""
        self.original_registry = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = self.original_registry
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_grpo_vectorized_matches_original_for_low_variance_rewards():
    token_level_rewards = torch.tensor([[1.0], [1.00001], [2.0], [2.00001]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,base",
    [
        ("k1+", "k1"),
        ("kl+", "kl"),
        ("abs+", "abs"),
        ("k3+", "k3"),
        ("low_var_kl+", "low_var_kl"),
    ],
)
def test_kl_penalty_straight_through_value_matches_base(name, base):
    """The ``+`` suffix is a straight-through trick that swaps in the k2
    gradient while keeping the base estimator's value. Therefore the forward
    value of e.g. ``k3+`` must match the value of plain ``k3``.

    Regression test for the bug where ``kl_penalty(..., "k3+")`` raised
    ``NotImplementedError`` because the wrapper forwarded the ``+`` suffix to
    ``kl_penalty_forward`` without stripping it.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    plus_value = kl_penalty(logprob, ref_logprob, name)
    base_value = kl_penalty(logprob, ref_logprob, base)
    assert torch.allclose(plus_value, base_value)


def test_kl_penalty_k3_plus_uses_k2_gradient():
    """With ``k3+`` the gradient w.r.t. ``logprob`` should equal the gradient
    obtained from the ``k2`` (``0.5 * log_ratio**2``) estimator, since the
    straight-through trick routes the backward pass through ``k2``.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
    (grad_plus,) = torch.autograd.grad(out_plus, logprob)

    logprob_k2 = logprob.detach().clone().requires_grad_(True)
    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
    (grad_k2,) = torch.autograd.grad(out_k2, logprob_k2)

    assert torch.allclose(grad_plus, grad_k2)


def _equal_length_config(mode, dp_size=1, **kwargs):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        use_dynamic_bsz=True,
        policy_loss=PolicyLossConfig(loss_mode=mode),
        global_batch_info={"dp_size": dp_size, "global_batch_size": 123, "batch_num_tokens": 456},
        **kwargs,
    )


def _equal_length_examples():
    # The first three complete trajectories form an N=2 subgroup. The next
    # two form N=1; the final row contributes no optimization terms.
    ids = torch.tensor([[0, 0, -1], [1, 1, 2], [-1, -1, 3], [0, 0, -1], [1, 1, -1], [-1, -1, -1]])
    weights = torch.tensor(
        [[1 / 8, 1 / 8, 0], [1 / 8, 1 / 8, 1 / 4], [0, 0, 1 / 4], [1 / 4, 1 / 4, 0], [1 / 4, 1 / 4, 0], [0, 0, 0]],
        dtype=torch.float64,
    )
    advantages = torch.tensor(
        [[1, 1, 0], [-1, -1, -1], [0, 0, 1], [-1, -1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float64
    )
    logp = torch.tensor(
        [
            [0.05, 0.1, float("nan")],
            [0.3, 0.4, -0.3],
            [float("nan"), float("nan"), 0.25],
            [-0.4, -0.3, float("nan")],
            [0.03, -0.04, float("nan")],
            [float("nan")] * 3,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    return logp, torch.zeros_like(logp), advantages, weights > 0, ids, weights


def _explicit_equal_length_loss(mode, logp, old, advantages, mask, ids, weights, count=2):
    terms = []
    for row in range(len(logp)):
        for segment in ids[row, mask[row]].unique():
            active = mask[row] & (ids[row] == segment)
            current = logp[row, active]
            advantage = advantages[row, active]
            if mode == "eqlen_rloo":
                token_loss = -advantage * current
            else:
                delta = current - old[row, active]
                ratio = delta.mean().exp() if mode == "eqlen_gspo" else delta.exp()
                token_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
            terms.append((token_loss * weights[row, active]).sum())
    return sum(terms) / count


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_value_and_gradient_match_per_segment_paper_objective(mode):
    logp, old, advantages, mask, ids, weights = _equal_length_examples()
    actual, metrics = get_policy_loss_fn(mode)(
        old,
        logp,
        advantages,
        mask,
        config=_equal_length_config(mode),
        segment_ids=ids,
        loss_weights=weights,
        global_num_subgroups=2,
    )
    expected = _explicit_equal_length_loss(mode, logp, old, advantages, mask, ids, weights)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logp, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logp)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and not actual_grad[~mask].any()
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_normalization_survives_dynamic_microbatches_and_dp_average(mode):
    logp, old, advantages, mask, ids, weights = _equal_length_examples()
    loss_fn = get_policy_loss_fn(mode)
    whole = loss_fn(
        old,
        logp,
        advantages,
        mask,
        config=_equal_length_config(mode),
        segment_ids=ids,
        loss_weights=weights,
        global_num_subgroups=2,
    )[0]
    ranks = [[[4, 0], [5]], [[3], [1, 2]]]
    rank_losses = []
    for chunks in ranks:
        rank_losses.append(
            sum(
                loss_fn(
                    old[index],
                    logp[index],
                    advantages[index],
                    mask[index],
                    config=_equal_length_config(mode, dp_size=2),
                    segment_ids=ids[index],
                    loss_weights=weights[index],
                    global_num_subgroups=2,
                )[0]
                for index in chunks
            )
        )
    distributed = sum(rank_losses) / 2
    torch.testing.assert_close(whole, distributed)
    torch.testing.assert_close(
        torch.autograd.grad(whole, logp, retain_graph=True)[0], torch.autograd.grad(distributed, logp)[0]
    )


def test_equal_length_gspo_clips_two_segments_independently_within_one_trajectory():
    logp = torch.tensor([[0.4, 0.4, -0.4, -0.4]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    mask = torch.ones_like(advantages, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1]])
    weights = torch.full_like(logp, 1 / 4)
    actual, _ = get_policy_loss_fn("eqlen_gspo")(
        torch.zeros_like(logp),
        logp,
        advantages,
        mask,
        config=_equal_length_config("eqlen_gspo"),
        segment_ids=ids,
        loss_weights=weights,
        global_num_subgroups=1,
    )
    assert actual.item() == pytest.approx(-0.2)
    assert not torch.autograd.grad(actual, logp)[0].any()


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_empty_active_set_has_finite_zero_loss_and_gradient(mode):
    logp = torch.full((2, 3), float("nan"), requires_grad=True)
    loss, metrics = get_policy_loss_fn(mode)(
        logp.detach(),
        logp,
        logp.detach(),
        torch.zeros_like(logp, dtype=torch.bool),
        config=_equal_length_config(mode),
        segment_ids=torch.full((2, 3), -1),
        loss_weights=torch.zeros_like(logp),
        global_num_subgroups=2,
    )
    assert loss.item() == 0 and not torch.autograd.grad(loss, logp)[0].any()
    assert all(value == 0 for value in metrics.values())


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_real_callback_preserves_conditioning_and_weights_regularizers(mode):
    # Predictions use full context, while only the second and third response
    # tokens enter the objective. Other response values can be nonfinite.
    response_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    active = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    ids = torch.tensor([[-1, 0, 0, -1], [0, 0, -1, -1]])
    weights = active.double() / 4
    advantages = torch.tensor([[0.0, 1.0, 1.0, 0.0], [-1.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    old = torch.where(active, 0.0, float("nan")).double()
    ref = torch.where(active, -0.05, float("nan")).double()
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[11, 12], [0, 13]]),
            "responses": torch.tensor([[21, 22, 23, 24], [25, 26, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]]),
            "response_mask": response_mask,
            "old_log_probs": old,
            "ref_log_prob": ref,
            "advantages": advantages,
            "eqlen_loss_mask": active,
            "eqlen_segment_ids": ids,
            "eqlen_loss_weights": weights,
        },
        meta_info={"dp_size": 1, "global_batch_size": 99, "batch_num_tokens": 99, "global_num_subgroups": 2},
    ).to_tensordict()
    packed_logp = torch.tensor(
        [float("nan"), float("nan"), 0.1, 0.05, float("nan"), float("nan"), -0.1, 0.03, float("nan")],
        dtype=torch.float64,
        requires_grad=True,
    )
    packed_entropy = torch.tensor(
        [float("nan"), float("nan"), 2.0, 3.0, float("nan"), float("nan"), 4.0, 5.0, float("nan")],
        dtype=torch.float64,
        requires_grad=True,
    )
    offsets = torch.tensor([0, 6, 9])
    outputs = {
        "log_probs": torch.nested.nested_tensor_from_jagged(packed_logp, offsets=offsets),
        "entropy": torch.nested.nested_tensor_from_jagged(packed_entropy, offsets=offsets),
    }
    config = _equal_length_config(mode, use_kl_loss=True, kl_loss_type="k2", kl_loss_coef=0.3, entropy_coeff=0.2)
    before_mask = data["response_mask"].clone()
    actual, metrics = ppo_loss(config, outputs, data)
    current = torch.stack((packed_logp[1:5], torch.nn.functional.pad(packed_logp[6:8], (0, 2))))
    entropy = torch.stack((packed_entropy[1:5], torch.nn.functional.pad(packed_entropy[6:8], (0, 2))))
    expected_policy = _explicit_equal_length_loss(mode, current, old, advantages, active, ids, weights)
    expected_entropy = (torch.where(active, entropy, 0) * weights).sum() / 2
    safe_current, safe_ref = torch.where(active, current, 0), torch.where(active, ref, 0)
    expected_kl = (0.5 * (safe_current - safe_ref).square() * weights).sum() / 2
    expected = expected_policy - 0.2 * expected_entropy + 0.3 * expected_kl
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (packed_logp, packed_entropy), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (packed_logp, packed_entropy))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.isfinite(actual_grad).all() and not actual_grad[[0, 1, 4, 5, 8]].any()
    torch.testing.assert_close(data["response_mask"], before_mask)
    assert metrics["actor/entropy_loss"].aggregate() == pytest.approx(expected_entropy.item())
    assert metrics["kl_loss"].aggregate() == pytest.approx(expected_kl.item())


@pytest.mark.parametrize("bad_field", ["weights", "ids", "count", "active_logp", "is_weights"])
def test_equal_length_loss_rejects_inconsistent_or_nonfinite_inputs(bad_field):
    logp, old, advantages, mask, ids, weights = _equal_length_examples()
    kwargs = dict(segment_ids=ids, loss_weights=weights, global_num_subgroups=2)
    if bad_field == "weights":
        weights[0, 0] = float("nan")
    elif bad_field == "ids":
        ids[0, 0] = -1
    elif bad_field == "count":
        kwargs["global_num_subgroups"] = 0
    elif bad_field == "active_logp":
        logp = logp.detach().clone()
        logp[0, 0] = float("inf")
    else:
        kwargs["rollout_is_weights"] = torch.ones_like(logp)
    with pytest.raises(ValueError):
        get_policy_loss_fn("eqlen_grpo")(
            old, logp, advantages, mask, config=_equal_length_config("eqlen_grpo"), **kwargs
        )


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_anchor_advantages_and_reduction_weights_are_stop_gradient(mode):
    logp, old, advantages, mask, ids, weights = _equal_length_examples()
    old.requires_grad_()
    advantages.requires_grad_()
    weights.requires_grad_()
    loss, _ = get_policy_loss_fn(mode)(
        old,
        logp,
        advantages,
        mask,
        config=_equal_length_config(mode),
        segment_ids=ids,
        loss_weights=weights,
        global_num_subgroups=2,
    )
    gradients = torch.autograd.grad(loss, (logp, old, advantages, weights), allow_unused=True)
    assert gradients[0].abs().sum() > 0 and all(gradient is None for gradient in gradients[1:])


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_tied_pairs_remain_in_segment_and_subgroup_denominators(mode):
    logp, old, advantages, mask, ids, weights = _equal_length_examples()
    logp = torch.zeros_like(logp, requires_grad=True)
    # The second pair of subgroup A and the whole subgroup B are tied. Keep
    # the existing N=2 weights of A's first pair and global subgroup count 2.
    weights[1, 2] = 0
    weights[2:].zero_()
    mask = weights > 0
    ids[~mask] = -1
    loss, _ = get_policy_loss_fn(mode)(
        old,
        logp,
        advantages,
        mask,
        config=_equal_length_config(mode),
        segment_ids=ids,
        loss_weights=weights,
        global_num_subgroups=2,
    )
    gradient = torch.autograd.grad(loss, logp)[0]
    assert loss.item() == 0
    assert gradient.abs().sum().item() == pytest.approx(0.25)
    assert not gradient[~mask].any()


if __name__ == "__main__":
    unittest.main()
