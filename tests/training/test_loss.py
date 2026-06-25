import jax.numpy as jnp
import pytest

from stateMINT.training.loss import weighted_mse, _weighted_mean


def test_unweighted_mse_is_plain_mean():
    pred = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.0, 4.0, 3.0])
    assert float(weighted_mse(pred, target)) == pytest.approx(4.0 / 3.0)


def test_weighted_mse_respects_weights():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([2.0, 4.0])
    w = jnp.array([1.0, 0.0])  # only first sample counts
    assert float(weighted_mse(pred, target, w)) == 4.0


def test_weighted_mean_clips_tiny_weight_sum():
    # weight sum < 1 must be clipped to 1.0 to avoid blow-up.
    loss = jnp.array([2.0])
    w = jnp.array([0.1])
    assert float(_weighted_mean(loss, w)) == pytest.approx(0.2)
