import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

from stateMINT.common.utils import (
    transform_targets_np,
    inverse_transform_np,
    inverse_transform_jax,
    forward,
)


@pytest.mark.parametrize("predictor", ["prevalence", "cases"])
def test_transform_inverse_roundtrip(predictor):
    y = np.array([0.1, 0.4, 0.8], dtype=np.float64)
    if predictor == "cases":
        y = np.array([0.5, 3.0, 42.0])
    out = inverse_transform_np(transform_targets_np(y, predictor), predictor)
    np.testing.assert_allclose(out, y, rtol=1e-4)


def test_prevalence_transform_clips_extremes():
    # 0 and 1 would blow up the logit; transform must clip them to finite values.
    out = transform_targets_np(np.array([0.0, 1.0]), "prevalence", eps=1e-5)
    assert np.all(np.isfinite(out))


def test_cases_transform_clamps_negatives_to_zero():
    # log1p(max(y, 0)): negatives become log1p(0) == 0.
    out = transform_targets_np(np.array([-5.0, 0.0]), "cases")
    np.testing.assert_allclose(out, 0.0)


@pytest.mark.parametrize("predictor", ["prevalence", "cases"])
def test_jax_inverse_matches_numpy(predictor):
    y = np.array([-1.0, 0.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(inverse_transform_jax(jnp.asarray(y), predictor)),
        inverse_transform_np(y, predictor),
        rtol=1e-5,
    )


def test_forward_squeezes_trailing_dim():
    model = nnx.Linear(3, 1, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 5, 3))
    out = forward(model, x)
    assert out.shape == (2, 5)
