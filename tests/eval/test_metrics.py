import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from stateMINT.eval import metrics as M
from stateMINT.data.dataset import make_loader


def test_perfect_prediction_metrics():
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert float(M.mse(y, y)) == 0.0
    assert float(M.rmse(y, y)) == 0.0
    assert float(M.mae(y, y)) == 0.0
    assert float(M.r2(y, y)) == pytest.approx(1.0)
    assert float(M.bias(y, y)) == 0.0
    assert float(M.smape(y, y)) == pytest.approx(0.0, abs=1e-3)


def test_basic_metric_values():
    preds = jnp.array([2.0, 4.0])
    targets = jnp.array([0.0, 0.0])
    assert float(M.mse(preds, targets)) == 10.0  # (4 + 16) / 2
    assert float(M.rmse(preds, targets)) == pytest.approx(np.sqrt(10.0))
    assert float(M.mae(preds, targets)) == 3.0
    assert float(M.bias(preds, targets)) == 3.0


def test_metrics_dict_log_likelihood_branch():
    preds = jnp.array([0.0, 0.5])
    targets = jnp.array([0.0, 0.5])
    prev = M._metrics_from_preds_targets(preds, targets, "prevalence")
    assert np.isfinite(float(prev["log_likelihood"]))
    cases = M._metrics_from_preds_targets(preds, targets, "cases")
    assert np.isnan(float(cases["log_likelihood"]))


def test_compute_metrics_over_loader(tiny_model_kwargs):
    from stateMINT.model import Mamba2Regressor

    model = Mamba2Regressor(input_dim=3, rngs=nnx.Rngs(0), **tiny_model_kwargs)
    model.eval()
    data = [
        {"x": np.zeros((4, 3), dtype=np.float32), "y": np.zeros(4, dtype=np.float32)}
        for _ in range(4)
    ]
    loader = make_loader(data, batch_size=2)
    out = M.compute_metrics(model, loader, "prevalence")
    assert set(out) == {"mse", "rmse", "mae", "r2", "smape", "bias", "log_likelihood"}
    assert all(isinstance(v, float) for v in out.values())
