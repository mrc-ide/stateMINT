import jax.numpy as jnp
import numpy as np
import pytest

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
    assert float(M.mse(preds, targets)) == 10.0
    assert float(M.rmse(preds, targets)) == pytest.approx(np.sqrt(10.0))
    assert float(M.mae(preds, targets)) == 3.0
    assert float(M.bias(preds, targets)) == 3.0


def test_compute_metrics_over_loader(model_factory, loader_records_factory):
    model = model_factory()
    model.eval()
    loader = make_loader(loader_records_factory(4, include_ps=True), batch_size=2)
    out = M.compute_metrics(model, loader, "prevalence")
    assert set(out) == {"mse", "rmse", "mae", "r2", "smape", "bias"}
    assert all(isinstance(v, float) for v in out.values())
