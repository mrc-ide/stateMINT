from functools import partial
import jax.numpy as jnp
from jaxtyping import Array
import jax

from ..common.utils import inverse_transform_jax, forward
from ..common.dataclasses import Predictor
import grain.python as grain
from flax import nnx


@partial(jax.jit, static_argnames=["predictor"])
def _metrics_from_preds_targets(preds: Array, targets: Array, predictor: Predictor) -> dict[str, Array | float]:
    """Compute evaluation metrics on the given predictions and targets."""
    preds = inverse_transform_jax(preds, predictor)
    targets = inverse_transform_jax(targets, predictor)

    eps = 1e-7
    mse = jnp.mean((preds - targets) ** 2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(preds - targets))
    r2 = 1.0 - jnp.sum((targets - preds) ** 2) / jnp.sum((targets - jnp.mean(targets)) ** 2)
    smape = 100 * jnp.mean(2 * jnp.abs(preds - targets) / (jnp.abs(preds) + jnp.abs(targets) + eps))
    bias = jnp.mean(preds - targets)

    if predictor == "prevalence":
        sp = jnp.clip(preds, eps, 1 - eps)
        st = jnp.clip(targets, eps, 1 - eps)
        log_likelihood = jnp.mean(jnp.log(sp) * st + jnp.log(1 - sp) * (1 - st))
    else:
        log_likelihood = jnp.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "bias": bias,
        "log_likelihood": log_likelihood,
    }


def compute_metrics(
    model: nnx.Module,
    data_loader: grain.DataLoader,
    predictor: Predictor,
) -> dict[str, float]:
    """Compute evaluation metrics on the given data loader."""
    all_preds, all_targets = [], []
    for batch in data_loader:
        all_preds.append(forward(model, batch["x"]))
        all_targets.append(batch["y"])

    preds = jnp.concatenate(all_preds)
    targets = jnp.concatenate(all_targets)

    metrics = _metrics_from_preds_targets(preds, targets, predictor)
    return {k: float(v) for k, v in metrics.items()}
