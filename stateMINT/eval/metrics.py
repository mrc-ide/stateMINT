from functools import partial
import jax.numpy as jnp
from jaxtyping import Array
import jax

from ..common.utils import inverse_transform_jax, forward
from ..common.dataclasses import Predictor
import grain.python as grain
from flax import nnx


EPS = 1e-7


@jax.jit
def mse(preds: Array, targets: Array) -> Array:
    """Mean squared error."""
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def rmse(preds: Array, targets: Array) -> Array:
    """Root mean squared error."""
    return jnp.sqrt(mse(preds, targets))


@jax.jit
def mae(preds: Array, targets: Array) -> Array:
    """Mean absolute error."""
    return jnp.mean(jnp.abs(preds - targets))


@jax.jit
def r2(preds: Array, targets: Array) -> Array:
    """Coefficient of determination."""
    return 1.0 - jnp.sum((targets - preds) ** 2) / jnp.sum((targets - jnp.mean(targets)) ** 2)


@jax.jit
def smape(preds: Array, targets: Array) -> Array:
    """Symmetric mean absolute percentage error."""
    return 100 * jnp.mean(2 * jnp.abs(preds - targets) / (jnp.abs(preds) + jnp.abs(targets) + EPS))


@jax.jit
def bias(preds: Array, targets: Array) -> Array:
    """Mean prediction bias."""
    return jnp.mean(preds - targets)


@jax.jit
def log_likelihood(preds: Array, targets: Array) -> Array:
    """Bernoulli log likelihood for prevalence predictions."""
    sp = jnp.clip(preds, EPS, 1 - EPS)
    st = jnp.clip(targets, EPS, 1 - EPS)
    return jnp.mean(jnp.log(sp) * st + jnp.log(1 - sp) * (1 - st))


@partial(jax.jit, static_argnames=["predictor"])
def _metrics_from_preds_targets(preds: Array, targets: Array, predictor: Predictor) -> dict[str, Array | float]:
    """Compute evaluation metrics on the given predictions and targets."""
    preds = inverse_transform_jax(preds, predictor)
    targets = inverse_transform_jax(targets, predictor)

    if predictor == "prevalence":
        log_likelihood_value = log_likelihood(preds, targets)
    else:
        log_likelihood_value = jnp.nan

    return {
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "r2": r2(preds, targets),
        "smape": smape(preds, targets),
        "bias": bias(preds, targets),
        "log_likelihood": log_likelihood_value,
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
