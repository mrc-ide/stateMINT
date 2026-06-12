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
    """
    Mean squared error.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        Mean squared error.
    """
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def rmse(preds: Array, targets: Array) -> Array:
    """
    Root mean squared error.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        Root mean squared error.
    """
    return jnp.sqrt(mse(preds, targets))


@jax.jit
def mae(preds: Array, targets: Array) -> Array:
    """
    Mean absolute error.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        Mean absolute error.
    """
    return jnp.mean(jnp.abs(preds - targets))


@jax.jit
def r2(preds: Array, targets: Array) -> Array:
    """
    Coefficient of determination.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        R2 score.
    """
    return 1.0 - jnp.sum((targets - preds) ** 2) / jnp.sum((targets - jnp.mean(targets)) ** 2)


@jax.jit
def smape(preds: Array, targets: Array) -> Array:
    """
    Symmetric mean absolute percentage error.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        SMAPE percentage.
    """
    return 100 * jnp.mean(2 * jnp.abs(preds - targets) / (jnp.abs(preds) + jnp.abs(targets) + EPS))


@jax.jit
def bias(preds: Array, targets: Array) -> Array:
    """
    Mean prediction bias.

    Args:
        preds: Predictions.
        targets: Targets.

    Returns:
        Mean prediction bias.
    """
    return jnp.mean(preds - targets)


@partial(jax.jit, static_argnames=["predictor"])
def _metrics_from_preds_targets(preds: Array, targets: Array, predictor: Predictor) -> dict[str, Array | float]:
    """
    Compute evaluation metrics on the given predictions and targets.

    Predictions and targets are first mapped back to the original target scale.

    Args:
        preds: Transformed predictions.
        targets: Transformed targets.
        predictor: Target type.

    Returns:
        Metric names mapped to values.
    """
    preds = inverse_transform_jax(preds, predictor)
    targets = inverse_transform_jax(targets, predictor)

    return {
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "r2": r2(preds, targets),
        "smape": smape(preds, targets),
        "bias": bias(preds, targets),
    }


def compute_metrics(
    model: nnx.Module,
    data_loader: grain.DataLoader,
    predictor: Predictor,
) -> dict[str, float]:
    """
    Compute evaluation metrics on the given data loader.

    All batch predictions and targets are concatenated before metrics are computed.

    Args:
        model: Model to evaluate.
        data_loader: Evaluation data loader.
        predictor: Target type.

    Returns:
        Metric names mapped to floats.
    """
    preds, targets, _ = get_preds_targets(model, data_loader)

    metrics = _metrics_from_preds_targets(preds, targets, predictor)
    return {k: float(v) for k, v in metrics.items()}


def get_preds_targets(model: nnx.Module, data_loader: grain.DataLoader) -> tuple[Array, Array, Array]:
    """
    Get concatenated predictions and targets from the model on the given data loader.

    Args:
        model: Model to get predictions from.
        data_loader: Data loader to get predictions on.
    Returns:
        Tuple of ``(predictions, targets, ps)``. Predictions and targets have
        shape ``(N, T)`` and ``ps`` has shape ``(N, 2)``, where ``N`` is the
        total number of samples in the data loader and ``T`` is the number of
        timesteps.
    """
    all_preds, all_targets, all_ps = [], [], []
    for batch in data_loader:
        all_preds.append(forward(model, batch["x"]))
        all_targets.append(batch["y"])
        all_ps.append(batch["ps"])

    preds = jnp.concatenate(all_preds)
    targets = jnp.concatenate(all_targets)
    ps = jnp.concatenate(all_ps)

    return preds, targets, ps
