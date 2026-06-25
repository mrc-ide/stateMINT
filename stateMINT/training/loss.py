import jax.numpy as jnp


def _weighted_mean(loss, w=None):
    """
    Compute weighted mean of loss.

    If weights are absent, this falls back to the plain mean.

    Args:
        loss: Per-example losses.
        w: Optional weights.

    Returns:
        Mean loss.
    """
    if w is None:
        return loss.mean()
    return (loss * w).sum() / jnp.clip(w.sum(), 1.0)


def weighted_mse(pred: jnp.ndarray, target: jnp.ndarray, w: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Mean squared error, optionally weighted.

    Args:
        pred: Predictions.
        target: Targets.
        w: Optional weights.

    Returns:
        Mean squared error.
    """
    loss = (pred - target) ** 2
    return _weighted_mean(loss, w)
