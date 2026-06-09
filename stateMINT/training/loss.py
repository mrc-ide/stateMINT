import jax.numpy as jnp


def _weighted_mean(loss, w=None):
    """Compute weighted mean of loss."""
    if w is None:
        return loss.mean()
    return (loss * w).sum() / jnp.clip(w.sum(), 1.0)


def weighted_mse(pred: jnp.ndarray, target: jnp.ndarray, w: jnp.ndarray | None = None) -> jnp.ndarray:
    """Mean squared error, optionally weighted."""
    loss = (pred - target) ** 2
    return _weighted_mean(loss, w)
