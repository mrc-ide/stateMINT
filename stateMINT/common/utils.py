import numpy as np
import jax
import jax.numpy as jnp
from .dataclasses import Predictor
from jaxtyping import Array
from flax import nnx


def transform_targets_np(y: np.ndarray, predictor: Predictor, eps: float = 1e-5) -> np.ndarray:
    """Apply train-time transform to targets."""
    if predictor == "prevalence":
        y = np.clip(y, eps, 1.0 - eps)
        return np.log(y / (1.0 - y))  # logit transform
    else:
        return np.log1p(np.maximum(y, 0.0))  # log1p transform for counts/rates


def inverse_transform_np(y: np.ndarray, predictor: Predictor) -> np.ndarray:
    """Invert transform for metrics/plots."""
    if predictor == "prevalence":
        return 1.0 / (1.0 + np.exp(-y))  # sigmoid
    else:
        return np.expm1(y)


def inverse_transform_jax(y: jax.Array, predictor: Predictor) -> jax.Array:
    if predictor == "prevalence":
        return jax.nn.sigmoid(y)
    else:
        return jnp.expm1(y)


@nnx.jit
def forward(model: nnx.Module, x: Array) -> Array:
    """Forward pass through the model."""
    return model(x).squeeze(-1)  # (B, T, 1) -> (B, T)
