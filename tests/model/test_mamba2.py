import jax.numpy as jnp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from stateMINT.model import Mamba2Regressor


def test_forward_output_shape(tiny_model_kwargs):
    model = Mamba2Regressor(input_dim=5, rngs=nnx.Rngs(0), **tiny_model_kwargs)
    out = model(jnp.ones((2, 7, 5)))
    assert out.shape == (2, 7, tiny_model_kwargs["output_dim"])


def test_invalid_head_dim_raises():
    # d_model * expand must be divisible by head_dim.
    with pytest.raises(AssertionError):
        Mamba2Regressor(input_dim=3, d_model=16, expand=2, head_dim=7, rngs=nnx.Rngs(0))


def test_from_cfg_builds_model(tiny_model_kwargs):
    cfg = OmegaConf.create({**tiny_model_kwargs, "seed": 0})
    model = Mamba2Regressor.from_cfg(cfg, input_size=4)
    out = model(jnp.ones((1, 3, 4)))
    assert out.shape == (1, 3, 1)
