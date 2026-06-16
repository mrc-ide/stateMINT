import time

import jax.numpy as jnp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from stateMINT.common.utils import forward
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


@pytest.mark.skip(reason="This test is for profiling and not for CI.")
def test_train_config_forward_pass_time():
    input_size = 16
    time_series_length = 157
    x = jnp.ones((1, time_series_length, input_size), dtype=jnp.float32)

    model = Mamba2Regressor(input_dim=input_size, d_model=256, d_state=64, rngs=nnx.Rngs(0))
    model.eval()

    # Warm up compilation before measuring the JIT-compiled forward pass.
    for _ in range(5):
        forward(model, x).block_until_ready()

    # Run 5 times and take average
    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = forward(model, x).block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_elapsed = sum(times) / len(times)
    print(f"train_config forward pass time: {avg_elapsed:.6f} seconds (avg of 5 runs)")
    assert out.shape == (1, time_series_length)  # type: ignore
    assert elapsed >= 0.0  # type: ignore
