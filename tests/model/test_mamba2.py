import time

import jax.numpy as jnp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from stateMINT.common.utils import forward
from stateMINT.model import Mamba2Regressor


def test_forward_output_shape(model_factory):
    model = model_factory(input_dim=5)
    out = model(jnp.ones((2, 7, 5)))
    assert out.shape == (2, 7, 1)


def test_invalid_head_dim_raises(model_factory):
    # d_model * expand must be divisible by head_dim.
    with pytest.raises(AssertionError):
        model_factory(input_dim=3, d_model=15, expand=2, head_dim=7)


def test_from_cfg_builds_model(tiny_model_kwargs):
    cfg = OmegaConf.create({**tiny_model_kwargs, "seed": 0})
    model = Mamba2Regressor.from_cfg(cfg, input_size=4)
    out = model(jnp.ones((1, 3, 4)))
    assert out.shape == (1, 3, 1)


@pytest.mark.skip(reason="This test is for profiling and not for regular test runs.")
def test_train_config_forward_pass_time(pretrained_input):
    model = Mamba2Regressor(
        input_dim=pretrained_input.shape[-1], d_model=256, d_state=128, n_layers=2, rngs=nnx.Rngs(0)
    )
    model.eval()

    # Warm up compilation before measuring the JIT-compiled forward pass.
    for _ in range(5):
        forward(model, pretrained_input).block_until_ready()

    # Run 5 times and take average
    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = forward(model, pretrained_input).block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_elapsed = sum(times) / len(times)
    print(f"train_config forward pass time: {avg_elapsed:.6f} seconds (avg of 5 runs)")
    assert out.shape == pretrained_input.shape[:2]  # type: ignore
    assert elapsed >= 0.0  # type: ignore


@pytest.mark.local
def test_prevalence_from_pretrained_from_hub(pretrained_input):
    model_artifact = Mamba2Regressor.from_pretrained(
        "dide-ic/stateMINT",
        predictor="prevalence",
    )

    out = forward(model_artifact.model, pretrained_input)

    assert out.shape == pretrained_input.shape[:2]


@pytest.mark.local
def test_cases_from_pretrained_from_hub(pretrained_input):
    model_artifact = Mamba2Regressor.from_pretrained(
        "dide-ic/stateMINT",
        predictor="cases",
    )

    out = forward(model_artifact.model, pretrained_input)

    assert out.shape == pretrained_input.shape[:2]


# ensure the artifacts are present running this test
@pytest.mark.local
def test_from_pretrained_with_local_dir(pretrained_input):
    model_artifact = Mamba2Regressor.from_pretrained(
        "dide-ic/stateMINT",
        predictor="prevalence",
        local_dir="artifacts/prevalence",
    )

    out = forward(model_artifact.model, pretrained_input)

    assert out.shape == pretrained_input.shape[:2]


@pytest.mark.local
def test_predict_prevalence_from_hub(static_covar_dicts):
    model_artifact = Mamba2Regressor.from_pretrained(
        "dide-ic/stateMINT",
        predictor="prevalence",
    )
    preds = model_artifact.predict(static_covar_dicts)
    assert preds.shape == (3, model_artifact.preprocessing_config["n_steps"])


@pytest.mark.local
def test_predict_cases_from_hub(static_covar_dicts):
    model_artifact = Mamba2Regressor.from_pretrained(
        "dide-ic/stateMINT",
        predictor="cases",
    )
    preds = model_artifact.predict(static_covar_dicts)
    assert preds.shape == (3, model_artifact.preprocessing_config["n_steps"])
