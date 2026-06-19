import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from stateMINT.data import AFTER_INTERVENTION_COVARS, BURNIN_DAY, INTERVENTION_DAY, STATIC_COVARS, StandardScaler


INFERENCE_N_STEPS = 157
INPUT_SIZE = (
    len(STATIC_COVARS) + 4
)  # with cyclical time features: 2 time features + static covars + post_intervention, t_since_intervention_yrs


@pytest.fixture
def tiny_model_kwargs():
    return dict(
        d_model=16,
        n_layers=1,
        d_state=8,
        d_conv=2,
        expand=2,
        head_dim=8,
        chunk_size=8,
        output_dim=1,
        dropout=0.0,
    )


@pytest.fixture
def sample_df():
    rows = []
    rng = np.random.default_rng(0)
    n_steps = 20
    # abs timesteps straddle INTERVENTION_DAY (9*365 = 3285)
    abs_ts = np.linspace(INTERVENTION_DAY - 5 * 14, INTERVENTION_DAY + 14 * 14, n_steps)
    for p in range(4):
        static = {c: float(rng.uniform(0, 1)) for c in STATIC_COVARS}
        for t, at in enumerate(abs_ts, start=1):
            rows.append(
                {
                    "parameter_index": p,
                    "simulation_index": 0,
                    "global_index": p,
                    "timesteps": t,
                    "abs_timesteps": float(at),
                    "prevalence": float(rng.uniform(0.05, 0.9)),
                    "cases": float(rng.uniform(0.5, 50.0)),
                    "exposure_pd": float(rng.uniform(100, 1000)),
                    **static,
                }
            )
    return pd.DataFrame(rows)


def make_cfg(tmp_path, predictor="prevalence", **overrides):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "predictor": predictor,
            "min_prevalence": 0.0,
            "min_cases": 0.0,
            "use_existing_split": False,
            "split_file": str(tmp_path / "split.csv"),
            "output_dir": str(tmp_path),
            "eps_prevalence": 1e-5,
            "use_cyclical_time": True,
        }
    )
    cfg.update(overrides)
    return cfg


@pytest.fixture
def cfg_factory(tmp_path):
    def _factory(predictor="prevalence", **overrides):
        return make_cfg(tmp_path, predictor, **overrides)

    return _factory


@pytest.fixture
def batch():
    rng = np.random.default_rng(1)
    return {
        "x": rng.standard_normal((2, 6, 3)).astype(np.float32),
        "y": rng.standard_normal((2, 6)).astype(np.float32),
        "w": np.ones((2, 6), dtype=np.float32),
    }


@pytest.fixture
def identity_scaler():
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(STATIC_COVARS), dtype=np.float32)
    scaler.scale_ = np.ones(len(STATIC_COVARS), dtype=np.float32)
    return scaler


@pytest.fixture
def preprocessing_config_factory():
    def _factory(predictor="prevalence", n_steps=INFERENCE_N_STEPS, **overrides):
        config = {
            "static_covars": STATIC_COVARS,
            "after_intervention": AFTER_INTERVENTION_COVARS,
            "intervention_day": INTERVENTION_DAY,
            "use_cyclical_time": True,
            "window_size": 14,
            "n_steps": n_steps,
            "burnin_day": BURNIN_DAY,
            "predictor": predictor,
        }
        config.update(overrides)
        return config

    return _factory


@pytest.fixture
def preprocessing_config(preprocessing_config_factory):
    return preprocessing_config_factory()


@pytest.fixture
def static_covar_dicts():
    rng = np.random.default_rng(7)
    covar_dicts = []
    for _ in range(3):
        static = {c: float(rng.uniform(0.0, 1.0)) for c in STATIC_COVARS}
        static["eir"] = float(rng.uniform(0.0, 200.0))
        static["seasonal"] = 1.0
        static["routine"] = 1.0
        covar_dicts.append(static)
    return covar_dicts


@pytest.fixture
def straddling_abs_t():
    return np.array(
        [INTERVENTION_DAY - 10, INTERVENTION_DAY - 5, INTERVENTION_DAY, INTERVENTION_DAY + 10],
        dtype=np.float32,
    )


@pytest.fixture
def simple_sub(straddling_abs_t):
    t = len(straddling_abs_t)
    rng = np.random.default_rng(42)
    data = {
        "timesteps": list(range(1, t + 1)),
        "abs_timesteps": straddling_abs_t.tolist(),
        "prevalence": [0.1, 0.2, 0.3, 0.4],
        "cases": [5.0, 10.0, 15.0, 20.0],
        "exposure_pd": [100.0, 200.0, 300.0, 400.0],
    }
    for covar in STATIC_COVARS:
        data[covar] = [float(rng.uniform(0.1, 1.0))] * t
    return pd.DataFrame(data)


@pytest.fixture
def base_static(simple_sub):
    return np.asarray(simple_sub.iloc[0][STATIC_COVARS].values, dtype=np.float32)


@pytest.fixture
def model_factory(tiny_model_kwargs):
    def _factory(input_dim=3, seed=0, **overrides):
        from flax import nnx

        from stateMINT.model import Mamba2Regressor

        kwargs = {**tiny_model_kwargs, **overrides}
        return Mamba2Regressor(input_dim=input_dim, rngs=nnx.Rngs(seed), **kwargs)

    return _factory


@pytest.fixture
def model_and_optimizer_factory(model_factory):
    def _factory(input_dim=3, learning_rate=1e-3, total_steps=10, seed=0):
        from stateMINT.training.train_step import create_optimizer

        model = model_factory(input_dim=input_dim, seed=seed)
        optimizer = create_optimizer(model, learning_rate=learning_rate, total_steps=total_steps)
        return model, optimizer

    return _factory


@pytest.fixture
def loader_records_factory():
    def _factory(n, x_shape=(4, 3), include_ps=False):
        records: list[dict[str, np.ndarray]] = [
            {
                "x": np.zeros(x_shape, dtype=np.float32),
                "y": np.zeros(x_shape[0], dtype=np.float32),
            }
            for _ in range(n)
        ]
        if include_ps:
            for record in records:
                record["ps"] = np.zeros((x_shape[0], 2), dtype=np.int32)
        return records

    return _factory


@pytest.fixture
def pretrained_input():
    import jax.numpy as jnp

    return jnp.ones((1, INFERENCE_N_STEPS, INPUT_SIZE), dtype=jnp.float32)


@pytest.fixture
def model_artifact_factory(identity_scaler, preprocessing_config_factory):
    def _factory(predictor="prevalence", n_steps=10):
        from unittest.mock import MagicMock

        from stateMINT.model.hub import ModelArtifact

        return ModelArtifact(
            model=MagicMock(),
            model_config={},
            preprocessing_config=preprocessing_config_factory(predictor=predictor, n_steps=n_steps),
            scaler=identity_scaler,
        )

    return _factory
