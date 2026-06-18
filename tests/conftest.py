import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from stateMINT.data.preprocessing import STATIC_COVARS, INTERVENTION_DAY


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
