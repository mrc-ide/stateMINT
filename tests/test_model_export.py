import json
import pickle

import jax
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp

from stateMINT import model_export
from stateMINT.data import STATIC_COVARS, StandardScaler, get_input_size
from stateMINT.model.mamba2 import Mamba2Regressor


def _make_model():
    return Mamba2Regressor(
        input_dim=get_input_size(use_cyclical_time=True),
        d_model=16,
        n_layers=1,
        d_state=8,
        d_conv=2,
        expand=2,
        head_dim=8,
        chunk_size=8,
        output_dim=1,
        dropout=0.0,
        rngs=nnx.Rngs(0),
    )


def _export_cfg(tmp_path, checkpoint_dir, scaler_file, predictor="prevalence"):
    return OmegaConf.create(
        {
            "predictor": predictor,
            "eps_prevalence": 1e-5,
            "scaler_file": str(scaler_file),
            "window_size": 14,
            "use_cyclical_time": True,
            "n_layers": 1,
            "d_conv": 2,
            "expand": 2,
            "head_dim": 8,
            "chunk_size": 8,
            "output_dim": 1,
            "d_model": 16,
            "d_state": 8,
            "dropout": 0.0,
            "checkpoint_dir": str(checkpoint_dir),
            "seed": 0,
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )


@pytest.fixture
def fitted_scaler_file(tmp_path):
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(STATIC_COVARS), dtype=np.float32)
    scaler.scale_ = np.ones(len(STATIC_COVARS), dtype=np.float32)
    path = tmp_path / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return path


@pytest.fixture
def checkpoint_dir(tmp_path):
    model = _make_model()
    ckpt_dir = tmp_path / "ckpts"
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        ckptr.save_checkpointables(0, {"model": nnx.state(model)}, overwrite=True)
    return ckpt_dir


def test_export_main_unfitted_scaler_raises(tmp_path, checkpoint_dir):
    unfitted = tmp_path / "unfitted.pkl"
    with open(unfitted, "wb") as f:
        pickle.dump(StandardScaler(), f)
    cfg = _export_cfg(tmp_path, checkpoint_dir, unfitted)

    with pytest.raises(ValueError, match="not been fitted"):
        model_export.main(cfg)


@pytest.mark.slow
def test_export_main_writes_artifact(tmp_path, checkpoint_dir, fitted_scaler_file):
    cfg = _export_cfg(tmp_path, checkpoint_dir, fitted_scaler_file)

    model_export.main(cfg)

    artifact_dir = tmp_path / "artifacts"
    assert (artifact_dir / "checkpoint").exists()

    model_config = json.loads((artifact_dir / "model_config.json").read_text())
    assert model_config["model_type"] == "Mamba2Regressor"
    assert model_config["d_model"] == 16

    preprocessing_config = json.loads((artifact_dir / "preprocessing_config.json").read_text())
    assert preprocessing_config["predictor"] == "prevalence"
    assert preprocessing_config["scaler_mean"] == [0.0] * len(STATIC_COVARS)


@pytest.mark.slow
def test_export_main_artifact_checkpoint_roundtrips_params(tmp_path, checkpoint_dir, fitted_scaler_file):
    cfg = _export_cfg(tmp_path, checkpoint_dir, fitted_scaler_file)
    original_params = nnx.state(_make_model(), nnx.Param)

    model_export.main(cfg)

    restored = _make_model()
    with ocp.training.Checkpointer(tmp_path / "artifacts" / "checkpoint") as ckptr:
        loaded = ckptr.load_checkpointables(0, abstract_checkpointables={"model": nnx.state(restored)})
    nnx.update(restored, loaded["model"])
    restored_params = nnx.state(restored, nnx.Param)

    for a, b in zip(jax.tree_util.tree_leaves(original_params), jax.tree_util.tree_leaves(restored_params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_export_main_missing_checkpoint_raises(tmp_path, fitted_scaler_file):
    cfg = _export_cfg(tmp_path, tmp_path / "no-such-checkpoint", fitted_scaler_file)

    with pytest.raises(Exception, match="no checkpoints were found"):
        model_export.main(cfg)
