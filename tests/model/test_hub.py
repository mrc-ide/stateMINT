import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from stateMINT.data import INPUT_SIZE
from stateMINT.model.hub import ModelArtifact, _download_from_hf, _load_json, _load_scaler, load_model_artifact


# --- _load_json ---


def test_load_json_reads_file(tmp_path):
    data = {"key": 42, "nested": [1, 2, 3]}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    assert _load_json(p) == data


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_json(tmp_path / "nonexistent.json")


# --- _load_scaler ---


def test_load_scaler_sets_arrays():
    scaler = _load_scaler([1.0, 2.0], [3.0, 4.0])
    np.testing.assert_array_equal(scaler.mean_, [1.0, 2.0])
    np.testing.assert_array_equal(scaler.scale_, [3.0, 4.0])


# --- _download_from_hf ---


def test_download_from_hf_returns_predictor_subpath():
    with patch("stateMINT.model.hub.snapshot_download", return_value="/cache/root") as mock_dl:
        result = _download_from_hf("org/repo", "prevalence")

    assert result == Path("/cache/root/prevalence")
    mock_dl.assert_called_once_with(
        repo_id="org/repo",
        allow_patterns=["prevalence/*", "prevalence/**"],
        revision=None,
        cache_dir=None,
        local_dir=None,
    )


def test_download_from_hf_forwards_optional_args():
    with patch("stateMINT.model.hub.snapshot_download", return_value="/cache/root") as mock_dl:
        _download_from_hf("org/repo", "cases", revision="v1", cache_dir="/c", local_dir="/l")

    _, kwargs = mock_dl.call_args
    assert kwargs["revision"] == "v1"
    assert kwargs["cache_dir"] == "/c"
    assert kwargs["local_dir"] == "/l"


# --- load_model_artifact ---


def _make_artifact_dir(tmp_path: Path) -> Path:
    model_cfg = {"input_size": 4, "d_model": 16, "n_layers": 1, "seed": 0}
    preprocessing_cfg = {"scaler_mean": [0.0, 1.0], "scaler_scale": [1.0, 2.0]}
    (tmp_path / "model_config.json").write_text(json.dumps(model_cfg))
    (tmp_path / "preprocessing_config.json").write_text(json.dumps(preprocessing_cfg))
    (tmp_path / "checkpoint").mkdir()
    return tmp_path


@pytest.fixture
def artifact_dir(tmp_path):
    return _make_artifact_dir(tmp_path)


@pytest.fixture
def mock_model_cls():
    cls = MagicMock()
    cls.from_cfg.return_value = MagicMock()
    return cls


def _patch_checkpoint(mock_model):
    ckptr = MagicMock()
    ckptr.__enter__ = MagicMock(return_value=ckptr)
    ckptr.__exit__ = MagicMock(return_value=False)
    checkpointer_cls = MagicMock(return_value=ckptr)
    restore = MagicMock(return_value=mock_model)
    return checkpointer_cls, restore


def test_load_model_artifact_local_path(artifact_dir, mock_model_cls):
    mock_model = mock_model_cls.from_cfg.return_value
    ckptr_cls, restore = _patch_checkpoint(mock_model)

    with (
        patch("stateMINT.model.hub.ocp.training.Checkpointer", ckptr_cls),
        patch("stateMINT.model.hub.restore_model", restore),
        patch("stateMINT.model.hub._download_from_hf") as mock_dl,
    ):
        result = load_model_artifact(str(artifact_dir), "prevalence", model_cls=mock_model_cls)

    mock_dl.assert_not_called()
    assert isinstance(result, ModelArtifact)
    assert result.model is mock_model
    assert result.model_config["input_size"] == 4
    mock_model.eval.assert_called_once()


def test_load_model_artifact_downloads_when_path_missing(tmp_path, mock_model_cls):
    subdir = tmp_path / "artifact"
    subdir.mkdir()
    artifact_dir = _make_artifact_dir(subdir)
    mock_model = mock_model_cls.from_cfg.return_value
    ckptr_cls, restore = _patch_checkpoint(mock_model)

    with (
        patch("stateMINT.model.hub.ocp.training.Checkpointer", ckptr_cls),
        patch("stateMINT.model.hub.restore_model", restore),
        patch("stateMINT.model.hub._download_from_hf", return_value=artifact_dir) as mock_dl,
    ):
        load_model_artifact("org/repo", "prevalence", model_cls=mock_model_cls)

    mock_dl.assert_called_once_with("org/repo", "prevalence", revision=None, cache_dir=None, local_dir=None)


def test_load_model_artifact_scaler_populated(artifact_dir, mock_model_cls):
    mock_model = mock_model_cls.from_cfg.return_value
    ckptr_cls, restore = _patch_checkpoint(mock_model)

    with (
        patch("stateMINT.model.hub.ocp.training.Checkpointer", ckptr_cls),
        patch("stateMINT.model.hub.restore_model", restore),
    ):
        result = load_model_artifact(str(artifact_dir), "prevalence", model_cls=mock_model_cls)

    np.testing.assert_array_equal(result.scaler.mean_, [0.0, 1.0])
    np.testing.assert_array_equal(result.scaler.scale_, [1.0, 2.0])


def test_load_model_artifact_missing_scaler_key(artifact_dir, mock_model_cls):
    bad_preprocessing = {"scaler_mean": [0.0, 1.0]}  # missing scaler_scale
    (artifact_dir / "preprocessing_config.json").write_text(json.dumps(bad_preprocessing))

    ckptr_cls, restore = _patch_checkpoint(mock_model_cls.from_cfg.return_value)

    with (
        patch("stateMINT.model.hub.ocp.training.Checkpointer", ckptr_cls),
        patch("stateMINT.model.hub.restore_model", restore),
        pytest.raises(KeyError),
    ):
        load_model_artifact(str(artifact_dir), "prevalence", model_cls=mock_model_cls)


# --- ModelArtifact.prepare_inputs / predict ---


def test_prepare_inputs_shape(model_artifact_factory, static_covar_dicts):
    artifact = model_artifact_factory(n_steps=10)
    X = artifact.prepare_inputs(static_covar_dicts[:2])
    assert X.shape == (2, 10, INPUT_SIZE)
    assert X.dtype == np.float32


def test_predict_applies_inverse_transform(model_artifact_factory, static_covar_dicts):
    artifact = model_artifact_factory("prevalence", n_steps=10)
    transformed = jnp.zeros((2, 10))  # logit 0 -> sigmoid -> 0.5
    with patch("stateMINT.model.hub.forward", return_value=transformed) as mock_forward:
        preds = artifact.predict(static_covar_dicts[:2])

    mock_forward.assert_called_once()
    assert preds.shape == (2, 10) and preds.dtype == np.float32
    np.testing.assert_allclose(preds, 0.5, atol=1e-6)


def test_predict_transformed_skips_inverse(model_artifact_factory, static_covar_dicts):
    artifact = model_artifact_factory("prevalence", n_steps=10)
    transformed = jnp.full((2, 10), 0.7)
    with patch("stateMINT.model.hub.forward", return_value=transformed):
        preds = artifact.predict(static_covar_dicts[:2], transformed=True)

    np.testing.assert_allclose(preds, 0.7, atol=1e-6)


def test_predict_cases_uses_expm1(model_artifact_factory, static_covar_dicts):
    artifact = model_artifact_factory("cases", n_steps=10)
    transformed = jnp.full((2, 10), float(np.log1p(3.0)))  # expm1 -> 3.0
    with patch("stateMINT.model.hub.forward", return_value=transformed):
        preds = artifact.predict(static_covar_dicts[:2])

    np.testing.assert_allclose(preds, 3.0, rtol=1e-5)
