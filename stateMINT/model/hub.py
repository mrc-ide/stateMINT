import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from etils import epath
from flax import nnx
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp

from stateMINT.common.dataclasses import Predictor, ModelFactory
from stateMINT.data.preprocessing import StandardScaler
from stateMINT.training.checkpoint import restore_model


@dataclass
class ModelArtifact:
    model: nnx.Module
    model_config: dict[str, Any]
    preprocessing_config: dict[str, Any]
    scaler: StandardScaler


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _download_from_hf(
    repo_id: str,
    predictor: Predictor,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{predictor}/*", f"{predictor}/**"],
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    return Path(root) / predictor


def _load_scaler(mean: list[float], scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(mean, dtype=np.float32)
    scaler.scale_ = np.array(scale, dtype=np.float32)
    return scaler


def load_model_artifact(
    path_or_repo_id: str,
    predictor: Predictor,
    *,
    model_cls: type[ModelFactory],
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> ModelArtifact:
    """
    Load a stateMINT inference artifact from a local folder or Hugging Face repo.

    Expected artifact layout:
        model_config.json
        preprocessing.json
        checkpoint/

    Args:
        path_or_repo_id: Hugging Face repo ID or local folder path.
        predictor: Target predictor, either "prevalence" or "cases".
        revision: Optional revision of the model to load from the repo.
        cache_dir: Optional cache directory for Hugging Face repo.
        local_dir: Optional local directory to load the artifact from.
    Return
         ModelArtifact: A dataclass containing the model, model configuration, preprocessing configuration, and scaler.
    """
    if Path(path_or_repo_id).exists():
        artifact_dir = Path(path_or_repo_id)
    else:
        artifact_dir = _download_from_hf(
            path_or_repo_id,
            predictor,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )

    model_config = _load_json(artifact_dir / "model_config.json")
    preprocessing_config = _load_json(artifact_dir / "preprocessing_config.json")
    scaler = _load_scaler(preprocessing_config["scaler_mean"], preprocessing_config["scaler_scale"])

    model = model_cls.from_cfg(OmegaConf.create(model_config), input_size=model_config["input_size"])

    ckpt_dir = epath.Path(artifact_dir / "checkpoint")
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        model = restore_model(ckptr, model)
    model.eval()

    return ModelArtifact(
        model=model,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        scaler=scaler,
    )
