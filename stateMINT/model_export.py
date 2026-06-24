import logging
import pickle
import json
from flax import nnx
import hydra
from etils import epath
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import v1 as ocp

from stateMINT.data import (
    AFTER_INTERVENTION_COVARS,
    MODEL_START_DAY,
    INTERVENTION_DAY,
    STATIC_COVARS,
    TOTAL_DAYS,
    get_input_size,
    StandardScaler,
)
from stateMINT.model.mamba2 import Mamba2Regressor
from stateMINT.training.checkpoint import restore_model

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="export_config")
def main(cfg: DictConfig) -> None:
    """
    Export the trained model for sharing to other users.

    Args:
        cfg: Hydra config for exporting the model.
    """
    log.info(OmegaConf.to_yaml(cfg))

    artifact_dir = epath.Path(cfg.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = epath.Path(cfg.checkpoint_dir).resolve()

    with open(cfg.scaler_file, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    if scaler.mean_ is None or scaler.scale_ is None:
        raise ValueError("Scaler has not been fitted yet. Please fit the scaler before exporting the model.")

    model = Mamba2Regressor.from_cfg(cfg, input_size=get_input_size(cfg.use_cyclical_time))
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        model = restore_model(ckptr, model)
    model.eval()

    with ocp.training.Checkpointer(artifact_dir / "checkpoint") as ckptr:
        ckptr.save_checkpointables(0, {"model": nnx.state(model)}, overwrite=True)

    model_config = dict(
        model_type="Mamba2Regressor",
        predictor=cfg.predictor,
        input_size=get_input_size(cfg.use_cyclical_time),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        head_dim=cfg.head_dim,
        chunk_size=cfg.chunk_size,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
    )
    n_steps = (TOTAL_DAYS - MODEL_START_DAY) // cfg.window_size + 1
    preprocessing_config = dict(
        static_covars=STATIC_COVARS,
        after_intervention=AFTER_INTERVENTION_COVARS,
        intervention_day=INTERVENTION_DAY,
        n_steps=n_steps,
        model_start_day=MODEL_START_DAY,
        window_size=cfg.window_size,
        use_cyclical_time=cfg.use_cyclical_time,
        predictor=cfg.predictor,
        eps_prevalence=cfg.eps_prevalence,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
    )

    with (artifact_dir / "model_config.json").open("w") as f:
        json.dump(model_config, f, indent=2)
    with (artifact_dir / "preprocessing_config.json").open("w") as f:
        json.dump(preprocessing_config, f, indent=2)

    log.info(f"Exported model and preprocessing configs to {artifact_dir}")


if __name__ == "__main__":
    main()
