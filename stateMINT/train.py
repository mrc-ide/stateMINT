import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from .common.dataclasses import Predictor
import wandb
import jax
import jax.numpy as jnp
from mamba2_jax import Mamba2Forecaster, create_random_forecaster
from flax import nnx

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.predictor not in ("prevalence", "cases"):
        raise ValueError(f"Unknown predictor: {cfg.predictor}")
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
            settings=wandb.Settings(start_method="thread"),
        )
    print("JAX devices:", jax.devices())


if __name__ == "__main__":
    main()
