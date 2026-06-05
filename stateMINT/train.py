import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from .common.dataclasses import Predictor
import wandb
import jax
import jax.numpy as jnp
from mamba2_jax import Mamba2Forecaster, create_random_forecaster
from flax import nnx
import duckdb
from pathlib import Path
from .data import make_loader, prepare_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.predictor not in ("prevalence", "cases"):
        raise ValueError(f"Unknown predictor: {cfg.predictor}")

    log.info(OmegaConf.to_yaml(cfg))
    log.info("JAX devices:", jax.devices())

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
            settings=wandb.Settings(start_method="thread"),
        )
    # ------------------- data loading and preprocessing -----------------------------
    log.info("Loading and preprocessing data...")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    raw_df = duckdb.read_parquet(cfg.data_file).df()

    prepared_data = prepare_data(raw_df, cfg)

    train_loader = make_loader(
        data=prepared_data.train_data,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    val_loader = make_loader(
        data=prepared_data.val_data,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=False,
    )
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        num_epochs=1,  # only one pass for testing
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=False,
    )
    # --- model + optimizer ---
    rngs = nnx.Rngs(params=cfg.seed, dropout=cfg.seed + 1)


if __name__ == "__main__":
    main()
