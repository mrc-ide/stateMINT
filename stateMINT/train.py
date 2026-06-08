import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from .common.dataclasses import Predictor
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import duckdb
from pathlib import Path
from .data import make_loader, prepare_data
from .model import Mamba2Regressor
import optax


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    if cfg.predictor not in ("prevalence", "cases"):
        raise ValueError(f"Unknown predictor: {cfg.predictor}")

    log.info(OmegaConf.to_yaml(cfg))
    log.info("JAX devices: %s", jax.devices())

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
    rngs = nnx.Rngs(cfg.seed)
    model = Mamba2Regressor(
        input_dim=prepared_data.input_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        head_dim=cfg.head_dim,
        chunk_size=cfg.chunk_size,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
        rngs=rngs,
    )
    params = nnx.state(model, nnx.Param)
    total_params = sum(jnp.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    log.info(f"Total parameters: {total_params / 1e6:.2f}M")

    # scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=cfg.learning_rate,
    #     warmup_steps=int(0.01 * cfg.num_epochs * len(train_loader)),  # warmup for 1% of training
    #     decay_steps=cfg.num_epochs * len(train_loader),
    #     end_value=0.1 * cfg.learning_rate,  # decay to 10% of initial LR
    # )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=cfg.learning_rate))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)


if __name__ == "__main__":
    main()
