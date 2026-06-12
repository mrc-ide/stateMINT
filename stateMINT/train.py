import logging
import time
from pathlib import Path

import duckdb
import hydra
import jax
import jax.numpy as jnp
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from stateMINT.model.mamba2 import get_total_params

from .data import make_loader, prepare_data
from .eval.metrics import compute_metrics
from .model import Mamba2Regressor
from .training.checkpoint import checkpoint_session, init_or_restore_last
from .training.train_step import create_optimizer, make_eval_step, make_train_step

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """
    Train and evaluate the model.

    Args:
        cfg: Hydra training config.

    """
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

    # TODO: sort what to do with drop_remainder
    val_loader = make_loader(
        data=prepared_data.val_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )

    # ------------- model + optimizer setup------------------------
    model = Mamba2Regressor.from_cfg(cfg, prepared_data.input_size)
    log.info(f"Total parameters: {get_total_params(model) / 1e6:.2f}M")

    total_steps = cfg.num_epochs * len(prepared_data.train_data) // cfg.batch_size
    log.info(f"Total training steps: {total_steps}")
    optimizer = create_optimizer(model, cfg.lr, total_steps)

    # ------------------- training loop -----------------------------
    loss_method = get_method(cfg.loss_method)
    train_step = make_train_step(cfg.predictor, cfg.diff_loss_alpha, loss_method)
    eval_step = make_eval_step(cfg.predictor, cfg.diff_loss_alpha, loss_method)

    with checkpoint_session(
        checkpoint_dir=cfg.checkpoint_dir,
        max_checkpoints_to_keep=cfg.max_checkpoints_to_keep,
        model=model,
        optimizer=optimizer,
        restore_checkpoint=cfg.restore_checkpoint,
    ) as ckpt:
        patience_n = 0
        model = ckpt.model
        optimizer = ckpt.optimizer

        epoch_pbar = tqdm(range(ckpt.start_epoch, cfg.num_epochs), desc="Epoch")
        started = time.perf_counter()
        for epoch in epoch_pbar:
            # remake train loader each epoch to reshuffle with new seed
            train_loader = make_loader(
                data=prepared_data.train_data,
                batch_size=cfg.batch_size,
                seed=cfg.seed + epoch,
                shuffle=True,
                num_workers=cfg.num_workers,
                drop_remainder=True,
            )
            model.train()
            train_losses: list[jax.Array] = [train_step(model, optimizer, batch) for batch in train_loader]

            model.eval()
            val_losses: list[jax.Array] = [eval_step(model, batch) for batch in val_loader]

            avg_train_loss = float(jnp.mean(jnp.stack(train_losses)))
            avg_val_loss = float(jnp.mean(jnp.stack(val_losses)))

            epoch_pbar.set_postfix(
                train=f"{avg_train_loss:.6f}",
                val=f"{avg_val_loss:.6f}",
                patience=f"{patience_n}/{cfg.patience}",
            )

            if cfg.use_wandb:
                wandb.log({"train/loss": avg_train_loss, "val/loss": avg_val_loss, "epoch": epoch})

            # Check for improvement & save checkpoint if improved
            if ckpt.save_if_best(epoch, avg_val_loss):
                patience_n = 0
            else:
                patience_n += 1
                if patience_n >= cfg.patience:
                    tqdm.write(f"No improvement for {patience_n} epochs, stopping training.")
                    break

        # ------------ test evaluation ----------------
        model, _, _, _ = init_or_restore_last(ckpt.ckptr, ckpt.model, ckpt.optimizer, restore_checkpoint=True)
        model.eval()
        metrics = compute_metrics(model, test_loader, cfg.predictor)
        training_loop_time = time.perf_counter() - started

        tqdm.write(f"Test metrics: {metrics}")
        tqdm.write(f"Training time: {training_loop_time:.2f} seconds")
        if cfg.use_wandb:
            wandb.log({f"test/{k}": v for k, v in metrics.items()})
            wandb.log({"training_loop_time_seconds": training_loop_time})
            wandb.finish()


if __name__ == "__main__":
    main()
