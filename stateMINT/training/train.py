import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable
from jaxtyping import Array
import optax
from functools import partial
import jax
from ..common.utils import inverse_transform_jax
from .loss import weighted_mse
from ..common.dataclasses import Predictor
import grain.python as grain


def _to_natural(pred: Array, target: Array, predictor: Predictor) -> tuple[Array, Array]:
    """Convert predictions and targets to natural space if loss is computed in transformed space."""
    pred = inverse_transform_jax(pred, predictor)
    target = inverse_transform_jax(target, predictor)
    return pred, target


@nnx.jit
def _forward(model: nnx.Module, x: Array) -> Array:
    """Forward pass through the model."""
    return model(x).squeeze(-1)  # (B, T, 1) -> (B, T)


def _compute_loss(
    model: nnx.Module,
    batch: dict,
    predictor: Predictor,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array],
) -> Array:
    """Compute loss for a batch, including shape-aware losses on temporal derivatives."""
    pred = _forward(model, batch["x"])  # (B, T)
    target = batch["y"]  # (B, T, )
    w = batch["w"]  # (B, T, )

    pred, target = _to_natural(pred, target, predictor)

    base_loss = loss_method(pred, target, w)

    # Shape-aware losses on temporal derivatives
    d1_pred = jnp.diff(pred, axis=1)  # (B, T-1)
    d1_target = jnp.diff(target, axis=1)  # (B, T-1)
    w_mid = (w[:, 1:] + w[:, :-1]) * 0.5  # (B, T-1) - average weight for adjacent timesteps
    d1 = loss_method(d1_pred, d1_target, w_mid)

    loss = base_loss + diff_alpha * d1
    return loss


def make_train_step(
    predictor: Predictor,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array] = weighted_mse,
) -> Callable[[nnx.Module, nnx.Optimizer, dict], Array]:
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict) -> Array:
        def loss_fn(model: nnx.Module) -> Array:
            return _compute_loss(model, batch, predictor, diff_alpha, loss_method)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_eval_step(
    predictor: Predictor,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array] = weighted_mse,  # loss takes (pred, target, w)
) -> Callable[[nnx.Module, dict], Array]:
    @nnx.jit
    def eval_step(model: nnx.Module, batch: dict) -> Array:
        return _compute_loss(
            model,
            batch,
            predictor,
            diff_alpha,
            loss_method,
        )

    return eval_step


@partial(jax.jit, static_argnames=["predictor"])
def _metrics_from_preds_targets(preds: Array, targets: Array, predictor: Predictor) -> dict[str, Array | float]:
    """Compute evaluation metrics on the given predictions and targets."""
    preds, targets = _to_natural(preds, targets, predictor)

    eps = 1e-7
    mse = jnp.mean((preds - targets) ** 2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(preds - targets))
    r2 = 1.0 - jnp.sum((targets - preds) ** 2) / jnp.sum((targets - jnp.mean(targets)) ** 2)
    smape = 100 * jnp.mean(2 * jnp.abs(preds - targets) / (jnp.abs(preds) + jnp.abs(targets) + eps))
    bias = jnp.mean(preds - targets)

    if predictor == "prevalence":
        sp = jnp.clip(preds, eps, 1 - eps)
        st = jnp.clip(targets, eps, 1 - eps)
        log_likelihood = jnp.mean(jnp.log(sp) * st + jnp.log(1 - sp) * (1 - st))
    else:
        log_likelihood = jnp.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "bias": bias,
        "log_likelihood": log_likelihood,
    }


def compute_metrics(
    model: nnx.Module,
    data_loader: grain.DataLoader,
    predictor: Predictor,
) -> dict[str, float]:
    """Compute evaluation metrics on the given data loader."""
    all_preds, all_targets = [], []
    for batch in data_loader:
        all_preds.append(_forward(model, batch["x"]))
        all_targets.append(batch["y"])

    # Concatenate (still transformed) and invert back to natural space
    preds = jnp.concatenate(all_preds)
    targets = jnp.concatenate(all_targets)

    metrics = _metrics_from_preds_targets(preds, targets, predictor)
    return {k: float(v) for k, v in metrics.items()}


def create_optimizer(model: nnx.Module, learning_rate: float, total_steps: int) -> nnx.Optimizer:
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=int(0.03 * total_steps),  # warmup for 3% of training
        decay_steps=total_steps,
        end_value=0.1 * learning_rate,  # decay to 10% of initial LR
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
