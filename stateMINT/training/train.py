import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable
from jaxtyping import Array
import optax
from ..common.utils import inverse_transform_jax
from .loss import weighted_mse
from ..common.dataclasses import Predictor, LossSpace
import grain.python as grain


def _to_natural(pred: Array, target: Array, predictor: Predictor, loss_space: LossSpace) -> tuple[Array, Array]:
    """Convert predictions and targets to natural space if loss is computed in transformed space."""
    if loss_space == "natural":
        pred = inverse_transform_jax(pred, predictor)
        target = inverse_transform_jax(target, predictor)
    return pred, target


def _compute_loss(
    model: nnx.Module,
    batch: dict,
    predictor: Predictor,
    loss_space: LossSpace,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array],
) -> Array:
    """Compute loss for a batch, including shape-aware losses on temporal derivatives."""
    pred = model(batch["x"]).squeeze(-1)  # (B, T, 1) -> (B, T)
    target = batch["y"]  # (B, T, )
    w = batch["w"]  # (B, T, )

    pred, target = _to_natural(pred, target, predictor, loss_space)

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
    loss_space: LossSpace,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array] = weighted_mse,
) -> Callable[[nnx.Module, nnx.Optimizer, dict], Array]:
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict) -> Array:
        def loss_fn(model: nnx.Module) -> Array:
            return _compute_loss(model, batch, predictor, loss_space, diff_alpha, loss_method)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_eval_step(
    predictor: Predictor,
    loss_space: LossSpace,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array] = weighted_mse,  # loss takes (pred, target, w)
) -> Callable[[nnx.Module, dict], Array]:
    @nnx.jit
    def eval_step(model: nnx.Module, batch: dict) -> Array:
        return _compute_loss(
            model,
            batch,
            predictor,
            loss_space,
            diff_alpha,
            loss_method,
        )

    return eval_step


def make_test_step(
    predictor: Predictor,

)
def compute_metrics(model: nnx.Module, data_loader: grain.DataLoader, predictor: Predictor) -> dict[str, float]:
    """Compute evaluation metrics on the given data loader."""


def create_optimizer(model: nnx.Module, learning_rate: float) -> nnx.Optimizer:
    # scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=cfg.lr,
    #     warmup_steps=int(0.01 * cfg.num_epochs * len(train_loader)),  # warmup for 1% of training
    #     decay_steps=cfg.num_epochs * len(train_loader),
    #     end_value=0.1 * cfg.lr,  # decay to 10% of initial LR
    # )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=learning_rate))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
