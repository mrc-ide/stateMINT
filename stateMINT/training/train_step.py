from typing import Callable

import flax.nnx as nnx
import jax.numpy as jnp
import optax
from jaxtyping import Array

from ..common.dataclasses import Predictor
from ..common.utils import forward, inverse_transform_jax
from .loss import weighted_mse


def _compute_loss(
    model: nnx.Module,
    batch: dict,
    predictor: Predictor,
    diff_alpha: float,
    loss_method: Callable[[Array, Array, Array], Array],
) -> Array:
    """
    Compute loss for a batch, including shape-aware losses on temporal derivatives.

    The base loss is evaluated after inverse-transforming predictions and targets, then
    a first-difference loss is added to encourage matching temporal shape.

    Args:
        model: Model to evaluate.
        batch: Training batch.
        predictor: Target type.
        diff_alpha: Derivative loss weight.
        loss_method: Base loss function.

    Returns:
        Scalar loss.
    """
    pred = forward(model, batch["x"])  # (B, T)
    target = batch["y"]  # (B, T, )
    w = batch["w"]  # (B, T, )

    pred = inverse_transform_jax(pred, predictor)
    target = inverse_transform_jax(target, predictor)

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
    """
    Create a jitted training step.

    Args:
        predictor: Target type.
        diff_alpha: Derivative loss weight.
        loss_method: Base loss function.

    Returns:
        Training step function.
    """
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict) -> Array:
        """
        Run one optimizer update.

        Args:
            model: Model to update.
            optimizer: Optimizer to apply.
            batch: Training batch.

        Returns:
            Scalar loss.
        """
        def loss_fn(model: nnx.Module) -> Array:
            """
            Compute differentiable loss.

            Args:
                model: Model to evaluate.

            Returns:
                Scalar loss.
            """
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
    """
    Create a jitted evaluation step.

    Args:
        predictor: Target type.
        diff_alpha: Derivative loss weight.
        loss_method: Base loss function.

    Returns:
        Evaluation step function.
    """
    @nnx.jit
    def eval_step(model: nnx.Module, batch: dict) -> Array:
        """
        Compute evaluation loss for one batch.

        Args:
            model: Model to evaluate.
            batch: Evaluation batch.

        Returns:
            Scalar loss.
        """
        return _compute_loss(
            model,
            batch,
            predictor,
            diff_alpha,
            loss_method,
        )

    return eval_step


def create_optimizer(model: nnx.Module, learning_rate: float, total_steps: int) -> nnx.Optimizer:
    """
    Create the training optimizer.

    Args:
        model: Model to optimize.
        learning_rate: Peak learning rate.
        total_steps: Total training steps.

    Returns:
        Configured optimizer.
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=int(0.03 * total_steps),  # warmup for 3% of training
        decay_steps=total_steps,
        end_value=0.05 * learning_rate,  # decay to 5% of initial LR
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
