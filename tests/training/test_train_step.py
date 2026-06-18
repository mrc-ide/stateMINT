import jax.numpy as jnp
import numpy as np
from flax import nnx

from stateMINT.model import Mamba2Regressor
from stateMINT.training.train_step import (
    create_optimizer,
    make_train_step,
    make_eval_step,
    _compute_loss,
)


def _model(tiny_model_kwargs, input_dim=3):
    return Mamba2Regressor(input_dim=input_dim, rngs=nnx.Rngs(0), **tiny_model_kwargs)


def test_compute_loss_is_scalar(tiny_model_kwargs, batch):
    model = _model(tiny_model_kwargs)
    loss = _compute_loss(model, batch, "prevalence", 0.05, lambda p, t, w: jnp.mean((p - t) ** 2 * (w > -1)))
    assert loss.shape == ()
    assert np.isfinite(float(loss))


def test_train_step_reduces_loss(tiny_model_kwargs, batch):
    from stateMINT.training.loss import weighted_mse

    model = _model(tiny_model_kwargs)
    optimizer = create_optimizer(model, learning_rate=1e-2, total_steps=20)
    step = make_train_step("prevalence", diff_alpha=0.05, loss_method=weighted_mse)

    first = float(step(model, optimizer, batch))
    last = first
    for _ in range(10):
        last = float(step(model, optimizer, batch))
    assert last < first  # overfits the single batch


def test_eval_step_does_not_update_params(tiny_model_kwargs, batch):
    from stateMINT.training.loss import weighted_mse

    model = _model(tiny_model_kwargs)
    before = nnx.state(model, nnx.Param)
    eval_step = make_eval_step("prevalence", diff_alpha=0.0, loss_method=weighted_mse)
    eval_step(model, batch)
    after = nnx.state(model, nnx.Param)
    leaves_before = jax_leaves(before)
    leaves_after = jax_leaves(after)
    for a, b in zip(leaves_before, leaves_after):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def jax_leaves(state):
    import jax

    return [x for x in jax.tree_util.tree_leaves(state) if hasattr(x, "shape")]
