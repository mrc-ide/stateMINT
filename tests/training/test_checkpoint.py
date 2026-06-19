import numpy as np
from flax import nnx

from stateMINT.training.checkpoint import checkpoint_session


def test_scratch_start_when_no_checkpoint(model_and_optimizer_factory, tmp_path):
    model, opt = model_and_optimizer_factory()
    with checkpoint_session(tmp_path / "ck", 1, model, opt, restore_checkpoint=False) as sess:
        assert sess.start_epoch == 0
        assert sess.best_val_loss == float("inf")


def test_save_if_best_only_on_improvement(model_and_optimizer_factory, tmp_path):
    model, opt = model_and_optimizer_factory()
    with checkpoint_session(tmp_path / "ck", 1, model, opt) as sess:
        assert sess.save_if_best(0, val_loss=1.0) is True
        assert sess.save_if_best(1, val_loss=2.0) is False  # worse, not saved
        assert sess.save_if_best(2, val_loss=0.5) is True  # better, saved
        assert sess.best_val_loss == 0.5


def test_restore_recovers_state_and_epoch(model_and_optimizer_factory, tmp_path):
    ck_dir = tmp_path / "ck"
    model, opt = model_and_optimizer_factory()
    with checkpoint_session(ck_dir, 1, model, opt) as sess:
        sess.save_if_best(3, val_loss=0.25)
        saved = nnx.state(sess.model, nnx.Param)

    # New objects, restore from disk.
    model2, opt2 = model_and_optimizer_factory()
    with checkpoint_session(ck_dir, 1, model2, opt2, restore_checkpoint=True) as sess2:
        assert sess2.start_epoch == 4  # checkpoint epoch + 1
        assert sess2.best_val_loss == 0.25
        restored = nnx.state(sess2.model, nnx.Param)

    import jax

    for a, b in zip(jax.tree_util.tree_leaves(saved), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
