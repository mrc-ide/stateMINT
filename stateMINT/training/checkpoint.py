import logging
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterator

import flax.nnx as nnx
from orbax.checkpoint import v1 as ocp
from etils import epath

log = logging.getLogger(__name__)


def init_or_restore_last(
    ckptr: ocp.training.Checkpointer,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    restore_checkpoint: bool = False,
) -> tuple[nnx.Module, nnx.Optimizer, int, float]:
    """Initialize or restore model and optimizer from checkpoint.

    Args:
        ckptr: An Orbax Checkpointer instance.
        model: The model to initialize or restore.
        optimizer: The optimizer to initialize or restore.
        restore_checkpoint: Whether to attempt restoring from checkpoint. If False, will always initialize from scratch

    Returns:
    A tuple of (model, optimizer, start_epoch, val_loss),
    where start_epoch is the epoch to start training from (0 if initializing from scratch, or the checkpoint epoch + 1 if restoring),
    and val_loss is the validation loss at the checkpoint (float('inf') if initializing from
    scratch, or the actual val loss from checkpoint metadata if restoring).

    """
    if not restore_checkpoint or not ckptr.latest:
        log.info("Initializing model and optimizer from scratch.")
        return model, optimizer, 0, float("inf")

    log.info(f"Restoring model and optimizer from checkpoint: {ckptr.latest.step}")
    abstract_state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    loaded_state = ckptr.load(abstract_state=abstract_state)
    nnx.update(model, loaded_state["model"])
    nnx.update(optimizer, loaded_state["optimizer"])
    metadata = ckptr.metadata()
    metrics: dict[str, Any] = metadata.metrics if isinstance(metadata.metrics, dict) else {}
    val_loss = float(metrics.get("val/loss", float("inf")))

    return model, optimizer, ckptr.latest.step + 1, val_loss


@dataclass
class CheckpointSession:
    ckptr: ocp.training.Checkpointer
    model: nnx.Module
    optimizer: nnx.Optimizer
    start_epoch: int
    best_val_loss: float

    def save_if_best(self, epoch: int, val_loss: float) -> bool:
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        self.ckptr.save(
            epoch,
            {
                "model": nnx.state(self.model),
                "optimizer": nnx.state(self.optimizer),
            },
            metrics={"val/loss": self.best_val_loss},
        )
        return True


@contextmanager
def checkpoint_session(
    checkpoint_dir: str | PathLike[str],
    max_checkpoints_to_keep: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    restore_checkpoint: bool = False,
) -> Iterator[CheckpointSession]:
    ckpt_dir = epath.Path(checkpoint_dir).resolve()
    with ocp.training.Checkpointer(
        ckpt_dir,
        preservation_policy=ocp.training.preservation_policies.LatestN(max_checkpoints_to_keep),  # type: ignore[arg-type]
    ) as ckptr:
        model, optimizer, start_epoch, best_val_loss = init_or_restore_last(ckptr, model, optimizer, restore_checkpoint)
        yield CheckpointSession(
            ckptr=ckptr,
            model=model,
            optimizer=optimizer,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
        )
