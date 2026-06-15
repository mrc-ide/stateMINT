import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import duckdb
from stateMINT.data.dataset import make_loader
from stateMINT.data.preprocessing import prepare_data
from stateMINT.model.mamba2 import Mamba2Regressor
from orbax.checkpoint import v1 as ocp

from stateMINT.training.checkpoint import init_or_restore_last
from stateMINT.training.train_step import create_optimizer
from stateMINT.eval.metrics import get_preds_targets
from stateMINT.eval.viz_preds_truth import plot_preds_targets
from etils import epath


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="viz_config")
def main(cfg: DictConfig) -> None:
    """
    Produce visualizations for test set predictions.

    Args:
        cfg: Hydra config for visualizations.
    """
    log.info(OmegaConf.to_yaml(cfg))
    raw_df = duckdb.read_parquet(cfg.data_file).df()

    prepared_data = prepare_data(raw_df, cfg)
    # TODO: do we need to make loader or just pass test_data straight.
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    model = Mamba2Regressor.from_cfg(cfg, prepared_data.input_size)
    total_steps = len(prepared_data.train_data) // cfg.batch_size
    optimizer = create_optimizer(model, cfg.lr, total_steps)  # TODO: dont need so refactor so dont need it
    ckpt_dir = epath.Path(cfg.checkpoint_dir).resolve()
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        model, _, _, _ = init_or_restore_last(ckptr, model, optimizer, cfg.restore_checkpoint)
        model.eval()
        preds, targets, ps = get_preds_targets(model, test_loader)
        plot_preds_targets(
            preds, targets, ps, cfg.plot_file, window_size=cfg.window_size, predictor=cfg.predictor, ylabel=cfg.ylabel
        )

    log.info(f"Finished running visualization saved to {cfg.plot_file}")


if __name__ == "__main__":
    main()
