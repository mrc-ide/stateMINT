import logging

import duckdb
import hydra
from etils import epath
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import v1 as ocp

from stateMINT.data.dataset import make_loader
from stateMINT.data.preprocessing import prepare_data
from stateMINT.eval.metrics import get_preds_targets
from stateMINT.eval.viz_preds_truth import plot_preds_targets
from stateMINT.model.mamba2 import Mamba2Regressor
from stateMINT.training.checkpoint import restore_model

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
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=1,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    model = Mamba2Regressor.from_cfg(cfg, prepared_data.input_size)
    ckpt_dir = epath.Path(cfg.checkpoint_dir).resolve()
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        model = restore_model(ckptr, model)
        model.eval()
        preds, targets, ps = get_preds_targets(model, test_loader)
        log.info("Predictions shape: %s, Targets shape: %s, Ps shape: %s", preds.shape, targets.shape, ps.shape)
        plot_preds_targets(
            preds, targets, ps, cfg.plot_file, window_size=cfg.window_size, predictor=cfg.predictor, ylabel=cfg.ylabel
        )

    log.info(f"Finished running visualization saved to {cfg.plot_file}")


if __name__ == "__main__":
    main()
