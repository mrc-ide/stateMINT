import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from .common.dataclasses import Predictor
import wandb

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
            settings=wandb.Settings(start_method="thread"),
        )


if __name__ == "__main__":
    main()
