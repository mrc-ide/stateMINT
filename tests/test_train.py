import duckdb
import pytest
from omegaconf import OmegaConf

from stateMINT import train


def _train_cfg(tmp_path, data_file, predictor="prevalence"):
    return OmegaConf.create(
        {
            "predictor": predictor,
            "data_file": str(data_file),
            "split_file": str(tmp_path / "split.json"),
            "num_workers": 0,
            "use_existing_split": False,
            "eps_prevalence": 1e-5,
            "use_cyclical_time": True,
            "min_prevalence": 0.0,
            "min_cases": 0.0,
            "loss_method": "stateMINT.training.loss.weighted_mse",
            "n_layers": 1,
            "d_conv": 2,
            "expand": 2,
            "head_dim": 8,
            "chunk_size": 8,
            "output_dim": 1,
            "d_model": 16,
            "d_state": 8,
            "num_epochs": 2,
            "min_epochs": 0,
            "patience": 1,
            "diff_loss_alpha": 0.05,
            "lr": 1e-3,
            "batch_size": 1,
            "dropout": 0.0,
            "weight_decay": 1e-4,
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "restore_checkpoint": False,
            "max_checkpoints_to_keep": 1,
            "seed": 0,
            "use_wandb": False,
            "output_dir": str(tmp_path / "out"),
        }
    )


def test_train_main_rejects_unknown_predictor(tmp_path):
    cfg = _train_cfg(tmp_path, tmp_path / "data.parquet", predictor="bogus")
    with pytest.raises(ValueError, match="Unknown predictor"):
        train.main(cfg)


@pytest.mark.slow
@pytest.mark.parametrize("predictor", ["prevalence", "cases"])
def test_train_main_runs_end_to_end(tmp_path, script_sample_df, predictor):
    data_file = tmp_path / "data.parquet"
    duckdb.sql("SELECT * FROM script_sample_df").write_parquet(str(data_file))
    cfg = _train_cfg(tmp_path, data_file, predictor=predictor)

    train.main(cfg)

    assert any((tmp_path / "ckpts").iterdir())
