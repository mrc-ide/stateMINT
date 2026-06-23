import duckdb
import pytest
from flax import nnx
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp

from stateMINT import visualise_predictions
from stateMINT.data import get_input_size
from stateMINT.model.mamba2 import Mamba2Regressor


def _viz_cfg(tmp_path, data_file, checkpoint_dir, predictor="prevalence"):
    return OmegaConf.create(
        {
            "predictor": predictor,
            "data_file": str(data_file),
            "split_file": str(tmp_path / "split.json"),
            "num_workers": 0,
            "use_existing_split": False,
            "eps_prevalence": 1e-5,
            "window_size": 14,
            "use_cyclical_time": True,
            "min_prevalence": 0.0,
            "min_cases": 0.0,
            "n_layers": 1,
            "d_conv": 2,
            "expand": 2,
            "head_dim": 8,
            "chunk_size": 8,
            "output_dim": 1,
            "d_model": 16,
            "d_state": 8,
            "dropout": 0.0,
            "checkpoint_dir": str(checkpoint_dir),
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
            "plot_file": str(tmp_path / "out" / "preds-vs-targets.pdf"),
            "ylabel": "Prevalence",
        }
    )


@pytest.fixture
def checkpoint_dir(tmp_path):
    model = Mamba2Regressor(
        input_dim=get_input_size(use_cyclical_time=True),
        d_model=16,
        n_layers=1,
        d_state=8,
        d_conv=2,
        expand=2,
        head_dim=8,
        chunk_size=8,
        output_dim=1,
        dropout=0.0,
        rngs=nnx.Rngs(0),
    )
    ckpt_dir = tmp_path / "ckpts"
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        ckptr.save_checkpointables(0, {"model": nnx.state(model)}, overwrite=True)
    return ckpt_dir


@pytest.mark.slow
@pytest.mark.parametrize("predictor", ["prevalence", "cases"])
def test_visualise_predictions_main_writes_plot(tmp_path, script_sample_df, checkpoint_dir, predictor):
    data_file = tmp_path / "data.parquet"
    duckdb.sql("SELECT * FROM script_sample_df").write_parquet(str(data_file))
    (tmp_path / "out").mkdir()
    cfg = _viz_cfg(tmp_path, data_file, checkpoint_dir, predictor=predictor)

    visualise_predictions.main(cfg)

    assert (tmp_path / "out" / "preds-vs-targets.pdf").exists()


def test_visualise_predictions_main_missing_checkpoint_raises(tmp_path, script_sample_df):
    data_file = tmp_path / "data.parquet"
    duckdb.sql("SELECT * FROM script_sample_df").write_parquet(str(data_file))
    (tmp_path / "out").mkdir()
    cfg = _viz_cfg(tmp_path, data_file, tmp_path / "no-such-checkpoint")

    with pytest.raises(Exception, match="no checkpoints were found"):
        visualise_predictions.main(cfg)
