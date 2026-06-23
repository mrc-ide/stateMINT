import duckdb
import pandas as pd

from stateMINT import filter_raw_data
from stateMINT.data import STATIC_COVARS


def _make_db(db_path, n_days=400, n_params=1):
    base = 2190  # 6 * 365 (BURNIN_DAY)
    rows = [
        {
            "parameter_index": p,
            "simulation_index": 0,
            "global_index": p,
            "timesteps": base - 50 + day,
            "n_detect_lm_0_1825": 5.0,
            "n_age_0_1825": 10.0,
            "n_inc_clinical_0_36500": 3.0,
            "n_age_0_36500": 100.0,
            **{c: 0.5 for c in STATIC_COVARS},
        }
        for p in range(n_params)
        for day in range(n_days)
    ]
    con = duckdb.connect(str(db_path))
    con.register("rows_df", pd.DataFrame(rows))
    con.execute("CREATE TABLE simulation_results AS SELECT * FROM rows_df")
    con.close()


def test_main_parses_args_and_writes_output(tmp_path, monkeypatch):
    db = tmp_path / "sim.duckdb"
    _make_db(db)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_raw_data",
            "--db-path",
            str(db),
            "--predictor",
            "prevalence",
            "--window-size",
            "14",
            "--output-folder",
            str(out_dir),
        ],
    )

    filter_raw_data.main()

    out_file = out_dir / "filtered_data_prevalence.parquet"
    assert out_file.exists()
    out = duckdb.read_parquet(str(out_file)).df()
    assert not out.empty
    assert set(STATIC_COVARS).issubset(out.columns)


def test_main_uses_defaults_when_optional_args_omitted(tmp_path, monkeypatch):
    db = tmp_path / "sim.duckdb"
    _make_db(db)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["filter_raw_data", "--db-path", str(db)])

    filter_raw_data.main()

    out = duckdb.read_parquet(str(tmp_path / "filtered_data_prevalence.parquet")).df()
    assert not out.empty


def test_main_param_limit_flag_propagates(tmp_path, monkeypatch):
    db = tmp_path / "sim.duckdb"
    _make_db(db, n_params=2)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_raw_data",
            "--db-path",
            str(db),
            "--param-limit",
            "1",
            "--predictor",
            "prevalence",
            "--output-folder",
            str(out_dir),
        ],
    )

    filter_raw_data.main()

    out = duckdb.read_parquet(str(out_dir / "filtered_data_prevalence.parquet")).df()
    assert (out["parameter_index"] < 1).all()
