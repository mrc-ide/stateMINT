import duckdb
import numpy as np
import pytest

from stateMINT.data.fetch import save_fetched_filtered_data


def _read_parquet(path):
    return duckdb.read_parquet(str(path)).df()


STATIC = [
    "eir",
    "dn0_use",
    "dn0_future",
    "Q0",
    "phi_bednets",
    "seasonal",
    "routine",
    "itn_use",
    "irs_use",
    "itn_future",
    "irs_future",
    "lsm",
]


def _make_db(db_path, n_days=400):
    base = 2190  # 6 * 365
    rows = []
    for day in range(n_days):
        ts = base - 50 + day  # starts before burn-in cutoff, crosses it
        rows.append(
            {
                "parameter_index": 0,
                "simulation_index": 0,
                "global_index": 0,
                "timesteps": ts,
                "n_detect_lm_0_1825": 5.0,
                "n_age_0_1825": 10.0,
                "n_inc_clinical_0_36500": 3.0,
                "n_age_0_36500": 100.0,
                **{c: 0.5 for c in STATIC},
            }
        )
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE simulation_results AS SELECT * FROM df")
    con.close()


def test_invalid_predictor_raises(tmp_path):
    with pytest.raises(ValueError):
        save_fetched_filtered_data(str(tmp_path / "x.duckdb"), "t", predictor="bogus")


def test_prevalence_fetch_filters_burnin_and_windows(tmp_path):
    db = tmp_path / "sim.duckdb"
    _make_db(db)
    save_fetched_filtered_data(
        str(db), "simulation_results", window_size=14, predictor="prevalence", output_folder=str(tmp_path)
    )
    out = _read_parquet(tmp_path / "filtered_data_prevalence.parquet")

    # burn-in (< 2190) dropped, so all kept rows are at/after the cutoff
    assert out["abs_timesteps"].min() >= 2190
    # prevalence = ratio of sums = 5/10 within each full window
    np.testing.assert_allclose(out["prevalence"].values, 0.5)
    # timesteps re-indexed sequentially from 1 per sim
    assert out["timesteps"].min() == 1
    assert set(STATIC).issubset(out.columns)


def test_cases_fetch_outputs_exposure(tmp_path):
    db = tmp_path / "sim.duckdb"
    _make_db(db)
    save_fetched_filtered_data(
        str(db), "simulation_results", window_size=14, predictor="cases", output_folder=str(tmp_path)
    )
    out = _read_parquet(tmp_path / "filtered_data_cases.parquet")
    assert "cases" in out.columns
    assert "exposure_pd" in out.columns
    # rate per 1000 = 1000 * 3 / 100 = 30
    np.testing.assert_allclose(out["cases"].values, 30.0)


def test_param_and_sim_limits(tmp_path):
    db = tmp_path / "sim.duckdb"
    _make_db(db)
    # param_limit=0 keeps no parameters (parameter_index < 0 is empty)
    save_fetched_filtered_data(
        str(db), "simulation_results", param_limit=1, predictor="prevalence", output_folder=str(tmp_path)
    )
    out = _read_parquet(tmp_path / "filtered_data_prevalence.parquet")
    assert (out["parameter_index"] < 1).all()
