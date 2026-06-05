import logging
import time
import duckdb
from pathlib import Path
from typing import Literal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_fetched_filtered_data(
    db_path: str,
    table_name: str,
    param_limit: int | None = None,
    sim_limit: int | None = None,
    window_size: int = 14,
    predictor: Literal["prevalence", "cases"] = "prevalence",
    output_folder: str = ".",
) -> None:
    """Fetch data from DuckDB with filtering and aggregation.

    The function fetches data from the specified DuckDB database and table. It applies the following to data:
    - filters simulations based on `param_limit` and `sim_limit`
    - filters out first 6 years. (epidemiological burn-in period). only last 6 years are kept (2190-4379 days)
    - aggregates into windows of `window_size` days (default 14) by parameter/simulation, computing either prevalence or cases as the target.

    """

    if predictor not in ["prevalence", "cases"]:
        raise ValueError(f"Unsupported predictor: {predictor}")
    log.info(f"Connecting to DuckDB: {db_path}")
    t0 = time.perf_counter()

    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA memory_limit='32GB';")
    con.execute("PRAGMA threads=16;")

    param_where = f"WHERE parameter_index < {param_limit}" if param_limit else ""

    distinct_sims = f"""
        SELECT DISTINCT parameter_index, simulation_index, global_index
        FROM {table_name}
        {param_where}
    """

    if sim_limit:
        sampled_sims = f"""
            SELECT parameter_index, simulation_index, global_index
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY parameter_index ORDER BY RANDOM()
                ) AS rn
                FROM ({distinct_sims})
            )
            WHERE rn <= {sim_limit}
        """
    else:
        sampled_sims = distinct_sims

    year6_day = 6 * 365

    if predictor == "prevalence":
        query = f"""
            WITH cte AS (
                SELECT
                    t.parameter_index, t.simulation_index, t.global_index,
                    t.timesteps AS abs_timesteps,
                    -- carry raw counts so we can do ratio-of-sums inside the window
                    CAST(t.n_detect_lm_0_1825 AS DOUBLE) AS n_detect_lm_0_1825,
                    CAST(t.n_age_0_1825 AS DOUBLE) AS n_age_0_1825,
                    t.eir, t.dn0_use, t.dn0_future, t.Q0, t.phi_bednets,
                    t.seasonal, t.routine, t.itn_use, t.irs_use,
                    t.itn_future, t.irs_future, t.lsm
                FROM {table_name} t
                JOIN ({sampled_sims}) rs USING (parameter_index, simulation_index)
            ),
            grouped AS (
                SELECT
                    parameter_index, simulation_index, global_index,
                    FLOOR((abs_timesteps - {year6_day}) / {window_size}) AS group_id,
                    -- prevalence over the window as ratio of sums across days
                    SUM(n_detect_lm_0_1825) / NULLIF(SUM(n_age_0_1825), 0) AS prevalence,
                    MIN(abs_timesteps) AS abs_timesteps,
                    MAX(eir) AS eir, MAX(dn0_use) AS dn0_use,
                    MAX(dn0_future) AS dn0_future, MAX(Q0) AS Q0,
                    MAX(phi_bednets) AS phi_bednets, MAX(seasonal) AS seasonal,
                    MAX(routine) AS routine, MAX(itn_use) AS itn_use,
                    MAX(irs_use) AS irs_use, MAX(itn_future) AS itn_future,
                    MAX(irs_future) AS irs_future, MAX(lsm) AS lsm
                FROM cte
                WHERE abs_timesteps >= {year6_day}
                GROUP BY 1, 2, 3, 4
            )
            SELECT
                parameter_index, simulation_index, global_index,
                ROW_NUMBER() OVER (
                    PARTITION BY parameter_index, simulation_index ORDER BY group_id
                ) AS timesteps,
                abs_timesteps, prevalence,
                eir, dn0_use, dn0_future, Q0, phi_bednets,
                seasonal, routine, itn_use, irs_use, itn_future, irs_future, lsm
            FROM grouped
            WHERE prevalence IS NOT NULL
            ORDER BY parameter_index, simulation_index, group_id
        """
    else:
        query = f"""
            WITH cte AS (
                SELECT
                    t.parameter_index, t.simulation_index, t.global_index,
                    t.timesteps AS abs_timesteps,
                    t.n_inc_clinical_0_36500, t.n_age_0_36500,
                    t.eir, t.dn0_use, t.dn0_future, t.Q0, t.phi_bednets,
                    t.seasonal, t.routine, t.itn_use, t.irs_use,
                    t.itn_future, t.irs_future, t.lsm
                FROM {table_name} t
                JOIN ({sampled_sims}) rs USING (parameter_index, simulation_index)
            ),
            grouped AS (
                SELECT
                    parameter_index, simulation_index, global_index,
                    FLOOR((abs_timesteps - {year6_day}) / {window_size}) AS group_id,
                    -- per-day rate per 1000 across the window (denominator-weighted):
                    1000.0 * SUM(n_inc_clinical_0_36500) / NULLIF(SUM(n_age_0_36500), 0) AS cases,
                    SUM(n_age_0_36500) AS exposure_pd,
                    MIN(abs_timesteps) AS abs_timesteps,
                    MAX(eir) AS eir, MAX(dn0_use) AS dn0_use,
                    MAX(dn0_future) AS dn0_future, MAX(Q0) AS Q0,
                    MAX(phi_bednets) AS phi_bednets, MAX(seasonal) AS seasonal,
                    MAX(routine) AS routine, MAX(itn_use) AS itn_use,
                    MAX(irs_use) AS irs_use, MAX(itn_future) AS itn_future,
                    MAX(irs_future) AS irs_future, MAX(lsm) AS lsm
                FROM cte
                WHERE abs_timesteps >= {year6_day}
                GROUP BY 1, 2, 3, 4
            )
            SELECT
                parameter_index, simulation_index, global_index,
                ROW_NUMBER() OVER (
                    PARTITION BY parameter_index, simulation_index ORDER BY group_id
                ) AS timesteps,
                abs_timesteps, cases, exposure_pd,
                eir, dn0_use, dn0_future, Q0, phi_bednets,
                seasonal, routine, itn_use, irs_use, itn_future, irs_future, lsm
            FROM grouped
            WHERE cases IS NOT NULL
            ORDER BY parameter_index, simulation_index, group_id
        """

    out_path = Path(output_folder) / f"filtered_data_{predictor}.parquet"
    con.execute(
        f"""
        COPY ({query})
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    con.close()
    log.info(f"Data fetched and saved in {time.perf_counter() - t0:.2f} seconds: {out_path}")
