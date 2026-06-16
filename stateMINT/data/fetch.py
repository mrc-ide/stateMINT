import logging
import time
import duckdb
from pathlib import Path
from typing import Literal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


STATIC_COLUMNS = (
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
)


def _build_static_sql_parts() -> tuple[str, str, str]:
    """Return SQL fragments for static covariates across query stages."""

    cte_columns = ", ".join(f"t.{col}" for col in STATIC_COLUMNS)
    grouped_aggregates = ",\n                ".join(f"MAX({col}) AS {col}" for col in STATIC_COLUMNS)
    final_select = ", ".join(STATIC_COLUMNS)
    return cte_columns, grouped_aggregates, final_select


def _build_predictor_sql_parts(
    predictor: Literal["prevalence", "cases"],
) -> tuple[str, str, str, str]:
    """Return predictor-specific SQL fragments for CTE, aggregation, and output."""

    if predictor == "prevalence":
        cte_columns = ",\n                ".join(
            [
                "CAST(t.n_detect_lm_0_1825 AS DOUBLE) AS n_detect_lm_0_1825",
                "CAST(t.n_age_0_1825 AS DOUBLE) AS n_age_0_1825",
            ]
        )
        grouped_aggregates = "SUM(n_detect_lm_0_1825) / NULLIF(SUM(n_age_0_1825), 0) AS prevalence"
        final_select = "prevalence"
        non_null_col = "prevalence"
    else:
        cte_columns = ",\n                ".join(["t.n_inc_clinical_0_36500", "t.n_age_0_36500"])
        grouped_aggregates = ",\n                ".join(
            [
                "1000.0 * SUM(n_inc_clinical_0_36500) / NULLIF(SUM(n_age_0_36500), 0) AS cases",
                "SUM(n_age_0_36500) AS exposure_pd",
            ]
        )
        final_select = "cases, exposure_pd"
        non_null_col = "cases"

    return cte_columns, grouped_aggregates, final_select, non_null_col


def _build_distinct_sims_query(table_name: str, param_limit: int | None) -> str:
    """Build SQL for unique parameter/simulation rows with optional parameter cap."""

    param_where = ""
    if param_limit is not None:
        param_where = f"WHERE parameter_index < {param_limit}"

    return f"""
        SELECT DISTINCT parameter_index, simulation_index, global_index
        FROM {table_name}
        {param_where}
    """


def _build_sampled_sims_query(
    table_name: str,
    param_limit: int | None,
    sim_limit: int | None,
) -> str:
    """Build SQL that optionally samples simulations per parameter."""

    distinct_sims = _build_distinct_sims_query(table_name=table_name, param_limit=param_limit)

    if sim_limit is None:
        return distinct_sims

    return f"""
        SELECT parameter_index, simulation_index, global_index
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY parameter_index ORDER BY RANDOM()
            ) AS rn
            FROM ({distinct_sims})
        )
        WHERE rn <= {sim_limit}
    """


def _build_data(
    table_name: str,
    sampled_sims_query: str,
    window_size: int,
    predictor: Literal["prevalence", "cases"],
    burnin_day: int,
) -> str:
    """Build the final windowed feature/target SQL query."""

    static_cte_columns, static_grouped_aggregates, static_final_select = _build_static_sql_parts()
    predictor_cte_columns, predictor_grouped_aggregates, predictor_final_select, predictor_non_null_col = (
        _build_predictor_sql_parts(predictor)
    )

    return f"""
        WITH cte AS (
            SELECT
                t.parameter_index, t.simulation_index, t.global_index,
                t.timesteps AS abs_timesteps,
                {predictor_cte_columns},
                {static_cte_columns}
            FROM {table_name} t
            JOIN ({sampled_sims_query}) rs USING (parameter_index, simulation_index)
        ),
        grouped AS (
            SELECT
                parameter_index, simulation_index, global_index,
                FLOOR((abs_timesteps - {burnin_day}) / {window_size}) AS group_id,
                {predictor_grouped_aggregates},
                MIN(abs_timesteps) AS abs_timesteps,
                {static_grouped_aggregates}
            FROM cte
            WHERE abs_timesteps >= {burnin_day}
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            parameter_index, simulation_index, global_index,
            ROW_NUMBER() OVER (
                PARTITION BY parameter_index, simulation_index ORDER BY group_id
            ) AS timesteps,
            abs_timesteps, {predictor_final_select},
            {static_final_select}
        FROM grouped
        WHERE {predictor_non_null_col} IS NOT NULL
        ORDER BY parameter_index, simulation_index, group_id
    """


def save_fetched_filtered_data(
    db_path: str,
    table_name: str,
    param_limit: int | None = None,
    sim_limit: int | None = None,
    window_size: int = 14,
    predictor: Literal["prevalence", "cases"] = "prevalence",
    output_folder: str = ".",
) -> None:
    """Fetch, filter, aggregate, and write model-ready parquet data from DuckDB."""

    if predictor not in ["prevalence", "cases"]:
        raise ValueError(f"Unsupported predictor: {predictor}")
    log.info(f"Connecting to DuckDB: {db_path}")
    t0 = time.perf_counter()

    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA memory_limit='32GB';")
    con.execute("PRAGMA threads=16;")

    sampled_sims_query = _build_sampled_sims_query(
        table_name=table_name,
        param_limit=param_limit,
        sim_limit=sim_limit,
    )

    query = _build_data(
        table_name=table_name,
        sampled_sims_query=sampled_sims_query,
        window_size=window_size,
        predictor=predictor,
        burnin_day=6 * 365,
    )

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
