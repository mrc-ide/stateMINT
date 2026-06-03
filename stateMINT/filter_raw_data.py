def main():
    """
    example usage:
    python -m stateMINT.filter_raw_data \
        --db-path malaria_simulations_3.duckdb \
        --table-name simulation_results \
        --param-limit 1000 \
        --sim-limit 10 \
        --window-size 14 \
        --predictor prevalence \
        --output-folder ./data
    """

    import argparse
    from stateMINT.data.fetch import save_fetched_filtered_data

    parser = argparse.ArgumentParser(description="Fetch and filter simulation data from DuckDB.")
    parser.add_argument("--db-path", type=str, help="Path to the DuckDB database file.")
    parser.add_argument("--table-name", type=str, help="Name of the table to query.")
    parser.add_argument("--param-limit", type=int, default=None, help="Limit number of parameter indices.")
    parser.add_argument("--sim-limit", type=int, default=None, help="Max simulations per parameter.")
    parser.add_argument("--window-size", type=int, default=14, help="Aggregation window in days (default: 14).")
    parser.add_argument(
        "--predictor",
        choices=["prevalence", "cases"],
        default="prevalence",
        help="Target variable (default: prevalence).",
    )
    parser.add_argument(
        "--output-folder", type=str, default=".", help="Output folder for the CSV (default: current dir)."
    )
    args = parser.parse_args()

    save_fetched_filtered_data(
        db_path=args.db_path,
        table_name=args.table_name,
        param_limit=args.param_limit,
        sim_limit=args.sim_limit,
        window_size=args.window_size,
        predictor=args.predictor,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
