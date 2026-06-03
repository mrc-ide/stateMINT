# StateMINT - Malaria Simulation Emulator

StateMINT is a state space based neural network emulator for [malariasimulation](https://github.com/mrc-ide/malariasimulation). This repository supersedes
the old [RNN emulator](https://github.com/CosmoNaught/MINTelligence).


## Dataset Creation and Usage
** Creating the dataset **
The dataset is created by running the `filter_raw_data.py` script, which fetches and filters the raw simulation data from a DuckDB database. The script takes the following arguments:
- `--db-path`: Path to the DuckDB database file.
- `--table-name`: Name of the table to query.
- `--param-limit`: Limit number of parameter indices (optional).
- `--sim-limit`: Max simulations per parameter (optional).
- `--window-size`: Aggregation window in days (default: 14).
- `--predictor`: Target variable (default: prevalence).
- `--output-folder`: Output folder for the CSV (default: current dir).

The filtered data is saved as a PARQUET file in the specified output folder, with the name format `filtered_data_{predictor}.parquet` (e.g., `filtered_data_prevalence.parquet`).

** Using the dataset **

The filtered dataset can be loaded with duckdb into a pandas DataFrame for further analysis and model training. For example:
```python
import duckdb
df = duckdb.query("SELECT * FROM 'path/to/filtered_data_prevalence.parquet'").to_df()
```
