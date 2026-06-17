# StateMINT - Malaria Simulation Emulator

StateMINT is a state space based neural network emulator for [malariasimulation](https://github.com/mrc-ide/malariasimulation). This repository supersedes
the old [RNN emulator](https://github.com/CosmoNaught/MINTelligence).


## Dataset Creation and Usage
** Creating the dataset **
The dataset is created by running the `filter_raw_data.py` script, which fetches and filters the raw simulation data from a DuckDB database.

The filtered data is saved as a PARQUET file in the specified output folder, with the name format `filtered_data_{predictor}.parquet` (e.g., `filtered_data_prevalence.parquet`).

** Using the dataset **

The filtered dataset can be loaded with duckdb into a pandas DataFrame for further analysis and model training. For example:
```python
import duckdb
df = duckdb.query("SELECT * FROM 'path/to/filtered_data_prevalence.parquet'").to_df()
```


## Exporting the Model
A model checkpoint can be exported to a HuggingFace repository using the `export_model.py` script. The hydraconfig is stored in `stateMINT/configs/export_model_config.yaml`. By default the exported model will be saved in the `artifacts` folder. You can change the output folder by modifying the `artifact_dir` parameter in the config file.

To upload to hugging face you need to have a huggingface account and be logged in using the `hf auth login` command. Ensure you are part of `dide-ic` organization on huggingface. You can then run the following command upload your exported model artifact to the huggingface hub:

```bash
hf upload dide-ic/stateMINT <LOCAL_PATH> <PATH_IN_HUB> --commit-message "<COMMIT_MESSAGE>"

####### examples ###########

# base
hf upload dide-ic/stateMINT artifacts/prevalence prevalence/ --commit-message "Add prevalence model artifact"

# with PR
hf upload dide-ic/stateMINT artifacts/prevalence prevalence/ --commit-message "Add prevalence model artifact" --create-pr
```
