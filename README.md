# StateMINT

StateMINT is a JAX/Flax neural emulator for
[`malariasimulation`](https://github.com/mrc-ide/malariasimulation) outputs. It
uses a Mamba2 state-space sequence model to predict malaria trajectories from
static scenario covariates and intervention timing features. It supersedes the
earlier
[`MINTelligence`](https://github.com/CosmoNaught/MINTelligence) RNN emulator.

## What StateMINT Provides

- Mamba2-based sequence regressors for malaria prevalence and case-count
  trajectories.
- Data extraction utilities for aggregating raw `malariasimulation` DuckDB
  outputs into model-ready parquet files.
- Preprocessing with target transforms, covariate scaling, and
  intervention-aware feature construction.
- Training, evaluation, visualization, checkpointing, and export workflows.
- Hugging Face Hub loading utilities for exported inference artifacts.

## Installation

StateMINT requires Python 3.12 or newer and uses
[`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/mrc-ide/stateMINT.git
cd stateMINT
uv sync
```

For development dependencies and optional extras:

```bash
uv sync --all-extras --dev
```

Or install extras individually:

```bash
uv sync --extra plot
uv sync --extra gpu
```

## Quick Start: Inference

Load an exported artifact from the Hugging Face Hub or a local directory with
`Mamba2Regressor.from_pretrained`.

```python
from stateMINT.model import Mamba2Regressor

artifact = Mamba2Regressor.from_pretrained(
    "dide-ic/stateMINT",
    predictor="prevalence",
    revision="v1.0.0",
)

static_covars = [{
    "eir": 50.0,
    "dn0_use": 0.3,
    "dn0_future": 0.4,
    "Q0": 0.8,
    "phi_bednets": 0.7,
    "seasonal": 1.0,
    "routine": 0.5,
    "itn_use": 0.2,
    "irs_use": 0.1,
    "itn_future": 0.3,
    "irs_future": 0.2,
    "lsm": 0.0,
}]

predicted_prevalence = artifact.predict(static_covars)

print(predicted_prevalence.shape)  # (batch, timesteps)
print(predicted_prevalence[0])     # first trajectory
```

For cases, load the cases artifact and use the same input format:

```python
artifact = Mamba2Regressor.from_pretrained(
    "dide-ic/stateMINT",
    predictor="cases",
    revision="v1.0.0",
)

predicted_cases = artifact.predict(static_covars)
```

By default, predictions are returned on the original target scale: prevalence as
probabilities and cases on the scale used by the training data. Pass
`transformed=True` to return model-space outputs.

```python
raw_model_space = artifact.predict(static_covars, transformed=True)
```

For local artifacts, pass the target artifact directory:

```python
artifact = Mamba2Regressor.from_pretrained(
    "artifacts/prevalence",
    predictor="prevalence",
)
```

## Static Covariates

Inference inputs need one dictionary per scenario with these static covariates:

```text
eir
dn0_use
dn0_future
Q0
phi_bednets
seasonal
routine
itn_use
irs_use
itn_future
irs_future
lsm
```

Artifacts include the fitted static scaler, timestep grid, intervention day,
target transform, and other preprocessing metadata needed for inference.

## Training Workflow

Typical workflow:

1. Fetch and aggregate simulation data from DuckDB.
2. Train a target-specific model.
3. Evaluate or visualize test-set predictions.
4. Export the checkpoint into a portable inference artifact.
5. Upload the artifact to the Hugging Face Hub, if needed.

### 1. Fetch Filtered Data

`stateMINT.filter_raw_data` reads raw DuckDB simulation rows, filters burn-in,
aggregates fixed windows, and writes `filtered_data_<predictor>.parquet`.

```bash
uv run python -m stateMINT.filter_raw_data \
  --db-path /path/to/simulations.duckdb \
  --table-name simulation_results \
  --predictor prevalence \
  --window-size 14 \
  --output-folder data
```

Useful options:

- `--predictor prevalence` or `--predictor cases`
- `--param-limit N` to keep only the first `N` parameter indices
- `--sim-limit N` to sample up to `N` simulations per parameter

The raw table should include identifiers (`parameter_index`, `simulation_index`,
`global_index`), daily timesteps, the static covariates above, and output
columns for prevalence or cases.

### 2. Train a Model

Training uses Hydra; the default target is prevalence.

```bash
uv run python -m stateMINT.train
```

Train the cases model:

```bash
uv run python -m stateMINT.train target=cases
```

Common overrides:

```bash
uv run python -m stateMINT.train \
  target=prevalence \
  data_file=data/filtered_data_prevalence.parquet \
  output_dir=train_outputs/prevalence \
  use_wandb=false
```

Training writes checkpoints under `checkpoint_dir`, saves
`static_scaler.pkl` in `output_dir`, and reuses a split assignment file for
consistent train/validation/test splits.

### 3. W&B Sweeps

Sweep definitions live in `stateMINT/conf/sweeps`. Create a sweep, then run one
or more agents with the sweep ID returned by W&B:

```bash
uv run wandb sweep stateMINT/conf/sweeps/prevalence.yaml
uv run wandb agent <entity>/stateMINT-sweep/<sweep-id>
```

Use `stateMINT/conf/sweeps/cases.yaml` for the cases target. Sweep commands set
`use_wandb=true` and pass Hydra overrides through `${args_no_hyphens}`.

### 4. Visualize Predictions

Compare predictions with test-set targets. `checkpoint_dir` is required.

```bash
uv run python -m stateMINT.visualise_predictions \
  target=prevalence \
  checkpoint_dir=train_outputs/prevalence/ckpts-YYYY-MM-DDTHH:MM:SS \
  data_file=data/filtered_data_prevalence.parquet
```

The default output path is `viz_outputs/<predictor>/preds-vs-targets.pdf`.

### 5. Export an Artifact

Export converts a trained Orbax checkpoint and preprocessing metadata into a
self-contained artifact.

```bash
uv run python -m stateMINT.model_export \
  predictor=prevalence \
  checkpoint_dir=train_outputs/prevalence/ckpts-YYYY-MM-DDTHH:MM:SS \
  scaler_file=train_outputs/prevalence/static_scaler.pkl \
  artifact_dir=artifacts/prevalence
```

Export config architecture values must match the checkpoint, including
`d_model`, `d_state`, `n_layers`, `dropout`, and related Mamba2 settings.

An exported artifact contains:

```text
artifact_dir/
|-- checkpoint/
|-- model_config.json
`-- preprocessing_config.json
```

`model_config.json` stores architecture settings; `preprocessing_config.json`
stores feature order, target transform, intervention timing, timestep
construction, and static scaler parameters.

### 6. Upload To Hugging Face

Authenticate first:

```bash
hf auth login
```

Upload an artifact:

```bash
hf upload dide-ic/stateMINT artifacts/prevalence prevalence/ \
  --commit-message "Add prevalence model artifact"
```

Create a release tag:

```bash
hf repos tag create dide-ic/stateMINT v1.0.0 \
  --revision main \
  --message "Release v1.0.0"
```

## Configuration

Main Hydra configs in `stateMINT/conf`:

- `train_config.yaml` for training.
- `viz_config.yaml` for prediction visualizations.
- `export_config.yaml` for artifact export.
- `target/prevalence.yaml` and `target/cases.yaml` for target-specific defaults.
- `sweeps/*.yaml` for Weights & Biases sweep definitions.

Select a target with `target=prevalence` or `target=cases`; override config
values from the command line with Hydra syntax.

## Development

Run the test suite:

```bash
uv run pytest tests/
```

Skip slow or local-only tests:

```bash
uv run pytest tests/ -m "not slow"
uv run pytest tests/ -m "not local"
```

Run linting and formatting:

```bash
uv run ruff check
uv run ruff format
```

## Repository Layout

```text
stateMINT/
|-- common/              # shared dataclasses, transforms, and model helpers
|-- conf/                # Hydra configs for training, export, viz, and sweeps
|-- data/                # DuckDB fetch, preprocessing, features, and loaders
|-- eval/                # metrics and prediction/target visualization helpers
|-- model/               # Mamba2 regressor and artifact loading utilities
|-- training/            # optimizer, train/eval steps, loss, and checkpointing
|-- filter_raw_data.py   # CLI for building filtered parquet datasets
|-- train.py             # Hydra training entry point
|-- visualise_predictions.py
`-- model_export.py      # Hydra artifact export entry point

tests/                   # unit tests
artifacts/               # exported model artifact examples/metadata
viz_outputs/             # generated prediction visualization outputs
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and the
development workflow.
