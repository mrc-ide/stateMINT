import logging
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..common.dataclasses import Predictor
from ..common.utils import transform_targets_np
from .features import (
    AFTER_INTERVENTION_COVARS,
    MODEL_START_DAY,
    INTERVENTION_DAY,
    STATIC_COVARS,
    StandardScaler,
    get_input_size,
)

log = logging.getLogger(__name__)

# Precomputed column indices for AFTER_INTERVENTION_COVARS within STATIC_COVARS.
_AFTER_INTERVENTION_COL_INDICES = np.array(
    [STATIC_COVARS.index(c) for c in AFTER_INTERVENTION_COVARS if c in STATIC_COVARS], dtype=np.intp
)


@dataclass
class PreparedData:
    train_data: list
    val_data: list
    test_data: list
    input_size: int
    scaler: StandardScaler
    train_param_sims: set[tuple[int, int]] = field(default_factory=set)
    val_param_sims: set[tuple[int, int]] = field(default_factory=set)
    test_param_sims: set[tuple[int, int]] = field(default_factory=set)


def prepare_data(df: pd.DataFrame, cfg: DictConfig):
    """
    Split and transform raw simulation data.

    Filters out low-signal parameter-simulation pairs, creates or loads the
    train/val/test split, fits static covariate scaling on the train split only,
    and builds per-sequence records for each split.

    Note: each malariasimulation run covers TOTAL_DAYS days: a MODEL_START_DAY's warmup followed by
    TOTAL_DAYS - MODEL_START_DAY days of actual simulation. Only the latter are used here; the
    warmup has already been discarded in the input `df` parameter.
    The intervention is applied at INTERVENTION_DAY.

    Args:
        df: Raw simulation dataframe.
        cfg: Data preparation config.

    Returns:
        Prepared train, validation, and test data.
    """
    random.seed(cfg.seed)

    # Filter by threshold
    threshold = cfg.min_prevalence if cfg.predictor == "prevalence" else cfg.min_cases
    df = _filter_by_threshold(df, cfg.predictor, threshold)

    # split data
    if cfg.use_existing_split and Path(cfg.split_file).exists():
        log.info(f"Loading existing split from {cfg.split_file}")
        train_ps, val_ps, test_ps = _load_split(cfg.split_file, df)
    else:
        log.info("Creating new train/val/test split (70/15/15)")
        train_ps, val_ps, test_ps = _create_split(df, cfg.seed)
        if cfg.split_file:
            log.info(f"Saving split to {cfg.split_file}")
            _save_split(cfg.split_file, train_ps, val_ps, test_ps, df)

    log.info(f"Split — train: {len(train_ps)}, val: {len(val_ps)}, test: {len(test_ps)}")

    # Scaler fitted on train data only
    scaler = _fit_scaler(df, train_ps, cfg.output_dir)

    # Build data
    train_data = _build_data(df, train_ps, scaler, cfg)
    val_data = _build_data(df, val_ps, scaler, cfg)
    test_data = _build_data(df, test_ps, scaler, cfg)

    return PreparedData(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_size=get_input_size(cfg.use_cyclical_time),
        scaler=scaler,
        train_param_sims=train_ps,
        val_param_sims=val_ps,
        test_param_sims=test_ps,
    )


def build_feature_matrix(
    base_static: np.ndarray,
    abs_t: np.ndarray,
    t: np.ndarray,
    scaler: StandardScaler,
    *,
    intervention_day: int = INTERVENTION_DAY,
    after_intervention_indices: np.ndarray = _AFTER_INTERVENTION_COL_INDICES,
    use_cyclical_time: bool = True,
) -> np.ndarray:
    """
    Assemble the (T, INPUT_SIZE) model input from raw static covars + timesteps.

    Shared by training (per parameter-simulation row) and inference (per user input).

    Args:
        base_static: Raw static covariate vector of shape (len(STATIC_COVARS),).
        abs_t: Absolute timestep values (float32, shape T).
        t: Relative timestep values (float32, shape T).
        scaler: Fitted static covariate scaler.
        intervention_day: Absolute day the intervention switches on.
        after_intervention_indices: Column indices to mask before intervention_day.

    Returns:
        Feature matrix of shape (T, INPUT_SIZE).
    """
    scaled_static = _build_static_features(base_static, abs_t, scaler, intervention_day, after_intervention_indices)
    post_intervention, t_since_intervention_yrs = _build_intervention_features(abs_t, intervention_day)
    time_feats = _build_time_features(abs_t, t, use_cyclical_time=use_cyclical_time)

    return np.concatenate(
        [time_feats, scaled_static, post_intervention[:, None], t_since_intervention_yrs[:, None]], axis=1
    )


def build_timestep_grid(window_size: int, n_steps: int, model_start_day: int = MODEL_START_DAY) -> tuple[np.ndarray, np.ndarray]:
    """
    Regenerate the (abs_t, t) grid the fetch/windowing step produces.

    Mirrors fetch.py: group_id = floor((abs - burnin) / window_size), and per group
    abs_timesteps = min(abs) = model_start_day + group_id * window_size, with timesteps the
    1-based row number.

    Args:
        window_size: Days aggregated per timestep.
        n_steps: Number of timesteps (sequence length).
        model_start_day: Absolute day the kept window starts.

    Returns:
        abs_t: Absolute timesteps (float32, shape n_steps).
        t: Relative timesteps 1..n_steps (float32, shape n_steps).
    """
    abs_t = np.arange(model_start_day, model_start_day + window_size * n_steps, window_size, dtype=np.float32)
    t = np.arange(1, n_steps + 1, dtype=np.float32)
    return abs_t, t


def build_inference_inputs(
    static_covars: list[dict[str, float]],
    scaler: StandardScaler,
    preprocessing_config: dict,
) -> np.ndarray:
    """
    Build a batched model input from raw static covariate dicts.

    The user supplies only the static covariates; the timestep grid, intervention
    masking, and scaling are reconstructed from preprocessing_config (the exported
    sidecar), so this exactly matches train-time preprocessing.

    Args:
        static_covars: One dict per series, keyed by STATIC_COVARS names.
        scaler: Fitted static covariate scaler.
        preprocessing_config: Exported preprocessing_config.json contents.

    Returns:
        Model input of shape (B, T, INPUT_SIZE), float32.
    """
    static_names = preprocessing_config["static_covars"]
    after_intervention_indices = np.array(
        [static_names.index(c) for c in preprocessing_config["after_intervention"]], dtype=np.intp
    )

    abs_t, t = build_timestep_grid(
        preprocessing_config["window_size"], preprocessing_config["n_steps"], preprocessing_config["model_start_day"]
    )
    batch = []
    for covars in static_covars:
        missing = [c for c in static_names if c not in covars]
        if missing:
            raise ValueError(f"Missing static covariates: {missing}")
        base_static = np.array([covars[name] for name in static_names], dtype=np.float32)
        batch.append(
            build_feature_matrix(
                base_static,
                abs_t,
                t,
                scaler,
                intervention_day=preprocessing_config["intervention_day"],
                after_intervention_indices=after_intervention_indices,
                use_cyclical_time=preprocessing_config["use_cyclical_time"],
            )
        )
    return np.stack(batch, dtype=np.float32)  # (B, T, INPUT_SIZE)


# -------------- internal helpers ----------------------------------------------------------


def _filter_by_threshold(df: pd.DataFrame, target_col: str, threshold: float) -> pd.DataFrame:
    """
    Filter parameter-simulation pairs where the mean target value is below the threshold.

    Args:
        df: Input dataframe.
        target_col: Target column name.
        threshold: Minimum mean target value.

    Returns:
        Filtered dataframe.
    """

    group_means = df.groupby(["parameter_index", "simulation_index"])[target_col].mean()
    valid = set(map(tuple, group_means[group_means >= threshold].index.tolist()))
    df["_ps"] = list(zip(df["parameter_index"], df["simulation_index"]))

    log.info(
        f"Filtering with threshold {threshold} on {target_col}: {len(valid)} valid parameter-simulation pairs out of {len(group_means)}"
    )

    return df[df["_ps"].isin(valid)]


def _load_split(
    split_file: str, df: pd.DataFrame
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Load an existing train/val/test split.

    Args:
        split_file: Split CSV path.
        df: Filtered dataframe.

    Returns:
        Train, validation, and test parameter-simulation sets.
    """
    split_df = pd.read_csv(split_file)
    present = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))
    train_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "train"].itertuples()
    } & present
    val_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "validate"].itertuples()
    } & present
    test_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "test"].itertuples()
    } & present
    return train_ps, val_ps, test_ps  # type: ignore


def _create_split(
    df: pd.DataFrame, seed: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Create a train/val/test split by parameter.

    Args:
        df: Filtered dataframe.
        seed: Shuffle seed.

    Returns:
        Train, validation, and test parameter-simulation sets.
    """
    random.seed(seed)  # TODO: check seeds set correctly!!
    params = list(df["parameter_index"].unique())
    random.shuffle(params)
    n = len(params)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_p = set(params[:n_train])
    val_p = set(params[n_train : n_train + n_val])
    test_p = set(params[n_train + n_val :])
    all_ps = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))
    return (
        {ps for ps in all_ps if ps[0] in train_p},
        {ps for ps in all_ps if ps[0] in val_p},
        {ps for ps in all_ps if ps[0] in test_p},
    )


def _save_split(path, train_ps, val_ps, test_ps, df):
    """
    Save parameter-simulation split assignments.

    Args:
        path: Output CSV path.
        train_ps: Training pairs.
        val_ps: Validation pairs.
        test_ps: Test pairs.
        df: Source dataframe.

    Returns:
        None.
    """
    rows = []
    ps_to_global = {
        (r.parameter_index, r.simulation_index): r.global_index
        for r in df[["parameter_index", "simulation_index", "global_index"]].drop_duplicates().itertuples()
    }
    for ps, split in (
        [(p, "train") for p in train_ps] + [(p, "validate") for p in val_ps] + [(p, "test") for p in test_ps]
    ):
        rows.append(
            {"parameter_index": ps[0], "simulation_index": ps[1], "global_index": ps_to_global.get(ps), "split": split}
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Split saved to {path}")


def _fit_scaler(df: pd.DataFrame, train_ps: set[tuple[int, int]], output_dir: str) -> StandardScaler:
    """
    Fit and save the static covariate scaler.

    Args:
        df: Filtered dataframe.
        train_ps: Training pairs.
        output_dir: Directory for scaler output.

    Returns:
        Fitted scaler.
    """
    train_mask = df["_ps"].isin(train_ps)
    train_static = (
        df.loc[train_mask, ["_ps"] + STATIC_COVARS]
        .drop_duplicates(subset=["_ps"])[STATIC_COVARS]
        .astype(np.float32)
        .values
    )
    scaler = StandardScaler()
    scaler.fit(train_static)

    save_path = Path(output_dir) / "static_scaler.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler


def _build_static_features(
    base_static: np.ndarray,
    abs_t: np.ndarray,
    scaler: StandardScaler,
    intervention_day: int = INTERVENTION_DAY,
    after_intervention_indices: np.ndarray = _AFTER_INTERVENTION_COL_INDICES,
) -> np.ndarray:
    """
    Build scaled static covariate matrix with pre-intervention masking.

    AFTER_INTERVENTION_COVARS are zeroed before intervention_day because those parameters
    aren't active yet.

    Args:
        base_static: Raw static covariate vector of shape (len(STATIC_COVARS),).
        abs_t: Absolute timestep values (float32, shape T).
        scaler: Fitted static covariate scaler.
        intervention_day: Absolute day the intervention switches on.
        after_intervention_indices: Column indices to mask before intervention_day.

    Returns:
        Scaled static feature matrix of shape (T, len(STATIC_COVARS)).
    """
    masked_static = base_static.copy()
    masked_static[after_intervention_indices] = 0.0
    raw_matrix = np.where((abs_t < intervention_day)[:, None], masked_static, base_static)
    return scaler.transform(raw_matrix)


def _build_intervention_features(
    abs_t: np.ndarray, intervention_day: int = INTERVENTION_DAY
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute post-intervention flag and time-since-intervention in years.

    Args:
        abs_t: Absolute timestep values (float32, shape T).
        intervention_day: Absolute day the intervention switches on.

    Returns:
        post_intervention: Binary flag, 1 on/after intervention_day (shape T).
        t_since_intervention_yrs: Years elapsed since intervention_day, 0 before (shape T).
    """
    post_intervention = (abs_t >= intervention_day).astype(np.float32)
    t_since_intervention_yrs = (np.maximum(0.0, abs_t - intervention_day) / 365.0).astype(np.float32)
    return post_intervention, t_since_intervention_yrs


def _build_time_features(abs_t: np.ndarray, t: np.ndarray, use_cyclical_time: bool) -> np.ndarray:
    """
    Build time feature columns — either cyclical (sin/cos of day-of-year) or
    min-max normalised absolute time.

    Args:
        abs_t: Absolute timestep values (float32, shape T).
        t: Relative timestep values (float32, shape T).
        use_cyclical_time: Whether to use cyclical encoding.

    Returns:
        Time feature matrix of shape (T, 2) for cyclical or (T, 1) for linear.
    """
    if use_cyclical_time:
        doy = abs_t % 365.0
        sin_t = np.sin(2 * math.pi * doy / 365.0).astype(np.float32)
        cos_t = np.cos(2 * math.pi * doy / 365.0).astype(np.float32)
        return np.stack([sin_t, cos_t], axis=1)
    else:
        t_min, t_max = t.min(), t.max()
        t_norm = ((t - t_min) / (t_max - t_min) if t_max > t_min else t).astype(np.float32)
        return t_norm[:, None]


def _build_targets(sub: pd.DataFrame, predictor: Predictor, eps_prevalence: float) -> np.ndarray:
    """
    Extract and transform target values.

    Args:
        sub: Rows for one parameter-simulation pair.
        predictor: Target column name.
        eps_prevalence: Small offset for prevalence log-transform.

    Returns:
        Transformed target array of shape (T,).
    """
    Y_raw = np.asarray(sub[predictor].values, dtype=np.float32)
    return transform_targets_np(Y_raw, predictor, eps_prevalence)


def _build_weights(sub: pd.DataFrame, predictor: str) -> np.ndarray:
    """
    Build per-timestep loss weights.

    For case prediction, weights are exposure counts. For prevalence, all weights are 1.

    Args:
        sub: Rows for one parameter-simulation pair.
        predictor: Target column name.

    Returns:
        Weight array of shape (T,).
    """
    if predictor == "cases":
        return np.asarray(sub["exposure_pd"].values, dtype=np.float32)
    return np.ones(len(sub), dtype=np.float32)


def _build_data(
    df: pd.DataFrame,
    param_sims: set[tuple[int, int]],
    scaler: StandardScaler,
    cfg: DictConfig,
) -> list[dict[str, np.ndarray]]:
    """
    Build model-ready sequence records.

    Each record contains time features, scaled static covariates, transformed targets,
    and optional exposure weights for case prediction.

    Args:
        df: Filtered dataframe.
        param_sims: Pairs to include.
        scaler: Static covariate scaler.
        cfg: Data preparation config.

    Returns:
        List of sequence records with keys x (T, input_size), y (T,), w (T,), ps (2,).
    """
    groups = df.groupby(["parameter_index", "simulation_index"])
    data = []

    for ps in param_sims:
        if ps not in groups.groups:
            continue
        sub = groups.get_group(ps).sort_values("timesteps")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[cfg.predictor])
        if len(sub) == 0:
            continue

        abs_t = np.asarray(sub["abs_timesteps"].values, dtype=np.float32)
        t = np.asarray(sub["timesteps"].values, dtype=np.float32)
        base_static = np.asarray(sub.iloc[0][STATIC_COVARS].values, dtype=np.float32)

        X = build_feature_matrix(base_static, abs_t, t, scaler, use_cyclical_time=cfg.use_cyclical_time)
        Y = _build_targets(sub, cfg.predictor, cfg.eps_prevalence)
        W = _build_weights(sub, cfg.predictor)

        data.append(
            {
                "x": X,  # (T, input_size)
                "y": Y,  # (T,)
                "w": W,  # (T,)
                "ps": np.asarray(ps, dtype=np.int32),  # (2,) parameter_index, simulation_index
            }
        )

    return data
