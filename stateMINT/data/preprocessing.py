import logging
import math
import random
from dataclasses import dataclass, field
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..common.utils import transform_targets_np

log = logging.getLogger(__name__)

STATIC_COVARS = [
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
AFTER9_COVARS = ["dn0_future", "itn_future", "irs_future", "lsm", "routine"]
INTERVENTION_DAY = 9 * 365

# Precomputed column indices for AFTER9_COVARS within STATIC_COVARS.
_AFTER9_COL_INDICES: list[int] = [STATIC_COVARS.index(c) for c in AFTER9_COVARS if c in STATIC_COVARS]


def get_input_size(use_cyclical_time: bool) -> int:
    """
    Compute the input size for the model based on time feature encoding.

    Args:
        use_cyclical_time: Whether to use cyclical encoding for time features.
    Returns:
        Input size for the model.
    """
    time_features = 2 if use_cyclical_time else 1
    intervention_features = 2  # post9 flag and time_since_post9
    static_features = len(STATIC_COVARS)
    return time_features + static_features + intervention_features


class StandardScaler:
    def __init__(self):
        """
        Initialize an unfitted scaler.

        Returns:
            None.
        """
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Fit feature means and scales.

        Args:
            X: Feature matrix.

        Returns:
            Fitted scaler.
        """
        self.mean_ = np.mean(X, axis=0)
        # To avoid division by zero, set scale to 1.0 for any feature with zero variance
        scale = np.std(X, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features.

        Args:
            X: Feature matrix.

        Returns:
            Standardized features.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and standardize features.

        Args:
            X: Feature matrix.

        Returns:
            Standardized features.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Restore standardized features.

        Args:
            X: Standardized feature matrix.

        Returns:
            Features in the original scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return X * self.scale_ + self.mean_


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

    This filters low-signal parameter-simulation pairs, creates or loads the split,
    fits static covariate scaling on train data, and builds sequence records.

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

    # Input size
    input_size = get_input_size(cfg.use_cyclical_time)
    log.info(f"Input size for models set to {input_size}")

    # Build data
    train_data = _build_data(df, train_ps, scaler, cfg)
    val_data = _build_data(df, val_ps, scaler, cfg)
    test_data = _build_data(df, test_ps, scaler, cfg)

    return PreparedData(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_size=input_size,
        scaler=scaler,
        train_param_sims=train_ps,
        val_param_sims=val_ps,
        test_param_sims=test_ps,
    )


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

    # TODO: check if need saving
    save_path = Path(output_dir) / "static_scaler.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler


def _build_static_features(sub: pd.DataFrame, abs_t: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Build scaled static covariate matrix with pre-intervention masking.

    AFTER9_COVARS are zeroed out for timesteps before INTERVENTION_DAY because those
    intervention parameters aren't active yet.

    Args:
        sub: Rows for one parameter-simulation pair, sorted by timestep.
        abs_t: Absolute timestep values (float32, shape T).
        scaler: Fitted static covariate scaler.

    Returns:
        Scaled static feature matrix of shape (T, len(STATIC_COVARS)).
    """
    T = len(abs_t)
    base_static = sub.iloc[0][STATIC_COVARS].values.astype(np.float32)
    raw_matrix = np.tile(base_static, (T, 1))
    pre_mask = abs_t < INTERVENTION_DAY
    if pre_mask.any():
        raw_matrix[np.ix_(pre_mask, _AFTER9_COL_INDICES)] = 0.0
    return scaler.transform(raw_matrix)


def _build_intervention_features(abs_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute post-intervention flag and time-since-intervention in years.

    Args:
        abs_t: Absolute timestep values (float32, shape T).

    Returns:
        post9: Binary flag, 1 on/after INTERVENTION_DAY (shape T).
        t_since9_yrs: Years elapsed since INTERVENTION_DAY, 0 before (shape T).
    """
    post9 = (abs_t >= INTERVENTION_DAY).astype(np.float32)
    t_since9_yrs = (np.maximum(0.0, abs_t - INTERVENTION_DAY) / 365.0).astype(np.float32)
    return post9, t_since9_yrs


def _build_time_features(abs_t: np.ndarray, t: np.ndarray, use_cyclical: bool) -> np.ndarray:
    """
    Build time feature columns — either cyclical (sin/cos of day-of-year) or
    min-max normalised absolute time.

    Args:
        abs_t: Absolute timestep values (float32, shape T).
        t: Relative timestep values (float32, shape T).
        use_cyclical: Whether to use cyclical encoding.

    Returns:
        Time feature matrix of shape (T, 2) for cyclical or (T, 1) for linear.
    """
    if use_cyclical:
        doy = abs_t % 365.0
        sin_t = np.sin(2 * math.pi * doy / 365.0).astype(np.float32)
        cos_t = np.cos(2 * math.pi * doy / 365.0).astype(np.float32)
        return np.stack([sin_t, cos_t], axis=1)
    else:
        t_min, t_max = t.min(), t.max()
        t_norm = ((t - t_min) / (t_max - t_min) if t_max > t_min else t).astype(np.float32)
        return t_norm[:, None]


def _build_targets(sub: pd.DataFrame, predictor: str, eps_prevalence: float) -> np.ndarray:
    """
    Extract and transform target values.

    Args:
        sub: Rows for one parameter-simulation pair.
        predictor: Target column name.
        eps_prevalence: Small offset for prevalence log-transform.

    Returns:
        Transformed target array of shape (T,).
    """
    Y_raw = sub[predictor].values.astype(np.float32)
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
        return sub["exposure_pd"].values.astype(np.float32)
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

        abs_t = sub["abs_timesteps"].values.astype(np.float32)
        t = sub["timesteps"].values.astype(np.float32)

        scaled_static = _build_static_features(sub, abs_t, scaler)
        post9, t_since9_yrs = _build_intervention_features(abs_t)
        time_feats = _build_time_features(abs_t, t, cfg.use_cyclical_time)

        X = np.concatenate([time_feats, scaled_static, post9[:, None], t_since9_yrs[:, None]], axis=1)
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
