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


class StandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(X, axis=0)
        # To avoid division by zero, set scale to 1.0 for any feature with zero variance
        scale = np.std(X, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
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
    input_size = (
        (2 if cfg.use_cyclical_time else 1) + len(STATIC_COVARS) + 2
    )  # time features + static + post9 flag, time_since_post9 (years)
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
    """Filter parameter-simulation pairs where the mean target value is below the threshold."""

    group_means = df.groupby(["parameter_index", "simulation_index"])[target_col].mean()
    valid = set(map(tuple, group_means[group_means >= threshold].index.tolist()))
    df["_ps"] = list(zip(df["parameter_index"], df["simulation_index"]))

    log.info(f"After threshold filter: {len(valid)} parameter-simulation pairs, {len(df)} rows")

    return df[df["_ps"].isin(valid)]


def _load_split(
    split_file: str, df: pd.DataFrame
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    split_df = pd.read_csv(split_file)
    present = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))
    train_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "train"].itertuples()
    } & present
    val_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "val"].itertuples()
    } & present
    test_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "test"].itertuples()
    } & present
    return train_ps, val_ps, test_ps  # type: ignore


def _create_split(
    df: pd.DataFrame, seed: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
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


def _build_data(
    df: pd.DataFrame,
    param_sims: set[tuple[int, int]],
    scaler: StandardScaler,
    cfg: DictConfig,
) -> list[dict[str, np.ndarray]]:
    groups = df.groupby(["parameter_index", "simulation_index"])
    data = []

    for ps in param_sims:
        if ps not in groups.groups:
            continue
        sub = groups.get_group(ps).sort_values("timesteps")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[cfg.predictor])
        T = len(sub)
        if T == 0:
            continue

        abs_t = sub["abs_timesteps"].values.astype(np.float32)
        t = sub["timesteps"].values.astype(np.float32)

        base_static = sub.iloc[0][STATIC_COVARS].values.astype(np.float32)
        raw_matrix = np.tile(base_static, (T, 1))
        post_mask = abs_t >= INTERVENTION_DAY
        for cov in AFTER9_COVARS:
            if cov in STATIC_COVARS:
                j = STATIC_COVARS.index(cov)
                raw_matrix[~post_mask, j] = 0.0
        scaled = scaler.transform(raw_matrix)

        post9 = post_mask.astype(np.float32)
        t_since9_yrs = (np.maximum(0.0, abs_t - INTERVENTION_DAY) / 365.0).astype(np.float32)

        if cfg.use_cyclical_time:
            doy = abs_t % 365.0
            sin_t = np.sin(2 * math.pi * doy / 365.0).astype(np.float32)
            cos_t = np.cos(2 * math.pi * doy / 365.0).astype(np.float32)
            X = np.concatenate([sin_t[:, None], cos_t[:, None], scaled, post9[:, None], t_since9_yrs[:, None]], axis=1)
        else:
            t_min, t_max = t.min(), t.max()
            t_norm = ((t - t_min) / (t_max - t_min) if t_max > t_min else t).astype(np.float32)
            X = np.concatenate([t_norm[:, None], scaled, post9[:, None], t_since9_yrs[:, None]], axis=1)

        Y_raw = sub[cfg.predictor].values.astype(np.float32)
        Y = transform_targets_np(Y_raw, cfg.predictor, cfg.eps_prevalence)
        W = sub["exposure_pd"].values.astype(np.float32) if cfg.predictor == "cases" else np.ones(T, dtype=np.float32)

        data.append(
            {
                "x": X,  # (T, input_size)
                "y": Y,  # (T,)
                "w": W,  # (T,)
            }
        )

    return data
