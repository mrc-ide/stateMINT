import numpy as np

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
BURNIN_DAY = 6 * 365  # 2190 — kept window starts here
TOTAL_DAYS = 12 * 365  # 6yr warmup + 6yr sim

INPUT_SIZE = 1 + len(STATIC_COVARS) + 2  # time features + static covars + post9, t_since9_yrs


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
