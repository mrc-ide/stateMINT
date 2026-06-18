import numpy as np
import pandas as pd
import pytest

from stateMINT.data.preprocessing import (
    StandardScaler,
    prepare_data,
    STATIC_COVARS,
    INTERVENTION_DAY,
    _AFTER9_COL_INDICES,
    INPUT_SIZE,
    _build_static_features,
    _build_intervention_features,
    _build_time_features,
    _build_targets,
    _build_weights,
)


# ----------------------- StandardScaler -----------------------


def test_scaler_fit_transform_standardizes():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-6)


def test_scaler_handles_zero_variance_column():
    X = np.array([[2.0, 1.0], [2.0, 5.0]])  # first column constant
    out = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, 0], 0.0)


def test_scaler_transform_before_fit_raises():
    with pytest.raises(ValueError):
        StandardScaler().transform(np.zeros((2, 2)))


def test_scaler_inverse_roundtrip():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaler = StandardScaler().fit(X)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X, rtol=1e-6)


# ----------------------- prepare_data -----------------------


@pytest.mark.parametrize("predictor", ["prevalence", "cases"])
def test_prepare_data_shapes_and_split(sample_df, cfg_factory, predictor):
    cfg = cfg_factory(predictor=predictor)
    prepared = prepare_data(sample_df, cfg)

    # splits partition the 4 parameters with no overlap
    all_ps = prepared.train_param_sims | prepared.val_param_sims | prepared.test_param_sims
    assert len(all_ps) == 4
    assert not (prepared.train_param_sims & prepared.test_param_sims)

    assert prepared.input_size == INPUT_SIZE

    sample = prepared.train_data[0]
    T = sample["x"].shape[0]
    assert sample["x"].shape == (T, prepared.input_size)
    assert sample["y"].shape == (T,)
    assert sample["w"].shape == (T,)
    assert np.all(np.isfinite(sample["x"]))


def test_prepare_data_writes_scaler_and_split(sample_df, cfg_factory, tmp_path):
    cfg = cfg_factory()
    prepare_data(sample_df, cfg)
    assert (tmp_path / "static_scaler.pkl").exists()
    assert (tmp_path / "split.csv").exists()


def test_cases_weights_use_exposure(sample_df, cfg_factory):
    # For 'cases', weights come from exposure_pd (not all-ones).
    prepared = prepare_data(sample_df, cfg_factory(predictor="cases"))
    w = prepared.train_data[0]["w"]
    assert not np.allclose(w, 1.0)


@pytest.mark.filterwarnings("ignore:.*empty slice.*", "ignore:.*invalid value.*", "ignore:.*Degrees of freedom.*")
def test_threshold_filters_out_low_pairs(sample_df, cfg_factory):
    # Threshold above every prevalence mean drops all data (scaler fits on empty).
    prepared = prepare_data(sample_df, cfg_factory(min_prevalence=10.0))
    assert prepared.train_data == [] and prepared.val_data == [] and prepared.test_data == []


# ----------------------- _build_* helpers -----------------------

# Fixtures shared across helper tests


@pytest.fixture
def straddling_abs_t():
    """Four timesteps: 2 before, 1 at, 1 after INTERVENTION_DAY."""
    return np.array(
        [INTERVENTION_DAY - 10, INTERVENTION_DAY - 5, INTERVENTION_DAY, INTERVENTION_DAY + 10],
        dtype=np.float32,
    )


@pytest.fixture
def simple_sub(straddling_abs_t):
    T = len(straddling_abs_t)
    rng = np.random.default_rng(42)
    data: dict = {
        "timesteps": list(range(1, T + 1)),
        "abs_timesteps": straddling_abs_t.tolist(),
        "prevalence": [0.1, 0.2, 0.3, 0.4],
        "cases": [5.0, 10.0, 15.0, 20.0],
        "exposure_pd": [100.0, 200.0, 300.0, 400.0],
    }
    for c in STATIC_COVARS:
        data[c] = [float(rng.uniform(0.1, 1.0))] * T
    return pd.DataFrame(data)


@pytest.fixture
def identity_scaler():
    """Scaler with mean=0, scale=1 so transform is a no-op."""
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(STATIC_COVARS), dtype=np.float32)
    scaler.scale_ = np.ones(len(STATIC_COVARS), dtype=np.float32)
    return scaler


# _build_static_features


def test_build_static_features_shape(simple_sub, straddling_abs_t, identity_scaler):
    out = _build_static_features(simple_sub, straddling_abs_t, identity_scaler)
    assert out.shape == (len(straddling_abs_t), len(STATIC_COVARS))


def test_build_static_features_pre_intervention_after9_zeroed(simple_sub, straddling_abs_t, identity_scaler):
    out = _build_static_features(simple_sub, straddling_abs_t, identity_scaler)
    pre_mask = straddling_abs_t < INTERVENTION_DAY
    # With identity scaler, zeroed raw values pass through as 0.0.
    np.testing.assert_array_equal(out[pre_mask][:, _AFTER9_COL_INDICES], 0.0)


def test_build_static_features_post_intervention_after9_nonzero(simple_sub, straddling_abs_t, identity_scaler):
    out = _build_static_features(simple_sub, straddling_abs_t, identity_scaler)
    post_mask = straddling_abs_t >= INTERVENTION_DAY
    # Static values are in [0.1, 1.0] so post-intervention AFTER9 columns must be non-zero.
    assert np.all(out[post_mask][:, _AFTER9_COL_INDICES] != 0.0)


def test_build_static_features_scaler_applied(simple_sub, straddling_abs_t):
    rng = np.random.default_rng(99)
    X_train = rng.uniform(0, 1, (20, len(STATIC_COVARS))).astype(np.float32)
    scaler = StandardScaler().fit(X_train)
    out = _build_static_features(simple_sub, straddling_abs_t, scaler)
    # Rows on/after intervention day should equal scaler.transform of the raw static row.
    post_mask = straddling_abs_t >= INTERVENTION_DAY
    raw = np.tile(simple_sub.iloc[0][STATIC_COVARS].values.astype(np.float32), (post_mask.sum(), 1))
    np.testing.assert_allclose(out[post_mask], scaler.transform(raw), rtol=1e-6)


# _build_intervention_features


def test_build_intervention_features_before_zero(straddling_abs_t):
    post9, t_since9 = _build_intervention_features(straddling_abs_t)
    pre = straddling_abs_t < INTERVENTION_DAY
    np.testing.assert_array_equal(post9[pre], 0.0)
    np.testing.assert_array_equal(t_since9[pre], 0.0)


def test_build_intervention_features_at_and_after(straddling_abs_t):
    post9, t_since9 = _build_intervention_features(straddling_abs_t)
    post = straddling_abs_t >= INTERVENTION_DAY
    np.testing.assert_array_equal(post9[post], 1.0)
    expected = (straddling_abs_t[post] - INTERVENTION_DAY) / 365.0
    np.testing.assert_allclose(t_since9[post], expected, rtol=1e-6)


def test_build_intervention_features_dtype(straddling_abs_t):
    post9, t_since9 = _build_intervention_features(straddling_abs_t)
    assert post9.dtype == np.float32
    assert t_since9.dtype == np.float32


# _build_time_features


def test_build_time_features_linear_shape(straddling_abs_t):
    t = np.arange(len(straddling_abs_t), dtype=np.float32)
    out = _build_time_features(t)
    assert out.shape == (len(straddling_abs_t), 1)


def test_build_time_features_linear_range(straddling_abs_t):
    t = np.arange(len(straddling_abs_t), dtype=np.float32)
    out = _build_time_features(t)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_build_time_features_linear_constant_t():
    t = np.array([5.0, 5.0], dtype=np.float32)
    out = _build_time_features(t)
    # When t is constant, no normalization — returns t unchanged.
    np.testing.assert_array_equal(out.squeeze(), t)


# _build_targets


def test_build_targets_shape_and_finite(simple_sub):
    out = _build_targets(simple_sub, "prevalence", eps_prevalence=1e-5)
    assert out.shape == (len(simple_sub),)
    assert np.all(np.isfinite(out))


def test_build_targets_prevalence_is_logit(simple_sub):
    out = _build_targets(simple_sub, "prevalence", eps_prevalence=1e-5)
    raw = simple_sub["prevalence"].values.astype(np.float32)
    expected = np.log(raw / (1.0 - raw))
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_build_targets_cases_is_log1p(simple_sub):
    out = _build_targets(simple_sub, "cases", eps_prevalence=1e-5)
    raw = simple_sub["cases"].values.astype(np.float32)
    np.testing.assert_allclose(out, np.log1p(raw), rtol=1e-5)


# _build_weights


def test_build_weights_prevalence_ones(simple_sub):
    w = _build_weights(simple_sub, "prevalence")
    np.testing.assert_array_equal(w, np.ones(len(simple_sub), dtype=np.float32))


def test_build_weights_cases_equals_exposure(simple_sub):
    w = _build_weights(simple_sub, "cases")
    np.testing.assert_array_equal(w, simple_sub["exposure_pd"].values.astype(np.float32))
