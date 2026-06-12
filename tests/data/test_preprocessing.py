import numpy as np
import pytest

from stateMINT.data.preprocessing import (
    StandardScaler,
    prepare_data,
    STATIC_COVARS,
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

    # input_size: 2 cyclical time feats + 12 static + post9 + time_since9
    assert prepared.input_size == 2 + len(STATIC_COVARS) + 2

    sample = prepared.train_data[0]
    T = sample["x"].shape[0]
    assert sample["x"].shape == (T, prepared.input_size)
    assert sample["y"].shape == (T,)
    assert sample["w"].shape == (T,)
    assert np.all(np.isfinite(sample["x"]))


def test_prepare_data_non_cyclical_input_size(sample_df, cfg_factory):
    cfg = cfg_factory(use_cyclical_time=False)
    prepared = prepare_data(sample_df, cfg)
    assert prepared.input_size == 1 + len(STATIC_COVARS) + 2
    assert prepared.train_data[0]["x"].shape[1] == prepared.input_size


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
