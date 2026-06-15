from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stateMINT.common.utils import inverse_transform_np
from stateMINT.eval.viz_preds_truth import _compute_y_limits, plot_preds_targets


def test_compute_y_limits_basic():
    arr1, arr2 = np.array([1.0, 2.0]), np.array([0.0, 4.0])
    lo, hi = _compute_y_limits(arr1, arr2)  # type: ignore
    pad = (4.0 - 0.0) * 0.05
    assert lo == pytest.approx(0.0 - pad)
    assert hi == pytest.approx(4.0 + pad)


def test_compute_y_limits_constant_values():
    arr = np.array([2.0, 2.0, 2.0])
    lo, hi = _compute_y_limits(arr)  # type: ignore
    pad = max(abs(2.0) * 0.05, 1.0)
    assert lo == pytest.approx(2.0 - pad)
    assert hi == pytest.approx(2.0 + pad)


def test_compute_y_limits_all_nonfinite():
    arr = np.array([np.nan, np.inf, -np.inf])
    assert _compute_y_limits(arr) is None


def _make_data(n, t):
    rng = np.random.default_rng(0)
    preds = rng.uniform(0.1, 0.9, size=(n, t))
    targets = rng.uniform(0.1, 0.9, size=(n, t))
    ps = np.array([[i // 2, i % 2] for i in range(n)])
    return preds, targets, ps


def test_plot_preds_targets_shape_mismatch_raises(tmp_path):
    preds, targets, ps = _make_data(4, 6)
    with pytest.raises(ValueError, match="matching shapes"):
        plot_preds_targets(preds, targets[:, :-1], ps, tmp_path / "out.pdf")


def test_plot_preds_targets_wrong_ndim_raises(tmp_path):
    preds, targets, ps = _make_data(4, 6)
    with pytest.raises(ValueError, match="shape \\(N, T\\)"):
        plot_preds_targets(preds.ravel(), targets.ravel(), ps, tmp_path / "out.pdf")


def test_plot_preds_targets_bad_ps_shape_raises(tmp_path):
    preds, targets, ps = _make_data(4, 6)
    with pytest.raises(ValueError, match="ps must have shape"):
        plot_preds_targets(preds, targets, ps[:, :1], tmp_path / "out.pdf")


def test_plot_preds_targets_prevalence_creates_pdf(tmp_path):
    preds, targets, ps = _make_data(4, 6)
    out = tmp_path / "out.pdf"
    plot_preds_targets(preds, targets, ps, out, predictor="prevalence", sims_per_parameter=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_preds_targets_partial_sim_grid(tmp_path):
    # Only one simulation for the parameter set, but sims_per_parameter requests two.
    preds, targets, ps = _make_data(1, 5)
    out = tmp_path / "out.pdf"
    plot_preds_targets(preds, targets, ps, out, predictor="prevalence", sims_per_parameter=2)
    assert out.exists()


def _run_with_mocked_plotting(preds, targets, ps, out, **kwargs):
    """Run plot_preds_targets with matplotlib mocked, capturing per-axes plot/title/ylim calls."""
    plot_calls = {}
    titles = {}
    ylims = {}

    def fake_subplots(nrows, ncols, **_):
        fig = MagicMock()
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                ax = MagicMock()
                ax.get_legend_handles_labels.return_value = ([], [])

                def record_plot(x, y, r=r, c=c, **plot_kwargs):
                    plot_calls.setdefault((r, c), []).append((np.asarray(x), np.asarray(y), plot_kwargs.get("label")))

                def record_title(text, r=r, c=c, **_kw):
                    titles[(r, c)] = text

                def record_ylim(lo, hi, r=r, c=c):
                    ylims[(r, c)] = (lo, hi)

                ax.plot.side_effect = record_plot
                ax.set_title.side_effect = record_title
                ax.set_ylim.side_effect = record_ylim
                axes[r, c] = ax
        return fig, axes

    pdf_cm = MagicMock()
    pdf_cm.__enter__.return_value = MagicMock()
    pdf_cm.__exit__.return_value = False

    with (
        patch("matplotlib.pyplot.subplots", side_effect=fake_subplots),
        patch("matplotlib.backends.backend_pdf.PdfPages", return_value=pdf_cm),
    ):
        plot_preds_targets(preds, targets, ps, out, **kwargs)

    return plot_calls, titles, ylims


def test_plot_preds_targets_sorts_transforms_and_labels_axes(tmp_path):
    # ps deliberately out of order; rows carry distinct constant values so we can
    # trace which original row ends up in which (sorted) subplot slot.
    ps = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    t = 4
    window_size = 14
    raw_preds = np.array([[0.0] * t, [1.0] * t, [2.0] * t, [3.0] * t])
    raw_targets = raw_preds * 10

    plot_calls, titles, _ = _run_with_mocked_plotting(
        raw_preds,
        raw_targets,
        ps,
        tmp_path / "out.pdf",
        predictor="cases",
        sims_per_parameter=2,
        parameter_sets_per_page=2,
        window_size=window_size,
    )

    # Sorted (param, sim) order is: (0,0)->orig idx2, (0,1)->orig idx1, (1,0)->orig idx0, (1,1)->orig idx3
    assert titles[(0, 0)] == "Param 0 | Sim 0"
    assert titles[(0, 1)] == "Param 0 | Sim 1"
    assert titles[(1, 0)] == "Param 1 | Sim 0"
    assert titles[(1, 1)] == "Param 1 | Sim 1"

    expected_pred = inverse_transform_np(raw_preds, "cases")
    expected_target = inverse_transform_np(raw_targets, "cases")

    for (row, col), orig_idx in [((0, 0), 2), ((0, 1), 1), ((1, 0), 0), ((1, 1), 3)]:
        calls = {label: y for _, y, label in plot_calls[(row, col)]}
        assert np.allclose(calls["ABM"], expected_target[orig_idx])
        assert np.allclose(calls["Mamba"], expected_pred[orig_idx])

    expected_x = np.linspace(0.0, t * window_size / 365, t)
    for x, _, _ in plot_calls[(0, 0)]:
        assert np.allclose(x, expected_x)


def test_plot_preds_targets_ylim_modes(tmp_path):
    preds, targets, ps = _make_data(2, 5)

    _, _, ylims_prevalence = _run_with_mocked_plotting(
        preds, targets, ps, tmp_path / "prev.pdf", predictor="prevalence", sims_per_parameter=2
    )
    assert all(ylim == (0.0, 1.0) for ylim in ylims_prevalence.values())

    _, _, ylims_cases = _run_with_mocked_plotting(
        preds, targets, ps, tmp_path / "cases.pdf", predictor="cases", sims_per_parameter=2
    )
    expected = _compute_y_limits(inverse_transform_np(targets, "cases"), inverse_transform_np(preds, "cases"))
    assert all(ylim == pytest.approx(expected) for ylim in ylims_cases.values())
