from pathlib import Path

import numpy as np
from jaxtyping import Array
from stateMINT.common.dataclasses import Predictor
from stateMINT.common.utils import inverse_transform_np


def plot_preds_targets(
    preds: Array,
    targets: Array,
    ps: Array,
    plot_file: str | Path,
    *,
    window_size: int = 14,
    model_label: str = "Mamba",
    target_label: str = "ABM",
    ylabel: str = "Prevalence",
    predictor: Predictor = "prevalence",
    sims_per_parameter: int = 4,
    parameter_sets_per_page: int = 3,
) -> None:
    """
    Plot model predictions against target time series, grouped by parameter set.

    Args:
        preds: Predicted sequences with shape ``(N, T)``.
        targets: Target sequences with shape ``(N, T)``.
        ps: Parameter/simulation ids with shape ``(N, 2)``.
        plot_file: Output PDF path.
        window_size: Number of days per time step, used to label x-axis in years.
        model_label: Label for the prediction line.
        target_label: Label for the target line.
        ylabel: Y-axis label.
        predictor: Target transform to invert before plotting.
        sims_per_parameter: Number of simulations expected per parameter set.
        parameter_sets_per_page: Number of parameter sets to show on one PDF page.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from tqdm import tqdm

    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)
    ps_np = np.asarray(ps)

    if preds_np.shape != targets_np.shape:
        raise ValueError(f"preds and targets must have matching shapes, got {preds_np.shape} and {targets_np.shape}")
    if preds_np.ndim != 2:
        raise ValueError(f"preds and targets must have shape (N, T), got {preds_np.shape}")
    if ps_np.shape != (preds_np.shape[0], 2):
        raise ValueError(f"ps must have shape ({preds_np.shape[0]}, 2), got {ps_np.shape}")

    preds_np = inverse_transform_np(preds_np, predictor)
    targets_np = inverse_transform_np(targets_np, predictor)

    sort_idx = np.lexsort((ps_np[:, 1], ps_np[:, 0]))
    preds_np = preds_np[sort_idx]
    targets_np = targets_np[sort_idx]
    ps_np = ps_np[sort_idx]

    plot_file = Path(plot_file)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    parameter_indices = np.unique(ps_np[:, 0])
    years = preds_np.shape[1] * window_size / 365
    x_np = np.linspace(0.0, years, preds_np.shape[1])
    if predictor == "prevalence":
        y_limits = (0.0, 1.0)
    else:
        y_values = np.concatenate((targets_np.ravel(), preds_np.ravel()))
        y_values = y_values[np.isfinite(y_values)]
        if y_values.size == 0:
            y_limits = None
        else:
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            if y_min == y_max:
                pad = max(abs(y_min) * 0.05, 1.0)
            else:
                pad = (y_max - y_min) * 0.05
            y_limits = (y_min - pad, y_max + pad)

    page_starts = range(0, len(parameter_indices), parameter_sets_per_page)

    with PdfPages(plot_file) as pdf:
        for start in tqdm(page_starts, desc="Plotting prediction pages", unit="page"):
            page_params = parameter_indices[start : start + parameter_sets_per_page]
            fig, axes = plt.subplots(
                len(page_params),
                sims_per_parameter,
                figsize=(4.8 * sims_per_parameter, 2.9 * len(page_params)),
                squeeze=False,
                sharex=True,
            )

            for row, parameter_index in enumerate(page_params):
                param_mask = ps_np[:, 0] == parameter_index
                sim_indices = np.unique(ps_np[param_mask, 1])[:sims_per_parameter]

                for col in range(sims_per_parameter):
                    ax = axes[row, col]
                    if col >= len(sim_indices):
                        ax.axis("off")
                        continue

                    sim_index = sim_indices[col]
                    sample_indices = np.flatnonzero(param_mask & (ps_np[:, 1] == sim_index))
                    sample_idx = sample_indices[0]

                    ax.plot(x_np, targets_np[sample_idx], color="black", linewidth=2.2, label=target_label)
                    ax.plot(x_np, preds_np[sample_idx], color="blue", linewidth=2.2, label=model_label)
                    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.5, alpha=0.75)
                    if y_limits is not None:
                        ax.set_ylim(*y_limits)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(
                        f"Param {int(parameter_index)} | Sim {int(sim_index)}",
                        fontsize=11,
                        fontweight="bold",
                    )
                    if col == 0:
                        ax.set_ylabel(f"Param set {int(parameter_index)}\n{ylabel}", fontweight="bold")
                    if row == len(page_params) - 1:
                        ax.set_xlabel("Years", fontweight="bold")

                for col in range(len(sim_indices), sims_per_parameter):
                    axes[row, col].axis("off")

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", frameon=True)
            first_param = int(page_params[0])
            last_param = int(page_params[-1])
            title = f"Parameter sets {first_param}"
            if first_param != last_param:
                title += f"--{last_param}"
            title += f" - {ylabel}: {target_label} vs {model_label}"
            fig.suptitle(title, fontsize=16, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 0.96, 0.93))
            pdf.savefig(fig)
            plt.close(fig)
