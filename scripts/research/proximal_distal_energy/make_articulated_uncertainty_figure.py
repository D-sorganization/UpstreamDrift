"""Render the governed articulated LHS sensitivity and status evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
RECORD = ARTICLE / "data/articulated_uncertainty_study.json"
ARRAYS = ARTICLE / "data/articulated_uncertainty_study.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_uncertainty_screen"


def _label(value: str) -> str:
    labels = {
        "peak_station_force_n": "Peak Station Force (N)",
        "peak_force_couple_nm": "Peak Force Couple (N m)",
        "max_sliding_speed_m_s": "Maximum Sliding Speed (m/s)",
        "total_transition_count": "Total Transition Count",
        "normalized_work_energy_residual": "Normalized Work--Energy Residual",
        "grip_stiffness_n_m": "Grip Stiffness (N/m)",
        "grip_damping_n_s_m": "Grip Damping (N s/m)",
        "club_mass_kg": "Club Mass (kg)",
        "initial_velocity_m_s": "Initial Velocity (m/s)",
    }
    return labels.get(value, value.replace("_", " ").title())


def _load_evidence(
    record_path: Path, arrays_path: Path
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "articulated-uncertainty-study/v2":
        raise ValueError("figure rendering requires the governed v2 record")
    with np.load(arrays_path) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    sample_count = int(record["results"]["sample_count"])
    if arrays["parameter_samples"].shape[0] != sample_count:
        raise ValueError("record and array sample counts do not match")
    parameters = tuple(str(value) for value in arrays["parameter_names"])
    metrics = tuple(str(value) for value in arrays["output_metric_names"])
    if parameters != tuple(record["uncertainty_parameters"]):
        raise ValueError("record and array parameter names do not match")
    if metrics != tuple(record["output_metrics"]):
        raise ValueError("record and array output names do not match")
    expected_shape = (len(metrics), len(parameters))
    if arrays["prcc_sensitivity_matrix"].shape != expected_shape:
        raise ValueError("PRCC matrix does not match the registered design")
    if arrays["failure_classes"].shape != (sample_count,):
        raise ValueError("trajectory-status rows do not match the sample count")
    return record, arrays


def render_articulated_uncertainty_figure(
    record_path: Path = RECORD,
    arrays_path: Path = ARRAYS,
    output_base: Path = OUTPUT,
) -> None:
    """Render PRCC screening and the retained trajectory-status distribution."""

    record, arrays = _load_evidence(record_path, arrays_path)
    parameters = [str(value) for value in arrays["parameter_names"]]
    metrics = [str(value) for value in arrays["output_metric_names"]]
    prcc = np.asarray(arrays["prcc_sensitivity_matrix"], dtype=float)
    distribution = record["results"]["failure_distribution"]
    statuses = list(distribution)
    counts = [int(distribution[name]) for name in statuses]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 7.4),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
        constrained_layout=True,
    )
    image = axes[0].imshow(prcc, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    axes[0].set_xticks(range(len(parameters)), [_label(name) for name in parameters])
    axes[0].set_yticks(range(len(metrics)), [_label(name) for name in metrics])
    axes[0].tick_params(axis="x", labelrotation=48)
    axes[0].set_title("Exploratory Partial Rank Correlations")
    for row in range(prcc.shape[0]):
        for column in range(prcc.shape[1]):
            value = float(prcc[row, column])
            axes[0].text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) >= 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82)
    colorbar.set_label("PRCC (Exploratory, No Population Interval)")

    positions = np.arange(len(statuses))
    axes[1].barh(positions, counts, color="#4c78a8")
    axes[1].set_yticks(positions, [_label(name) for name in statuses])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Retained Sample Count")
    axes[1].set_title("Complete Trajectory-Status Distribution")
    axes[1].grid(axis="x", alpha=0.22)
    for position, count in zip(positions, counts, strict=True):
        axes[1].text(count + 0.15, position, str(count), va="center", fontsize=9)

    included = int(record["results"]["analysis_included_count"])
    total = int(record["results"]["sample_count"])
    fig.suptitle(
        "Articulated Closed-State and Local-Trajectory Uncertainty Screen\n"
        f"Deterministic LHS: {included}/{total} Samples Enter PRCC; "
        "Engineering Bounds, Not Human Calibration",
        fontsize=13,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main() -> None:
    render_articulated_uncertainty_figure()
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
