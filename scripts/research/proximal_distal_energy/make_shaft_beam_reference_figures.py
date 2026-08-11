"""Render publication figures for the distributed-shaft comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_DIR = ARTICLE / "data"
FIGURE_DIR = ARTICLE / "figures"
FIGURE_STEMS = (
    "fig_shaft_beam_identification",
    "fig_shaft_beam_response",
    "fig_shaft_beam_energy",
)

INK = "#263238"
BLUE = "#246A8D"
ORANGE = "#E07A2D"
RED = "#B23A48"
GREEN = "#2A7F62"
PURPLE = "#7251A3"
GRAY = "#7A858A"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _load(data_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads(
        (data_dir / "shaft_beam_reference.json").read_text(encoding="utf-8")
    )
    with np.load(data_dir / "shaft_beam_reference.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    return record, arrays


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path,
            format=suffix,
            metadata={"Creator": "Proximal-Distal Open Research Resource"},
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def _identification_figure(record: dict, output_dir: Path) -> list[Path]:
    identified = record["identification"]
    target = np.asarray(identified["target_frequencies_hz"])
    fitted = np.asarray(identified["fitted_frequencies_hz"])
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    positions = np.arange(2)
    width = 0.34
    axes[0].bar(positions - width / 2, target, width, color=BLUE, label="Target")
    axes[0].bar(positions + width / 2, fitted, width, color=ORANGE, label="Fitted")
    axes[0].set_xticks(positions, ("Mode 1", "Mode 2"))
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Synthetic Modal Identification")
    axes[0].legend(frameon=False)

    fitted_values = np.array(
        [identified["youngs_modulus_pa"] / 1e9, identified["head_mass_kg"]]
    )
    truth = np.array(
        [
            identified["declared_truth_youngs_modulus_pa"] / 1e9,
            identified["declared_truth_head_mass_kg"],
        ]
    )
    lower = np.array(
        [
            identified["youngs_modulus_interval_pa"][0] / 1e9,
            identified["head_mass_interval_kg"][0],
        ]
    )
    upper = np.array(
        [
            identified["youngs_modulus_interval_pa"][1] / 1e9,
            identified["head_mass_interval_kg"][1],
        ]
    )
    normalized = fitted_values / truth
    axes[1].errorbar(
        (0, 1),
        normalized,
        yerr=np.vstack(
            ((fitted_values - lower) / truth, (upper - fitted_values) / truth)
        ),
        fmt="o",
        color=PURPLE,
        capsize=4,
        label="Fit and Assumed-Noise Interval",
    )
    axes[1].axhline(1.0, color=INK, lw=1.2, ls="--", label="Declared Truth")
    axes[1].set_xticks((0, 1), ("Elastic Modulus", "Head Mass"))
    axes[1].set_ylabel("Estimate / Declared Truth")
    axes[1].set_title("Parameter Recovery Is Synthetic")
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle("Distributed-Shaft Identification and Structural Convergence")
    figure.tight_layout()
    return _save(figure, output_dir, FIGURE_STEMS[0])


def _response_figure(arrays: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    time_ms = arrays["time_s"] * 1e3
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), sharey=True)
    for axis, prefix, title in (
        (axes[0], "low", "Slow Tip-Force Pulse"),
        (axes[1], "high", "Short Tip-Force and Moment Pulse"),
    ):
        axis.plot(
            time_ms,
            arrays[f"{prefix}_reference_tip_deflection_m"] * 1e3,
            color=BLUE,
            lw=1.7,
            label="Six-Mode Beam Reference",
        )
        axis.plot(
            time_ms,
            arrays[f"{prefix}_reduced_tip_deflection_m"] * 1e3,
            color=ORANGE,
            lw=1.3,
            ls="--",
            label="One-Mode Reduction",
        )
        axis.set_xlabel("Time (ms)")
        axis.set_title(title)
        axis.axhline(0.0, color=GRAY, lw=0.7)
    axes[0].set_ylabel("Tip Deflection (mm)")
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle("Higher Modes Become Visible Under Short-Duration Loading")
    figure.tight_layout()
    return _save(figure, output_dir, FIGURE_STEMS[1])


def _cumulative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _energy_figure(
    record: dict, arrays: dict[str, np.ndarray], output_dir: Path
) -> list[Path]:
    time = arrays["time_s"]
    time_ms = time * 1e3
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.35))
    for prefix, color, label in (
        ("high_reference", BLUE, "Six-Mode Beam Reference"),
        ("high_reduced", ORANGE, "One-Mode Reduction"),
    ):
        energy = arrays[f"{prefix}_energy_j"]
        work = _cumulative(arrays[f"{prefix}_input_power_w"], time)
        loss = _cumulative(arrays[f"{prefix}_damping_power_w"], time)
        axes[0].plot(time_ms, energy, color=color, lw=1.6, label=label)
        axes[1].plot(
            time_ms,
            energy - work - loss,
            color=color,
            lw=1.4,
            label=label,
        )
    axes[0].set_title("Mechanical Energy")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Energy (J)")
    axes[0].legend(frameon=False)
    axes[1].set_title("Work-Energy Closure Residual")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Residual (J)")
    maximum = record["closure"]["maximum_reference_work_energy_residual_j"]
    axes[1].text(
        0.98,
        0.95,
        f"Final absolute residual ≤ {maximum:.2e} J",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=INK,
    )
    figure.suptitle("Declared Input, Damping Loss, and Stored Beam Energy Close")
    figure.tight_layout()
    return _save(figure, output_dir, FIGURE_STEMS[2])


def render_beam_reference_figures(
    data_dir: Path = DATA_DIR, output_dir: Path = FIGURE_DIR
) -> list[Path]:
    """Render three figures in paired PDF and SVG formats."""
    if not isinstance(data_dir, Path) or not isinstance(output_dir, Path):
        raise TypeError("data_dir and output_dir must be pathlib.Path values")
    _style()
    record, arrays = _load(data_dir)
    return [
        *_identification_figure(record, output_dir),
        *_response_figure(arrays, output_dir),
        *_energy_figure(record, arrays, output_dir),
    ]


def main() -> None:
    render_beam_reference_figures()


if __name__ == "__main__":
    main()


__all__ = ["render_beam_reference_figures"]
