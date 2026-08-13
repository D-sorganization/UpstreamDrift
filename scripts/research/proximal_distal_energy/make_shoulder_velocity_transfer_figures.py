"""Render the phase-resolved shoulder-velocity transfer atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.proximal_distal_energy.run_shoulder_velocity_transfer_study import (
    FIGURE_DIR,
    JSON_PATH,
    write_outputs,
)

_COLORS = {
    "Transition": "#4C78A8",
    "Early Downswing": "#72B7B2",
    "Mid-Downswing": "#F58518",
    "Delivery and Release": "#E45756",
    "Pre-Impact": "#7A5195",
}


def _load_rows() -> list[dict]:
    if not JSON_PATH.exists():
        write_outputs()
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))["rows"]


def _save(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)
    return pdf, svg


def _power_by_phase(rows: list[dict]) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for axis, constraint in zip(
        axes,
        ("preserve_relative_club_rate", "preserve_absolute_club_rate"),
        strict=True,
    ):
        for phase, color in _COLORS.items():
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["velocity_constraint"] == constraint
            ]
            axis.plot(
                [row["proximal_velocity_rad_s"] for row in selected],
                [row["drift_grip_power_w"] for row in selected],
                marker="o",
                linewidth=1.6,
                color=color,
                label=phase,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        title = (
            "Relative Club Rate Preserved"
            if constraint == "preserve_relative_club_rate"
            else "Absolute Club Rate Preserved"
        )
        axis.set_title(title)
        axis.set_xlabel("Proximal-Link Angular Velocity (rad/s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Pointwise Drift Grip Power (W)")
    axes[1].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Drift Grip Power Depends on Phase and the Matched-Velocity Contract"
    )
    figure.tight_layout()
    return _save(figure, "fig_shoulder_velocity_drift_power")


def _braking_map(rows: list[dict]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for phase, color in _COLORS.items():
        selected = [
            row
            for row in rows
            if row["phase"] == phase
            and row["velocity_constraint"] == "preserve_absolute_club_rate"
        ]
        axis.scatter(
            [row["clubhead_speed_m_s"] for row in selected],
            [row["total_grip_power_w"] for row in selected],
            color=color,
            label=phase,
            s=34,
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.fill_between(
        [0.0, axis.get_xlim()[1]],
        axis.get_ylim()[0],
        0.0,
        color="#E45756",
        alpha=0.08,
        label="Negative Grip-Force Work Rate",
    )
    axis.set_title("High Proximal Speed Does Not Uniformly Remove Grip Braking")
    axis.set_xlabel("Instantaneous Clubhead Speed (m/s)")
    axis.set_ylabel("Total Grip-Force Power (W)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return _save(figure, "fig_shoulder_velocity_braking_map")


def _slope_summary(rows: list[dict]) -> tuple[Path, Path]:
    phases = list(_COLORS)
    constraints = (
        "preserve_relative_club_rate",
        "preserve_absolute_club_rate",
    )
    slopes = np.zeros((len(phases), len(constraints)))
    for row_index, phase in enumerate(phases):
        for column, constraint in enumerate(constraints):
            selected = [
                row
                for row in rows
                if row["phase"] == phase and row["velocity_constraint"] == constraint
            ]
            x = np.asarray([row["proximal_velocity_rad_s"] for row in selected])
            y = np.asarray([row["drift_grip_power_w"] for row in selected])
            slopes[row_index, column] = np.polyfit(x, y, 1)[0]
    limit = float(np.max(np.abs(slopes)))
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    image = axis.imshow(slopes, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks((0, 1), ("Relative Rate Held", "Absolute Rate Held"))
    axis.set_yticks(np.arange(len(phases)), phases)
    for row in range(slopes.shape[0]):
        for column in range(slopes.shape[1]):
            axis.text(
                column, row, f"{slopes[row, column]:.1f}", ha="center", va="center"
            )
    axis.set_title("Local Drift-Power Sensitivity Changes Sign Across Swing Phases")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Slope (W per rad/s)")
    figure.tight_layout()
    return _save(figure, "fig_shoulder_velocity_phase_sensitivity")


def make_figures() -> tuple[Path, ...]:
    """Render all registered figures and return their paths."""
    rows = _load_rows()
    return (*_power_by_phase(rows), *_braking_map(rows), *_slope_summary(rows))


def main() -> None:
    for path in make_figures():
        print(path)


if __name__ == "__main__":
    main()
