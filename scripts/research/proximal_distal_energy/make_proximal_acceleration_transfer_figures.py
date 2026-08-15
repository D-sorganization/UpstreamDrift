"""Render the pointwise proximal-acceleration intervention atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "proximal-acceleration-transfer-v1"
import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.proximal_distal_energy.run_proximal_acceleration_transfer_study import (  # noqa: E402
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


def _save(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    figure.savefig(
        pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return pdf, svg


def make_figure() -> tuple[Path, Path]:
    """Plot interface power and required proximal torque across the dose."""
    if not JSON_PATH.exists():
        write_outputs()
    rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for phase, color in _COLORS.items():
        selected = [row for row in rows if row["phase"] == phase]
        acceleration = [row["proximal_acceleration_rad_s2"] for row in selected]
        axes[0].plot(
            acceleration,
            [row["total_grip_power_w"] for row in selected],
            marker="o",
            color=color,
            label=phase,
        )
        axes[1].plot(
            acceleration,
            [row["proximal_control_nm"] for row in selected],
            marker="o",
            color=color,
            label=phase,
        )
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Pointwise Interface-Force Power (W)")
    axes[0].set_title("Transfer Response Is Phase Dependent")
    axes[1].set_ylabel("Required Proximal Actuator Torque (N m)")
    axes[1].set_title("Acceleration Is Not a Free Intervention")
    for axis in axes:
        axis.set_xlabel("Target Proximal-Link Acceleration (rad/s²)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Identical-State Proximal-Acceleration Dose Response")
    return _save(figure, "fig_proximal_acceleration_transfer")


def main() -> None:
    for path in make_figure():
        print(path)


if __name__ == "__main__":
    main()
