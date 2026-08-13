"""Render publication figures for the rotating-base torso-velocity study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rotating-base-torso-velocity-v1"
import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.proximal_distal_energy.run_rotating_base_torso_velocity_study import (
    JSON_PATH,
    NPZ_PATH,
    main as write_outputs,
)

FIGURE_DIR = JSON_PATH.parent / "rotating_base_torso_velocity" / "figures"
_PROFILE_COLORS = {
    "accelerate": "#4C78A8",
    "constant_rate": "#72B7B2",
    "decelerate": "#E45756",
}


def _load() -> tuple[dict, np.lib.npyio.NpzFile]:
    if not JSON_PATH.exists() or not NPZ_PATH.exists():
        write_outputs()
    return json.loads(JSON_PATH.read_text(encoding="utf-8")), np.load(NPZ_PATH)


def _save(figure: plt.Figure, stem: str) -> tuple[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    figure.savefig(
        pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)
    return (pdf,)


def _matched_grid(record: dict) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for axis, matching_rule in zip(
        axes, ("relative_club_rate", "absolute_club_rate"), strict=True
    ):
        for profile, color in _PROFILE_COLORS.items():
            rows = [
                row
                for row in record["cases"]
                if row["matching_rule"] == matching_rule
                and row["torso_profile"] == profile
            ]
            axis.plot(
                [row["initial_torso_rate_rad_s"] for row in rows],
                [row["impact_speed_m_s"] for row in rows],
                marker="o",
                color=color,
                label=profile.replace("_", " ").title(),
            )
            invalid = [row for row in rows if not row["valid"]]
            axis.scatter(
                [row["initial_torso_rate_rad_s"] for row in invalid],
                [row["impact_speed_m_s"] for row in invalid],
                marker="x",
                s=70,
                color="#111111",
                zorder=5,
            )
        axis.set_title(matching_rule.replace("_", " ").title())
        axis.set_xlabel("Initial Torso Rate (rad/s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Delivery Clubhead Speed (m/s)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Torso-Rate Associations Depend on the Matched-State Contract")
    figure.tight_layout()
    return _save(figure, "fig_rotating_base_matched_grid")


def _force_power_atlas(record: dict, arrays: np.lib.npyio.NpzFile) -> tuple[Path, Path]:
    selected = [
        row
        for row in record["cases"]
        if row["matching_rule"] == "relative_club_rate"
        and row["torso_profile"] == "constant_rate"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    for row, color in zip(selected, ("#4C78A8", "#F58518", "#E45756"), strict=True):
        prefix = f"case_{row['case_index']:02d}_"
        time = arrays[prefix + "time_s"]
        label = f"{row['initial_torso_rate_rad_s']:.1f} rad/s"
        axes[0, 0].plot(
            time, arrays[prefix + "torso_rate_rad_s"], color=color, label=label
        )
        axes[0, 1].plot(time, arrays[prefix + "clubhead_speed_m_s"], color=color)
        axes[1, 0].plot(time, arrays[prefix + "contact_power_on_club_w"], color=color)
        axes[1, 1].plot(time, arrays[prefix + "force_generated_couple_nm"], color=color)
    axes[0, 0].set_ylabel("Torso Rate (rad/s)")
    axes[0, 1].set_ylabel("Clubhead Speed (m/s)")
    axes[1, 0].set_ylabel("Bilateral Contact Power (W)")
    axes[1, 1].set_ylabel("Force-Generated Couple (N m)")
    for axis in axes[1]:
        axis.set_xlabel("Time From Registered State (s)")
    for axis in axes.flat:
        axis.axhline(0.0, color="#555555", linewidth=0.7)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Bilateral Reactions Mediate the Rotating-Base Transfer")
    figure.tight_layout()
    return _save(figure, "fig_rotating_base_force_power_atlas")


def _falsification_summary(record: dict) -> tuple[Path, Path]:
    channels = record["same_state_killswitch"]["channels"]
    labels = ["Torso", "Bilateral Arms", "Bilateral Wrists"]
    speed_effects = [
        channels[key]["delivery_speed_difference_m_s"]
        for key in ("torso", "bilateral_arm", "bilateral_wrist")
    ]
    shaft = record["shaft_sensitivity"]
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    axes[0].bar(labels, speed_effects, color=("#4C78A8", "#72B7B2", "#F58518"))
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Baseline Minus Killswitch Delivery Speed (m/s)")
    axes[0].set_title("Exact Same-State Command Killswitches")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].plot(
        [row["shaft_stiffness_nm_rad"] for row in shaft],
        [row["impact_speed_m_s"] for row in shaft],
        marker="o",
        color="#7A5195",
    )
    axes[1].set_xlabel("Shaft Torsional Stiffness (N m/rad)")
    axes[1].set_ylabel("Delivery Clubhead Speed (m/s)")
    axes[1].set_title("Compliance Sensitivity")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Killswitch and Compliance Checks Bound the Mechanism Claim")
    figure.tight_layout()
    return _save(figure, "fig_rotating_base_falsification_summary")


def make_figures() -> tuple[Path, ...]:
    """Render all registered figures and return their paths."""
    record, arrays = _load()
    try:
        return (
            *_matched_grid(record),
            *_force_power_atlas(record, arrays),
            *_falsification_summary(record),
        )
    finally:
        arrays.close()


def main() -> None:
    for path in make_figures():
        print(path)


if __name__ == "__main__":
    main()
