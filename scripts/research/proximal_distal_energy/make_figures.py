"""Generate report figures from the recorded experiment data.

Reads ``data/e1_sweep.json`` and ``data/representative_traces.npz``
produced by :mod:`run_experiments` and writes publication figures (PDF)
into the report ``figures/`` directory. Run after the experiments::

    python3 -m scripts.research.proximal_distal_energy.make_figures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"

REP_LABELS = {
    "passive": "Passive wrist (free hinge)",
    "early_drive": "Early drive (onset 0 s)",
    "best_drive": "Late drive (best onset)",
    "best_restrain": "Restrain, then drive (best)",
}
REP_COLORS = {
    "passive": "#666666",
    "early_drive": "#c0392b",
    "best_drive": "#1f77b4",
    "best_restrain": "#2ca02c",
}


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    with (DATA_DIR / "e1_sweep.json").open(encoding="utf-8") as fh:
        sweep = json.load(fh)
    traces = dict(np.load(DATA_DIR / "representative_traces.npz"))
    return sweep, traces


def _impact_time(traces: dict[str, np.ndarray], label: str, summary: dict) -> float:
    rep = summary["representatives"][label]
    impact = rep.get("impact")
    return float(impact[0]) if impact else float(traces[f"{label}__t"][-1])


def fig_sweep(sweep: dict) -> None:
    """Clubhead speed at impact versus wrist-drive onset time."""
    rows = sweep["rows"]
    taus = sorted({r["shoulder_torque_nm"] for r in rows})
    fig, axes = plt.subplots(1, len(taus), figsize=(10.5, 4.0), sharey=False)
    for ax, tau in zip(np.atleast_1d(axes), taus, strict=False):
        passive = next(
            r
            for r in rows
            if r["profile"] == "passive" and r["shoulder_torque_nm"] == tau
        )
        if passive["clubhead_speed_mps"] is not None:
            ax.axhline(
                passive["clubhead_speed_mps"],
                color="#666666",
                ls="--",
                lw=1.2,
                label="passive baseline",
            )
        series: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            if r["shoulder_torque_nm"] != tau or r["onset_s"] is None:
                continue
            if r["clubhead_speed_mps"] is None:
                continue
            key = r["profile"]
            if key == "restrain_then_drive":
                key = f"restrain {r['wrist_restrain_nm']:.0f} N·m, then drive"
            elif key == "drive_only":
                key = "drive only"
            series.setdefault(key, []).append((r["onset_s"], r["clubhead_speed_mps"]))
        for key, pts in sorted(series.items()):
            pts.sort()
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, label=key)
        ax.set_title(f"Shoulder torque {tau:.0f} N·m")
        ax.set_xlabel("Wrist drive onset time [s]")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Clubhead speed at impact [m/s]")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "E1 — Timing of the distal handoff vs clubhead speed "
        "(2-DOF golf model, ODE backend)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_e1_onset_sweep.pdf")
    fig.savefig(FIG_DIR / "fig_e1_onset_sweep.svg")
    plt.close(fig)


def fig_kinematic_sequence(traces: dict[str, np.ndarray], summary: dict) -> None:
    """Arm and club angular speeds: the model's kinematic sequence."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=False)
    for ax, label in zip(axes.flat, REP_LABELS, strict=False):
        t = traces[f"{label}__t"]
        v = traces[f"{label}__v"]
        t_imp = _impact_time(traces, label, summary)
        mask = t <= t_imp + 0.02
        ax.plot(t[mask], np.abs(v[mask, 0]), label=r"$|\omega_{arm}|$", lw=1.6)
        ax.plot(
            t[mask],
            np.abs(v[mask, 0] + v[mask, 1]),
            label=r"$|\omega_{club}|$",
            lw=1.6,
        )
        ax.axvline(t_imp, color="k", ls=":", lw=1.0)
        ax.set_title(REP_LABELS[label], color=REP_COLORS[label])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular speed [rad/s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Kinematic sequence by torque program (dotted line: impact)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_kinematic_sequence.pdf")
    fig.savefig(FIG_DIR / "fig_kinematic_sequence.svg")
    plt.close(fig)


def fig_segment_energies(traces: dict[str, np.ndarray], summary: dict) -> None:
    """Segment kinetic-energy time courses for all representative swings."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for label in REP_LABELS:
        t = traces[f"{label}__t"]
        t_imp = _impact_time(traces, label, summary)
        mask = t <= t_imp
        axes[0].plot(
            t[mask],
            traces[f"{label}__e_arm"][mask],
            color=REP_COLORS[label],
            lw=1.5,
            label=REP_LABELS[label],
        )
        axes[1].plot(
            t[mask],
            traces[f"{label}__e_club"][mask],
            color=REP_COLORS[label],
            lw=1.5,
            label=REP_LABELS[label],
        )
    axes[0].set_title("Arm segment kinetic energy")
    axes[1].set_title("Club segment kinetic energy")
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kinetic energy [J]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Segmental kinetic energy up to impact")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_segment_energies.pdf")
    fig.savefig(FIG_DIR / "fig_segment_energies.svg")
    plt.close(fig)


def fig_wrist_power(traces: dict[str, np.ndarray], summary: dict) -> None:
    """Wrist-interface power accounting for the best restrain swing."""
    label = "best_restrain"
    t = traces[f"{label}__t"]
    t_imp = _impact_time(traces, label, summary)
    mask = t <= t_imp
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(
        t[mask],
        traces[f"{label}__power__joint_force_power"][mask],
        label="joint-force power",
        lw=1.5,
    )
    ax1.plot(
        t[mask],
        traces[f"{label}__power__moment_power_on_club"][mask],
        label="wrist moment power on club",
        lw=1.5,
    )
    ax1.plot(
        t[mask],
        traces[f"{label}__power__muscle_moment_power"][mask],
        label="wrist actuator power",
        lw=1.5,
        ls="--",
    )
    ax1.set_title(f"Interface powers — {REP_LABELS[label]}")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Power [W]")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    balance = (
        traces[f"{label}__power__joint_force_power"]
        + traces[f"{label}__power__moment_power_on_club"]
    )
    ax2.plot(
        t[mask],
        traces[f"{label}__power__club_energy_rate"][mask],
        label=r"$\dot{E}_{club}$ (analytic kinematics)",
        lw=1.5,
    )
    ax2.plot(
        t[mask],
        balance[mask],
        label="joint-force + moment power",
        lw=1.2,
        ls="--",
    )
    ax2.set_title("Segmental energy-balance check (Robertson–Winter)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Power [W]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_wrist_interface_power.pdf")
    fig.savefig(FIG_DIR / "fig_wrist_interface_power.svg")
    plt.close(fig)


def fig_ztcf_split(traces: dict[str, np.ndarray], summary: dict) -> None:
    """Drift vs control club angular acceleration (pointwise ZTCF split)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for ax, label in zip(axes, ("early_drive", "best_restrain"), strict=False):
        t = traces[f"{label}__t"]
        t_imp = _impact_time(traces, label, summary)
        mask = t <= t_imp
        drift = traces[f"{label}__drift"]
        control = traces[f"{label}__control"]
        alpha_drift = drift[:, 0] + drift[:, 1]
        alpha_ctrl = control[:, 0] + control[:, 1]
        ax.plot(t[mask], alpha_drift[mask], label="drift (ZTCF)", lw=1.5)
        ax.plot(t[mask], alpha_ctrl[mask], label="control", lw=1.5)
        ax.plot(
            t[mask],
            (alpha_drift + alpha_ctrl)[mask],
            label="total",
            lw=1.0,
            ls=":",
            color="k",
        )
        ax.set_title(REP_LABELS[label], color=REP_COLORS[label])
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Club angular acceleration [rad/s$^2$]")
    fig.suptitle(
        "E2 — Pointwise drift/control decomposition of club angular acceleration"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ztcf_drift_control.pdf")
    fig.savefig(FIG_DIR / "fig_ztcf_drift_control.svg")
    plt.close(fig)


def fig_energy_budget(summary: dict) -> None:
    """Early-half vs late-half club-energy and work budget bars."""
    labels = [lab for lab in REP_LABELS if summary["representatives"][lab]["budget"]]
    metrics = [
        ("club_ke_gain_early_j", "club KE gain, early half"),
        ("club_ke_gain_late_j", "club KE gain, late half"),
        ("wrist_actuator_work_early_j", "wrist actuator work, early"),
        ("wrist_actuator_work_late_j", "wrist actuator work, late"),
        ("joint_force_transfer_early_j", "joint-force transfer, early"),
        ("joint_force_transfer_late_j", "joint-force transfer, late"),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    for i, lab in enumerate(labels):
        budget = summary["representatives"][lab]["budget"]
        vals = [budget[m[0]] for m in metrics]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=REP_LABELS[lab],
            color=REP_COLORS[lab],
        )
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([m[1] for m in metrics], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Energy / work [J]")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("E2/E4 — Early-half vs late-half energy budget (to impact)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_energy_budget.pdf")
    fig.savefig(FIG_DIR / "fig_energy_budget.svg")
    plt.close(fig)


def fig_montage(traces: dict[str, np.ndarray], summary: dict) -> None:
    """Stick-figure montage of the best restrain-then-drive swing."""
    label = "best_restrain"
    t = traces[f"{label}__t"]
    q = traces[f"{label}__q"]
    t_imp = _impact_time(traces, label, summary)
    samples = np.linspace(0.0, t_imp, 9)
    l1, l2 = 0.75, 1.0
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    cmap = plt.get_cmap("viridis")
    for i, ts in enumerate(samples):
        k = int(np.argmin(np.abs(t - ts)))
        th1 = q[k, 0]
        thc = q[k, 0] + q[k, 1]
        hand = np.array([l1 * np.sin(th1), -l1 * np.cos(th1)])
        head = hand + np.array([l2 * np.sin(thc), -l2 * np.cos(thc)])
        color = cmap(i / (len(samples) - 1))
        ax.plot([0, hand[0]], [0, hand[1]], color=color, lw=2.0)
        ax.plot([hand[0], head[0]], [hand[1], head[1]], color=color, lw=1.4)
        ax.plot(*head, marker="o", ms=4, color=color)
    ax.set_aspect("equal")
    ax.set_title(f"{REP_LABELS[label]}: downswing progression in the swing plane")
    ax.set_xlabel("x [m] (target side)")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_swing_montage.pdf")
    fig.savefig(FIG_DIR / "fig_swing_montage.svg")
    plt.close(fig)


def main() -> None:
    """Render all report figures from the recorded experiment data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep, traces = _load()
    with (DATA_DIR / "results_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)
    fig_sweep(sweep)
    fig_kinematic_sequence(traces, summary)
    fig_segment_energies(traces, summary)
    fig_wrist_power(traces, summary)
    fig_ztcf_split(traces, summary)
    fig_energy_budget(summary)
    fig_montage(traces, summary)
    logger.info("wrote figures to %s", FIG_DIR)


if __name__ == "__main__":
    main()
