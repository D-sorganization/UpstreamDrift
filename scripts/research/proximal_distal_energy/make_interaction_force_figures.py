"""Render publication figures for the interaction-force mechanism study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"

COLORS = {
    "total": "#111827",
    "drift": "#2563eb",
    "control": "#dc2626",
    "proximal_tangential": "#7c3aed",
    "proximal_centripetal": "#0f766e",
    "distal_tangential": "#ea580c",
    "distal_centripetal": "#0284c7",
    "gravity_reaction": "#64748b",
}
LABELS = {
    "proximal_tangential": "Proximal Tangential",
    "proximal_centripetal": "Proximal Centripetal",
    "distal_tangential": "Distal Tangential",
    "distal_centripetal": "Distal Centripetal",
    "gravity_reaction": "Gravity Reaction",
}


def _load() -> tuple[dict[str, np.ndarray], dict]:
    arrays = dict(np.load(DATA_DIR / "interaction_force_mechanisms.npz"))
    with (DATA_DIR / "interaction_force_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    return arrays, summary


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _arrow(
    ax: plt.Axes, origin: np.ndarray, vector: np.ndarray, color: str, label: str = ""
) -> None:
    ax.annotate(
        "",
        xy=origin + vector,
        xytext=origin,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8},
    )
    if label:
        ax.text(*(origin + 1.08 * vector), label, color=color, fontsize=8)


def fig_free_body() -> None:
    """Draw the declared planar coordinate and club free-body convention."""
    theta1, theta2 = -0.65, -0.9
    phi = theta1 + theta2
    hand = 0.75 * np.array([np.sin(theta1), -np.cos(theta1)])
    head = hand + np.array([np.sin(phi), -np.cos(phi)])
    com = hand + 0.756 * np.array([np.sin(phi), -np.cos(phi)])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, hand[0]], [0, hand[1]], lw=6, color="#94a3b8", solid_capstyle="round")
    ax.plot([hand[0], head[0]], [hand[1], head[1]], lw=3, color="#334155")
    ax.scatter(*hand, s=45, color="#111827", zorder=4)
    ax.scatter(*com, s=45, color="#f59e0b", zorder=4)
    ax.scatter(*head, s=70, color="#111827", zorder=4)
    _arrow(ax, hand, np.array([0.42, 0.24]), COLORS["total"], r"$\mathbf{F}_W$")
    _arrow(ax, hand, np.array([-0.13, 0.42]), COLORS["control"], r"$\tau_W$")
    _arrow(
        ax, com, np.array([0.0, -0.38]), COLORS["gravity_reaction"], r"$m_2\mathbf{g}$"
    )
    _arrow(ax, hand, np.array([0.35, 0.0]), "#475569", r"$\mathbf{v}_W$")
    ax.text(-1.46, -0.55, "Fixed Hub")
    ax.text(hand[0] + 0.03, hand[1] - 0.08, "Wrist")
    ax.text(com[0] + 0.03, com[1] + 0.03, "Club COM")
    ax.set_title("Double-Pendulum Club Free-Body Diagram")
    ax.set_xlabel("Target-Side Coordinate, x [m]")
    ax.set_ylabel("Vertical Swing-Plane Coordinate, y [m]")
    ax.set_xlim(-1.55, 0.12)
    ax.set_ylim(-1.08, 0.12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    _save(fig, "fig_interaction_free_body")


def fig_vector_montage(arrays: dict[str, np.ndarray], summary: dict) -> None:
    """Show total, zero-torque drift, and control forces over six phases."""
    t, q = arrays["t"], arrays["q"]
    t_imp = summary["impact"]["time_s"]
    sample_times = np.linspace(0.08 * t_imp, 0.98 * t_imp, 6)
    force_scale = 0.0017
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2))
    for ax, sample_time in zip(axes.flat, sample_times, strict=True):
        k = int(np.argmin(np.abs(t - sample_time)))
        theta1, phi = q[k, 0], q[k].sum()
        hand = 0.75 * np.array([np.sin(theta1), -np.cos(theta1)])
        head = hand + np.array([np.sin(phi), -np.cos(phi)])
        ax.plot([0, hand[0]], [0, hand[1]], lw=4, color="#94a3b8")
        ax.plot([hand[0], head[0]], [hand[1], head[1]], lw=2.2, color="#334155")
        for key in (
            "force_total",
            "force_drift",
            "force_control",
        ):
            color = COLORS[
                {
                    "force_total": "total",
                    "force_drift": "drift",
                    "force_control": "control",
                }[key]
            ]
            _arrow(ax, hand, force_scale * arrays[key][k], color)
        ax.set_title(f"{100 * sample_time / t_imp:.0f}% of Time to Impact")
        ax.set_xlim(-1.55, 1.05)
        ax.set_ylim(-1.65, 0.55)
        ax.set_aspect("equal")
        ax.grid(alpha=0.16)
    fig.suptitle("Wrist Reaction-Force Vectors Through the Downswing")
    handles = [
        plt.Line2D([0], [0], color=COLORS["total"], lw=2, label="Total"),
        plt.Line2D([0], [0], color=COLORS["drift"], lw=2, label="Pointwise ZTCF"),
        plt.Line2D([0], [0], color=COLORS["control"], lw=2, label="Control"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.10)
    _save(fig, "fig_interaction_vector_montage")


def fig_force_components(arrays: dict[str, np.ndarray], summary: dict) -> None:
    """Plot exact vector-component magnitudes and total Cartesian force."""
    t = arrays["t"]
    mask = t <= summary["impact"]["time_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, label in LABELS.items():
        magnitude = np.linalg.norm(arrays[f"force_component__{name}"], axis=1)
        axes[0].plot(t[mask], magnitude[mask], color=COLORS[name], label=label)
    axes[0].set_title("Exact Inertial Component Magnitudes")
    axes[0].set_ylabel("Force Magnitude [N]")
    axes[0].legend(fontsize=7)
    axes[1].plot(t[mask], arrays["force_total"][mask, 0], label="Target-Side Component")
    axes[1].plot(t[mask], arrays["force_total"][mask, 1], label="Vertical Component")
    axes[1].plot(
        t[mask],
        np.linalg.norm(arrays["force_total"][mask], axis=1),
        color="k",
        ls="--",
        label="Magnitude",
    )
    axes[1].set_title("Total Wrist Force on the Club")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.25)
    _save(fig, "fig_interaction_force_components")


def fig_power_components(arrays: dict[str, np.ndarray], summary: dict) -> None:
    """Plot component force powers and cumulative transmitted work."""
    t = arrays["t"]
    mask = t <= summary["impact"]["time_s"]
    t_c = t[mask]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, label in LABELS.items():
        axes[0].plot(
            t_c,
            arrays[f"power_component__{name}"][mask],
            color=COLORS[name],
            label=label,
        )
    axes[0].plot(t_c, arrays["power_total"][mask], color="k", lw=2.0, label="Total")
    axes[0].axhline(0.0, color="#64748b", lw=0.8)
    axes[0].set_title("Force-Power Pathways at the Moving Wrist")
    axes[0].set_ylabel("Power Into the Club [W]")
    axes[0].legend(fontsize=7)
    dt = np.diff(t_c)
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (arrays["power_total"][mask][1:] + arrays["power_total"][mask][:-1])
                * dt
            ),
        )
    )
    cumulative_drift = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (arrays["power_drift"][mask][1:] + arrays["power_drift"][mask][:-1])
                * dt
            ),
        )
    )
    axes[1].plot(t_c, cumulative, color=COLORS["total"], lw=2, label="Total")
    axes[1].plot(
        t_c, cumulative_drift, color=COLORS["drift"], lw=1.7, label="Pointwise ZTCF"
    )
    axes[1].set_title("Cumulative Work Transmitted by Wrist Force")
    axes[1].set_ylabel("Work Into the Club [J]")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.25)
    _save(fig, "fig_interaction_force_power")


def fig_geometry(arrays: dict[str, np.ndarray]) -> None:
    """Plot exact orientation coefficients for distal force-power terms."""
    angle = np.rad2deg(arrays["theta2_grid"])
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(
        angle,
        arrays["geometry_distal_tangential"],
        lw=2,
        label=r"Tangential Projection, $\cos\theta_2$",
    )
    ax.plot(
        angle,
        arrays["geometry_distal_centripetal"],
        lw=2,
        label=r"Centripetal Projection, $-\sin\theta_2$",
    )
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.axvline(-90, color="#94a3b8", ls=":", label="Nominal 90° Wrist Cock")
    ax.axvline(0, color="#94a3b8", ls="--", label="Links Aligned")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel(r"Relative Wrist Angle, $\theta_2$ [deg]")
    ax.set_ylabel("Projection Onto Hand-Velocity Direction")
    ax.set_title("Geometry Controls Which Inertial Forces Can Transfer Power")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "fig_interaction_geometry_coefficients")


def fig_killswitch(arrays: dict[str, np.ndarray], summary: dict) -> None:
    """Show trajectory divergence after a late-downswing torque killswitch."""
    t = arrays["killswitch_t"]
    q_cmd, q_zero = arrays["killswitch_commanded_q"], arrays["killswitch_zero_q"]
    v_cmd, v_zero = arrays["killswitch_commanded_v"], arrays["killswitch_zero_v"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(t, q_cmd.sum(axis=1), label="Commanded", lw=2)
    axes[0].plot(t, q_zero.sum(axis=1), label="Zero-Torque Future", lw=2, ls="--")
    axes[0].set_title("Club Absolute Angle After the Matched State")
    axes[0].set_ylabel("Angle [rad]")
    axes[1].plot(t, v_cmd.sum(axis=1), label="Commanded", lw=2)
    axes[1].plot(t, v_zero.sum(axis=1), label="Zero-Torque Future", lw=2, ls="--")
    axes[1].set_title("Club Absolute Angular Velocity After the Matched State")
    axes[1].set_ylabel("Angular Velocity [rad/s]")
    for ax in axes:
        ax.axvline(summary["killswitch"]["cut_time_s"], color="k", ls=":")
        ax.set_xlabel("Source-Trace Time [s]")
        ax.grid(alpha=0.25)
        ax.legend()
    _save(fig, "fig_interaction_killswitch")


def fig_wscg_source_series() -> None:
    """Redraw the registered WSCG BASE and counterfactual hand-force series."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    with (DATA_DIR / "wscg_2024_hand_force_series.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["series"], []).append(
                (float(row["time_s"]), float(row["value"]))
            )
    pairs = (
        ("LeadHandAxial", "LeadHandCFAxial", "Lead Hand Axial"),
        ("LeadHandNormal", "LeadHandCFNormal", "Lead Hand Normal"),
        ("TrailHeadAxial", "TrailHandCFAxial", "Trail Hand Axial"),
        ("TrailHandNormal", "TrailHandCFNormal", "Trail Hand Normal"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for ax, (base, counterfactual, title) in zip(axes.flat, pairs, strict=True):
        xb, yb = zip(*grouped[base], strict=True)
        xc, yc = zip(*grouped[counterfactual], strict=True)
        ax.plot(xb, yb, lw=1.7, label="BASE")
        ax.plot(xc, yc, lw=1.7, ls="--", label="Counterfactual")
        ax.set_title(title)
        ax.set_ylabel("Force [N]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Registered WSCG Two-Hand Force Series")
    _save(fig, "fig_wscg_registered_hand_forces")


def main() -> None:
    """Render all interaction-force figures in PDF and SVG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arrays, summary = _load()
    fig_free_body()
    fig_vector_montage(arrays, summary)
    fig_force_components(arrays, summary)
    fig_power_components(arrays, summary)
    fig_geometry(arrays)
    fig_killswitch(arrays, summary)
    fig_wscg_source_series()


if __name__ == "__main__":
    main()
