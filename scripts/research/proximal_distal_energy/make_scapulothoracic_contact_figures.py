"""Render the paired scapulothoracic contact-screen figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .scapulothoracic_contact_screen import (
    ScapulothoracicConfig,
    ellipsoid_surface_point,
)

matplotlib.rcParams["svg.hashsalt"] = "proximal-distal-scapulothoracic-v1"
ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/scapulothoracic_contact_screen.json"
ARRAYS = ARTICLE / "data/scapulothoracic_contact_screen.npz"
FIGURE = ARTICLE / "figures/fig_scapulothoracic_contact_screen"
INK = "#132238"
BLUE = "#2C6EAA"
ORANGE = "#D97706"
GREEN = "#2A7F62"
RED = "#B23A48"


def _save(figure: plt.Figure) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    svg = FIGURE.with_suffix(".svg")
    figure.savefig(svg, bbox_inches="tight")
    normalized = "\n".join(line.rstrip() for line in svg.read_text().splitlines())
    svg.write_text(normalized + "\n", encoding="utf-8")
    plt.close(figure)


def _ellipsoid_panel(axis: plt.Axes, arrays: np.lib.npyio.NpzFile) -> None:
    config = ScapulothoracicConfig()
    latitude = np.linspace(-0.15, 1.25, 240)
    b, c = config.ellipsoid_radii_m[1:]
    axis.plot(b * np.cos(latitude), c * np.sin(latitude), color=INK, lw=1.8)
    axis.plot(-b * np.cos(latitude), c * np.sin(latitude), color=INK, lw=1.8)
    neutral_lead = ellipsoid_surface_point(config, "lead", 0.0, 0.0, linear_scale=1.0)
    neutral_trail = ellipsoid_surface_point(config, "trail", 0.0, 0.0, linear_scale=1.0)
    coordinates = arrays["scapular_coordinates_rad"]
    for side_index, (side, neutral) in enumerate(
        (("lead", neutral_lead), ("trail", neutral_trail))
    ):
        points = np.asarray(
            [
                ellipsoid_surface_point(
                    config,
                    side,
                    float(values[0]),
                    float(values[1]),
                    linear_scale=1.0,
                )
                for values in coordinates[..., side_index, :].reshape(-1, 4)
            ]
        )
        axis.scatter(points[:, 1], points[:, 2], s=14, alpha=0.45, color=BLUE)
        axis.scatter(neutral[1], neutral[2], marker="x", s=60, color=ORANGE, zorder=4)
    axis.set_aspect("equal")
    axis.set_xlabel("Thorax-Lateral Coordinate (m)")
    axis.set_ylabel("Thorax-Vertical Coordinate (m)")
    axis.set_title("A  Shoulder-Center Envelope")
    axis.text(
        0.02,
        0.02,
        "Crosses: fixed centers\nDots: solved ellipsoid centers",
        transform=axis.transAxes,
        fontsize=8,
        color=INK,
    )


def _closure_panel(
    axis: plt.Axes, arrays: np.lib.npyio.NpzFile, tolerance: float
) -> None:
    fixed = arrays["fixed_max_contact_error_m"].ravel()
    scapular = arrays["scapular_max_contact_error_m"].ravel()
    qualified = (scapular <= tolerance) & arrays[
        "scapular_solver_termination_success"
    ].ravel()
    order = np.argsort(fixed)
    index = np.arange(fixed.size)
    axis.semilogy(index, fixed[order], color=ORANGE, lw=2.0, label="Fixed Shoulder")
    axis.semilogy(index, scapular[order], color=BLUE, lw=2.0, label="Mobile Scapula")
    qualified_indices = index[qualified[order]]
    axis.scatter(
        qualified_indices,
        scapular[order][qualified[order]],
        color=GREEN,
        s=18,
        label="Closed + Solver Success",
        zorder=4,
    )
    axis.axhline(tolerance, color=RED, ls="--", lw=1.3, label="Closure Tolerance")
    axis.set_xlabel("Registered State (Sorted by Fixed-Shoulder Error)")
    axis.set_ylabel("Maximum Bilateral Contact Error (m)")
    axis.set_title("B  Paired Contact Residual")
    axis.legend(loc="upper left", frameon=True, framealpha=0.88, fontsize=7.5)
    axis.grid(alpha=0.2, which="both")


def _boundary_panel(
    axis: plt.Axes, arrays: np.lib.npyio.NpzFile, tolerance: float
) -> None:
    margin = arrays["scapular_minimum_bound_margin_rad"].ravel()
    error = arrays["scapular_max_contact_error_m"].ravel()
    termination = arrays["scapular_solver_termination_success"].ravel()
    colors = np.where(termination, GREEN, RED)
    axis.scatter(margin, error, c=colors, s=24, alpha=0.75)
    axis.axhline(tolerance, color=RED, ls="--", lw=1.3)
    axis.axvline(0.0, color=INK, ls=":", lw=1.2)
    axis.set_yscale("log")
    axis.set_xlabel("Minimum Coordinate-Bound Margin (rad)")
    axis.set_ylabel("Maximum Bilateral Contact Error (m)")
    axis.set_title("C  Closure, Bounds, and Termination")
    axis.grid(alpha=0.2, which="both")
    axis.text(
        0.98,
        0.97,
        "Green: successful termination\nRed: residual retained without success flag",
        transform=axis.transAxes,
        fontsize=8,
        color=INK,
        horizontalalignment="right",
        verticalalignment="top",
    )


def main() -> None:
    """Render the single governed figure from committed data."""
    record = json.loads(DATA.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS)
    tolerance = float(record["model"]["config"]["closure_tolerance_m"])
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    _ellipsoid_panel(axes[0], arrays)
    _closure_panel(axes[1], arrays, tolerance)
    _boundary_panel(axes[2], arrays, tolerance)
    figure.suptitle(
        "Scapular Mobility Expands Geometric Reach but Does Not Establish Anatomy",
        fontsize=13,
        color=INK,
    )
    _save(figure)
    print(FIGURE.with_suffix(".pdf"))
    print(FIGURE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
