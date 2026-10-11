"""Traceable, one-page graphical summary of selected relative scoring."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SG = {
    "Driver": {
        "Draw": (0.00002691177, 0.007797655, "6 positive · 2 span zero"),
        "Fade": (-0.001460539, 0.000234387, "1 positive · 4 negative · 3 span zero"),
    },
    "Seven Iron": {
        "Draw": (-0.00135572, -0.0002882714, "8 negative"),
        "Fade": (0.0008933618, 0.002581634, "8 positive"),
    },
    "Pitching Wedge": {
        "Draw": (0.00009508286, 0.0111171, "6 positive · 2 span zero"),
        "Fade": (-0.01096597, 0.0002919228, "1 positive · 6 negative · 1 span zero"),
    },
}


def _verify_ranges(tex: Path) -> None:
    source = tex.read_text()
    for club, patterns in SG.items():
        table_club = "7-Iron" if club == "Seven Iron" else club
        for pattern, (low, high, _) in patterns.items():
            match = re.search(
                rf"^{re.escape(table_club)} & {pattern} & \\numtext\{{([^}}]+)\}} & \\numtext\{{([^}}]+)\}}",
                source,
                re.MULTILINE,
            )
            if match is None or not np.allclose(
                [float(match[1]), float(match[2])], [low, high], rtol=1e-8, atol=1e-12
            ):
                raise ValueError(
                    f"summary range differs from numerical table: {club} {pattern}"
                )


def _plot_ranges(fig: plt.Figure, ax: plt.Axes, colors: dict[str, str]) -> None:
    y = 0
    labels = []
    for club, patterns in SG.items():
        for pattern, (low, high, _) in patterns.items():
            color = colors[pattern]
            ax.plot([low, high], [y, y], color=color, lw=8, solid_capstyle="round")
            ax.plot([low, high], [y, y], "o", color=color, ms=5)
            label_club = (
                "7-Iron"
                if club == "Seven Iron"
                else "PW"
                if club == "Pitching Wedge"
                else club
            )
            labels.append(f"{label_club} · {pattern}")
            y += 1
    ax.axvline(0, color="#202b38", lw=1.2)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.013, 0.017)
    ax.set_xlabel(
        "Relative SG per Modeled Opportunity (Pattern − Straight)", fontsize=10
    )
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", alpha=0.18)


def _annotate_summary(fig: plt.Figure) -> None:
    fig.text(
        0.07,
        0.985,
        "No Universal Winning Shot Shape",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07, 0.925, "Selected Relative Scoring Ranges", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.07,
        0.895,
        "Minimum–Maximum Across Eight Selected Zero-Lean Conditions",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.222,
        "Distance Varies With Shape and Club. Closure Lowers Loft Here;",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.196,
        "carry direction depends on the full impact and flight response.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.162,
        "Relative SG effects are small and conditional on the scoring",
        fontsize=10,
    )
    fig.text(0.07, 0.136, "model. Straight is each cell's zero reference.", fontsize=10)
    fig.text(
        0.07,
        0.088,
        "These ranges are not pooled intervals or per-round gains.",
        fontsize=9,
        color="#536577",
    )
    fig.text(
        0.07,
        0.062,
        "Source: Selected relative-SG table; carry only; assumed lies",
        fontsize=9,
        color="#536577",
    )
    fig.text(
        0.07,
        0.040,
        "and historical benchmark. See numerical results for paired intervals.",
        fontsize=9,
        color="#536577",
    )


def render_summary(tex: Path, out: Path, save_figure, colors: dict[str, str]) -> Path:
    """Render selected ranges after checking every value against the table."""
    _verify_ranges(tex)
    fig, ax = plt.subplots(figsize=(7.0, 8.5))
    _plot_ranges(fig, ax, colors)
    _annotate_summary(fig)
    fig.subplots_adjust(left=0.28, right=0.95, top=0.83, bottom=0.32)
    path = out / "summary"
    save_figure(fig, path)
    return path


def render_social_summary(tex: Path, destination: Path, colors: dict[str, str]) -> Path:
    """Save a Facebook preview PNG after checking its ranges against the report."""
    _verify_ranges(tex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 6.3), dpi=100)
    _plot_ranges(fig, ax, colors)
    fig.text(
        0.055,
        0.95,
        "No Universal Winning Shot Shape",
        ha="left",
        va="top",
        fontsize=25,
        fontweight="bold",
        color="#153a52",
    )
    fig.text(
        0.058,
        0.87,
        "Selected Relative SG Ranges Across Eight Zero-Lean Conditions per Club",
        ha="left",
        va="top",
        fontsize=13,
        color="#34465a",
    )
    fig.text(
        0.055,
        0.075,
        "GEOMETRIC SIMULATION  ·  CONDITIONAL SG  ·  NOT PLAYER DATA",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#153a52",
    )
    fig.text(
        0.055,
        0.035,
        "Ranges are descriptive, not pooled intervals or per-round gains.",
        ha="left",
        va="center",
        fontsize=10,
        color="#536577",
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.78, bottom=0.22)
    fig.savefig(destination, dpi=100, facecolor="white")
    plt.close(fig)
    return destination
