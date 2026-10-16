"""Static matplotlib (Agg) figures for spine reports (SPN-15a, #12771).

Figures are built from :mod:`src.shared.python.spine.viz_data` products only.
They use the object-oriented ``Figure`` API with an Agg canvas, never
``pyplot``, so nothing touches a GUI backend or global state. Every title
states the scientific status; unqualified data is never drawn without the word
``UNQUALIFIED``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.shared.python.spine.viz_data import HeatmapData, StackData

__all__ = ["plot_angle_stack", "plot_load_heatmap", "save_png"]

_TERM_TITLES = {
    "x_factor_jcs_axial": "Axial Rotation",
    "trunk_extension_jcs": "Trunk Extension",
    "trunk_lateral_flexion_jcs": "Trunk Lateral Flexion",
}


def _canvas_figure(size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def plot_angle_stack(data: StackData) -> Figure:
    """Stacked per-level contributions over time, caudal level first.

    Positive and negative shares stack separately so signed contributions stay
    readable. The y-axis is degrees; the title carries the term and the status.
    """
    if not isinstance(data, StackData):
        raise TypeError(f"data must be a StackData, got {type(data).__name__}")
    figure = _canvas_figure((8.0, 4.5))
    ax = figure.add_subplot()
    values = np.nan_to_num(np.asarray(data.values_deg), nan=0.0)
    base_pos = np.zeros(values.shape[0])
    base_neg = np.zeros(values.shape[0])
    for k, level in enumerate(data.levels):
        col = values[:, k]
        base = np.where(col >= 0, base_pos, base_neg)
        ax.bar(
            data.time_s,
            col,
            bottom=base,
            width=_bar_width(data.time_s),
            label=level.name,
        )
        base_pos = base_pos + np.where(col >= 0, col, 0.0)
        base_neg = base_neg + np.where(col < 0, col, 0.0)
    term = _TERM_TITLES.get(data.term_id, data.term_id)
    ax.set_title(f"Per-Level {term} Contributions ({data.status_label()})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contribution (deg)")
    ax.legend(title="Level (Caudal First)", fontsize="small", ncols=2)
    ax.text(
        0.01,
        0.01,
        f"{data.status_label()} | source: {data.distribution_source}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    return figure


def _bar_width(time_s: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=float)
    return float(np.min(np.diff(t))) * 0.9 if t.size > 1 else 1.0


def plot_load_heatmap(data: HeatmapData) -> Figure:
    """Time-by-level heatmap of one load component with a signed colour scale.

    Levels run caudal (bottom) to cranial (top). The colour bar label carries
    the quantity, sign convention and units.
    """
    if not isinstance(data, HeatmapData):
        raise TypeError(f"data must be a HeatmapData, got {type(data).__name__}")
    figure = _canvas_figure((8.0, 4.5))
    ax = figure.add_subplot()
    values = np.asarray(data.values)
    limit = float(np.max(np.abs(values))) or 1.0
    t = np.asarray(data.time_s)
    pad = _bar_width(t) / 1.8
    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=(t[0] - pad, t[-1] + pad, -0.5, len(data.levels) - 0.5),
    )
    ax.set_yticks(range(len(data.levels)))
    ax.set_yticklabels([lv.name for lv in data.levels])
    quantity = data.quantity.replace("_", " ").title()
    ax.set_title(f"Per-Level {quantity} ({data.status_label()})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Level")
    bar = figure.colorbar(image, ax=ax)
    bar.set_label(f"{quantity} ({data.units}), {data.sign_convention}")
    ax.text(
        0.01,
        0.01,
        f"{data.observable} | source: {data.distribution_source}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    return figure


def save_png(figure: Figure, path: str | Path) -> Path:
    """Write ``figure`` to ``path`` (must end in ``.png``) and return the path.

    Raises:
        ValueError: if the suffix is not ``.png``.
    """
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError(f"path must end in .png, got {target.name!r}")
    figure.savefig(target, dpi=150)
    return target
