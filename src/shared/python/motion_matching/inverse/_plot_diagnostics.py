"""Matplotlib plotters for inverse-CVAE diagnostics (issue #4004 / #035).

Each function returns a :class:`matplotlib.figure.Figure` so callers can
either ``fig.savefig(...)`` or compose into a multi-panel report. We
deliberately do not call ``plt.show()`` here -- the diagnostics module
must remain headless (CI uses the ``Agg`` backend).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .diagnostics import CoverageMap, DiversityReport, LatentProjection

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "plot_coverage_map",
    "plot_diversity_report",
    "plot_latent_projection",
]


def _new_figure(figsize: tuple[float, float] = (6.0, 5.0)) -> Figure:
    # Local import keeps matplotlib optional at module-import time.
    import matplotlib.figure as _mfig

    return _mfig.Figure(figsize=figsize)


def plot_latent_projection(
    projection: LatentProjection,
    *,
    color_by: NDArray[np.float64] | None = None,
    title: str | None = None,
) -> Figure:
    """Scatter plot of the 2-D latent projection.

    Args:
        projection: :class:`LatentProjection` returned by
            :func:`latent_projection`.
        color_by: Optional ``(N,)`` ndarray of scalars (e.g. clubhead
            speed) used as the per-point colour.
        title: Optional title; defaults to a method/seed annotation.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    fig = _new_figure()
    ax = fig.add_subplot(111)
    coords = projection.coords
    if color_by is not None and len(color_by) != coords.shape[0]:
        raise ValueError(
            "color_by length must match projection.coords; got "
            f"{len(color_by)} vs {coords.shape[0]}"
        )
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_by if color_by is not None else None,
        cmap="viridis",
        s=18,
    )
    if color_by is not None:
        fig.colorbar(sc, ax=ax)
    ax.set_xlabel(f"{projection.method} dim 1")
    ax.set_ylabel(f"{projection.method} dim 2")
    ax.set_title(title or f"latent ({projection.method}, seed={projection.seed})")
    return fig


def plot_diversity_report(
    report: DiversityReport,
    *,
    title: str | None = None,
) -> Figure:
    """Histogram of pairwise distances for the diversity report.

    Args:
        report: :class:`DiversityReport` returned by :func:`sample_diversity`.
        title: Optional title; defaults to a mean/threshold annotation.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    fig = _new_figure()
    ax = fig.add_subplot(111)
    if report.pairwise_distances.size:
        ax.hist(report.pairwise_distances, bins=20, color="#4477AA", alpha=0.85)
    ax.axvline(
        report.threshold,
        color="red",
        linestyle="--",
        label=f"threshold={report.threshold:.2g}",
    )
    ax.axvline(
        report.mean_distance,
        color="green",
        linestyle=":",
        label=f"mean={report.mean_distance:.2g}",
    )
    ax.set_xlabel("pairwise L2")
    ax.set_ylabel("count")
    ax.set_title(title or f"sample diversity (collapsed={report.collapsed})")
    ax.legend(loc="best")
    return fig


def plot_coverage_map(
    coverage: CoverageMap,
    *,
    title: str | None = None,
) -> Figure:
    """Bar plot of per-trial round-trip RMSE with a flag-threshold line.

    Args:
        coverage: :class:`CoverageMap` returned by
            :func:`dataset_coverage_map`.
        title: Optional title; defaults to a count of flagged trials.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    fig = _new_figure(figsize=(8.0, 4.0))
    ax = fig.add_subplot(111)
    n = coverage.rmses_m.shape[0]
    xs = np.arange(n)
    colors = ["#CC3311" if f else "#228833" for f in coverage.flagged_mask]
    ax.bar(xs, coverage.rmses_m, color=colors)
    ax.axhline(
        coverage.flag_threshold_m,
        color="black",
        linestyle="--",
        label=f"flag={coverage.flag_threshold_m:.3g} m",
    )
    n_flagged = int(coverage.flagged_mask.sum())
    ax.set_xlabel("validation trial index")
    ax.set_ylabel("round-trip RMSE [m]")
    ax.set_title(title or f"dataset coverage ({n_flagged}/{n} flagged)")
    ax.legend(loc="best")
    return fig
