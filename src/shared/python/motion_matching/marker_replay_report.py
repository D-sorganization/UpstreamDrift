"""Observed-marker diagnostics for actual continuous forward trajectories."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

Array = NDArray[np.float64]


def marker_errors(target: Array, prediction: Array, valid: NDArray[np.bool_]) -> Array:
    """Return Euclidean errors in metres, retaining unobserved entries as NaN."""
    measured, replay = np.asarray(target), np.asarray(prediction)
    observed = np.asarray(valid)
    if (
        measured.ndim != 3
        or measured.shape[-1] != 3
        or measured.shape != replay.shape
        or observed.shape != measured.shape[:2]
        or observed.dtype != np.bool_
        or not measured.size
    ):
        raise ValueError("Expected matching (time, markers, 3) arrays and boolean mask")
    if not np.isfinite(replay).all() or not np.isfinite(measured[observed]).all():
        raise ValueError("Replay and observed target coordinates must be finite")
    errors = np.full(observed.shape, np.nan)
    errors[observed] = np.linalg.norm(replay[observed] - measured[observed], axis=1)
    return errors


def plot_marker_replay(
    time_s: Array,
    target: Array,
    prediction: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    *,
    candidate_sha256: str,
) -> "Figure":
    """Plot observed RMS over time and terminal per-marker errors; no acceptance."""
    errors = marker_errors(target, prediction, valid)
    clock = np.asarray(time_s)
    if (
        clock.shape != (errors.shape[0],)
        or not np.isfinite(clock).all()
        or np.any(np.diff(clock) <= 0)
        or len(labels) != errors.shape[1]
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Clock and unique marker labels must match the replay")
    if len(candidate_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in candidate_sha256
    ):
        raise ValueError("Expected canonical candidate SHA256")
    import matplotlib.pyplot as plt

    count = np.sum(np.isfinite(errors), axis=1)
    rms = np.full(clock.shape, np.nan)
    np.divide(np.nansum(errors**2, axis=1), count, out=rms, where=count > 0)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    axes[0].plot(clock, 1000 * np.sqrt(rms), linewidth=2, label="Observed Marker RMS")
    axes[0].plot(clock, 1000 * errors, color="gray", alpha=0.2, linewidth=0.7)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Position Error (mm)",
        title="Continuous Forward Replay",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    missing = count < errors.shape[1]
    axes[0].fill_between(
        clock,
        0,
        1,
        where=missing,
        transform=axes[0].get_xaxis_transform(),
        color="orange",
        alpha=0.12,
    )
    axes[1].barh(labels, 1000 * errors[-1])
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Position Error (mm)", title=f"Terminal Errors at {clock[-1]:.3f} s"
    )
    axes[1].tick_params(axis="y", labelsize=8)
    fig.suptitle(
        f"Experimental Candidate {candidate_sha256[:12]} — Acceptance Not Established"
    )
    fig.text(
        0.02,
        0.01,
        "Gray: Individual Markers. Orange: Incomplete Observation Coverage. Missing Terminal Bars: Unobserved.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig
