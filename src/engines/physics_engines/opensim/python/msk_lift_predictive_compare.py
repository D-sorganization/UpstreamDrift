"""Predictive-versus-tracked deadlift comparison (OSL-7, #13044).

Bar path, joint angles, activations and L5/S1 load of a predictive Moco
deadlift against the tracked OSL-3/OSL-4 solution. The predictive duration
is free and the tracked solution covers only the bundle's 0.2 s window, so
samples align by bar-rise fraction ``(h - h_start) / (h_final - h_start)``
(each solution's own ends). A quantity either side lacks is ``unavailable``
with a reason, never zero. Method: ``lift_models.tex``, OSL-7 section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
    _summarize_groups,
)
from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
    _processed_model,
    _replay_states,
)
from src.shared.python.contracts import require
from src.shared.python.motion_matching.same_input import InputBundle

_EVIDENCE = Path(__file__).resolve().parents[5] / "docs/development/lifting/evidence"

#: The tracked solutions' inputs: the OSL-3 barbell bundle and receipts.
TRACKED_BUNDLE = _EVIDENCE / "deadlift/deadlift_barbell_bundle.npz"
OSL4_MUSCLE_RECEIPT = _EVIDENCE / "opensim_muscles/receipts.json"
OSL5_JOINT_REACTION_RECEIPT = _EVIDENCE / "opensim_joint_reactions/receipts.json"

ALIGNMENT = "bar-rise fraction (each solution's own start and final bar height)"

#: Tracked joint angles exist only at the window start (OSL-4 ``start_pose``);
#: the OSL-3 receipt keeps IK summaries, not coordinate trajectories.
START_POSE_COORDINATES = (
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
)


@dataclass(frozen=True)
class Trajectory:
    """Samples of one quantity against bar-rise fraction."""

    progress: np.ndarray
    values: np.ndarray
    unit: str

    def __post_init__(self) -> None:
        require(self.progress.ndim == 1, "progress must be one-dimensional")
        require(len(self.values) == len(self.progress), "length mismatch")
        rising = bool(np.all(np.diff(self.progress) > 0.0))
        require(rising, "progress must be strictly increasing")


def unavailable(reason: str) -> dict[str, Any]:
    """An unavailable comparison entry; the reason is mandatory."""
    require(bool(reason), "an unavailable entry needs a reason")
    return {"status": "unavailable", "reason": reason}


def bar_rise_fraction(
    bar_height_m: np.ndarray,
    start_height_m: float | None = None,
    final_height_m: float | None = None,
) -> np.ndarray:
    """``(h - h_start) / (h_final - h_start)``; the ends default to the samples'."""
    h = np.asarray(bar_height_m, dtype=np.float64)
    h0 = float(h[0]) if start_height_m is None else float(start_height_m)
    h1 = float(h[-1]) if final_height_m is None else float(final_height_m)
    require(h1 - h0 > 0.0, f"bar must rise: start {h0} m, final {h1} m")
    return (h - h0) / (h1 - h0)


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """The bar-rise interval both progress axes cover, or ``None``."""
    lo, hi = max(float(a[0]), float(b[0])), min(float(a[-1]), float(b[-1]))
    return (lo, hi) if hi > lo else None


def compare_trajectories(predictive: Trajectory, tracked: Trajectory) -> dict[str, Any]:
    """Difference of the predictive trajectory at the tracked samples.

    The predictive values are linearly interpolated onto the tracked samples
    that lie inside the shared bar-rise interval; differences are predictive
    minus tracked.
    """
    require(predictive.unit == tracked.unit, "units differ")
    window = overlap(predictive.progress, tracked.progress)
    if window is None:
        return unavailable("the solutions share no bar-rise interval")
    inside = (tracked.progress >= window[0]) & (tracked.progress <= window[1])
    if int(np.count_nonzero(inside)) < 2:
        return unavailable("fewer than two tracked samples in the shared interval")
    x = tracked.progress[inside]
    ref = tracked.values[inside]
    pred = np.interp(x, predictive.progress, predictive.values)
    diff = pred - ref
    return {
        "status": "compared",
        "alignment": ALIGNMENT,
        "bar_rise_window": [window[0], window[1]],
        "samples": int(x.size),
        "rms_difference": float(np.sqrt(np.mean(diff**2))),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "predictive_mean": float(np.mean(pred)),
        "tracked_mean": float(np.mean(ref)),
        "unit": tracked.unit,
    }


def compare_point(
    predictive: Trajectory, tracked_value: float | None, at_progress: float
) -> dict[str, Any]:
    """The predictive value at one bar-rise fraction against a tracked value."""
    if tracked_value is None:
        return unavailable("no tracked value")
    lo, hi = float(predictive.progress[0]), float(predictive.progress[-1])
    if not lo <= at_progress <= hi:
        return unavailable(f"fraction {at_progress:.3f} outside [{lo:.3f}, {hi:.3f}]")
    pred = float(np.interp(at_progress, predictive.progress, predictive.values))
    return {
        "status": "compared",
        "alignment": ALIGNMENT,
        "bar_rise_fraction": at_progress,
        "predictive": pred,
        "tracked": float(tracked_value),
        "difference": pred - float(tracked_value),
        "unit": predictive.unit,
    }


def compare_group_activations(
    progress: np.ndarray,
    activations: dict[str, np.ndarray],
    tracked_groups: dict[str, Any],
    window: tuple[float, float],
) -> dict[str, Any]:
    """Peak and mean activation per major group over the tracked window.

    The predictive statistics use the OSL-4 definition (``_summarize_groups``:
    peak and mean over every constituent muscle and sample), restricted to the
    predictive samples inside ``window``.
    """
    inside = (progress >= window[0]) & (progress <= window[1])
    if not np.any(inside):
        return {"_all": unavailable("no predictive sample in the tracked window")}
    names = list(activations)
    act = np.column_stack([activations[n][inside] for n in names])
    summary = _summarize_groups(act, np.zeros_like(act), names)
    out: dict[str, Any] = {}
    for group, ref in tracked_groups.items():
        pred = summary.get(group, {})
        if ref.get("peak_activation") is None or pred.get("peak_activation") is None:
            out[group] = unavailable(f"group {group!r} has no muscles on one side")
            continue
        out[group] = {
            "status": "compared",
            "predictive_peak": pred["peak_activation"],
            "tracked_peak": ref["peak_activation"],
            "predictive_mean": pred["mean_activation"],
            "tracked_mean": ref["mean_activation"],
            "mean_difference": pred["mean_activation"] - ref["mean_activation"],
        }
    return out


def _deadlift_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))["lifts"]["deadlift"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def load_tracked_reference(
    bundle_path: Path = TRACKED_BUNDLE,
    muscle_receipt: Path = OSL4_MUSCLE_RECEIPT,
    joint_reaction_receipt: Path = OSL5_JOINT_REACTION_RECEIPT,
) -> dict[str, Any]:
    """The tracked deadlift quantities, each present or with a reason.

    Bar path: the bundle's window bar path (the OSL-3 IK target), as
    bar-rise fraction of the bundle's full-lift bar path.
    """
    prov = InputBundle.load(Path(bundle_path)).provenance
    full_z = np.asarray(prov["ik_bar_path_m"], dtype=np.float64)[:, 2]
    window_z = np.asarray(prov["window_bar_path_m"], dtype=np.float64)[:, 2]
    progress = bar_rise_fraction(window_z, float(full_z[0]), float(full_z[-1]))
    tracked: dict[str, Any] = {
        "bar": Trajectory(progress, window_z - float(full_z[0]), "m"),
        "window": (float(progress[0]), float(progress[-1])),
    }
    muscles = _deadlift_record(Path(muscle_receipt))
    if muscles is None or muscles.get("status") != "complete":
        reason = "OSL-4 deadlift muscle receipt missing or not complete"
        tracked["muscles_reason"] = reason
    else:
        tracked["start_pose_deg"] = muscles.get("start_pose", {})
        tracked["groups"] = muscles.get("major_groups", {})
        tracked["muscle_classification"] = muscles.get("classification")
        tracked["muscle_solver"] = muscles.get("moco", {}).get("status")
    reactions = _deadlift_record(Path(joint_reaction_receipt)) or {}
    if reactions.get("status") != "complete":
        tracked["l5_s1_reason"] = (
            f"OSL-5 deadlift joint reactions {reactions.get('status')!r}: "
            f"{reactions.get('reason', 'receipt unreadable')}"
        )
    else:
        tracked["l5_s1"] = reactions.get("l5_s1", {})
    return tracked


def predictive_trajectories(study: Any, solution: Any) -> dict[str, Any]:
    """The :func:`compare_with_tracked` input from a Moco solution.

    The bar height is the barbell body origin (ground Y) replayed on the
    processed model; coordinates and activations are the states columns.
    """
    rep, model = _processed_model(study)
    states_table = solution.exportToStatesTable()
    _times, bar, _com = _replay_states(model, states_table)
    del rep  # kept alive until the replay on its model copy is done
    labels = list(states_table.getColumnLabels())
    data = states_table.getMatrix().to_numpy()
    coords: dict[str, np.ndarray] = {}
    acts: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        parts = label.split("/")
        if label.endswith("/value"):
            coords[parts[-2]] = np.asarray(data[:, i], dtype=np.float64)
        elif label.endswith("/activation"):
            acts[parts[-2]] = np.asarray(data[:, i], dtype=np.float64)
    return {"bar_height_m": bar[:, 1], "coordinates": coords, "activations": acts}


def compare_with_tracked(
    predictive: dict[str, Any], tracked: dict[str, Any]
) -> dict[str, Any]:
    """Compare predictive trajectories with :func:`load_tracked_reference`.

    ``predictive`` holds ``bar_height_m`` (vertical, metres), ``coordinates``
    (name to radians) and optional ``activations`` (muscle name to
    activation), all sampled on the same grid. The predictive solution does
    not compute an L5/S1 load yet, so that entry is always unavailable.
    """
    height = np.asarray(predictive["bar_height_m"], dtype=np.float64)
    if not bool(np.all(np.diff(height) > 0.0)):
        reason = "predictive bar height is not strictly rising; alignment undefined"
        return {"status": "unavailable", "reason": reason}
    progress = bar_rise_fraction(height)
    result: dict[str, Any] = {"status": "compared", "alignment": ALIGNMENT}
    result["bar_path"] = compare_trajectories(
        Trajectory(progress, height - height[0], "m"), tracked["bar"]
    )
    window = tracked["window"]
    if "start_pose_deg" in tracked:
        angles = {}
        for coord in START_POSE_COORDINATES:
            q = predictive.get("coordinates", {}).get(coord)
            ref = tracked["start_pose_deg"].get(f"{coord}_deg")
            angles[coord] = (
                unavailable(f"predictive solution has no {coord}")
                if q is None
                else compare_point(
                    Trajectory(progress, np.degrees(np.asarray(q)), "deg"),
                    ref,
                    window[0],
                )
            )
        result["joint_angles"] = angles
        acts = predictive.get("activations")
        result["activations"] = (
            compare_group_activations(progress, acts, tracked["groups"], window)
            if acts
            else unavailable("predictive solution has no muscle activations")
        )
        result["activations_tracked_classification"] = tracked["muscle_classification"]
        result["activations_tracked_solver"] = tracked["muscle_solver"]
    else:
        result["joint_angles"] = unavailable(tracked["muscles_reason"])
        result["activations"] = unavailable(tracked["muscles_reason"])
    result["l5_s1"] = unavailable(
        tracked.get("l5_s1_reason", "predictive L5/S1 load is not computed")
    )
    return result
