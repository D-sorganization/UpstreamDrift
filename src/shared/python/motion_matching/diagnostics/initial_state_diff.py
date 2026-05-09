"""Visualize the delta between specified and constraint-resolved initial poses.

The Simscape 3D golf model is a closed kinematic chain: both arms grip
the same club, and the club itself is a kinematic loop with the shaft.
When you load an input MAT (e.g. ``3DModelInputs_Impact.mat``) and start
the simulation, Simscape's constraint solver projects the requested
pose onto the constraint manifold at ``t=0``. The actual converged pose
can differ from what's documented in the input file.

This module loads a diagnostic report produced by
``diagnose_initial_state.m`` and renders it: skeleton overlay,
per-joint angle deltas, Cartesian deltas at butt and clubhead, and a
markdown summary suitable for a PR comment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from ._skeleton_render import (
    draw_delta_arrows,
    draw_segments,
    equalize_3d_axes,
)

# Significance thresholds — match the MATLAB defaults so the Python and
# MATLAB tools agree on whether a report is "interesting".
DEFAULT_JOINT_THRESHOLD_DEG = 1.0
DEFAULT_POS_THRESHOLD_MM = 5.0


@dataclass(frozen=True)
class InitialStateDiffReport:
    """Normalized view of the MATLAB diagnostic report.

    Both ``specified`` and ``actual`` are dictionaries keyed by
    ``"q"``, ``"r_butt"``, ``"r_clubhead"``. Joint angles are in
    radians; Cartesian markers are in metres. ``delta`` carries the
    pre-computed deltas from MATLAB (degrees and millimetres).
    """

    specified: dict[str, np.ndarray]
    actual: dict[str, np.ndarray]
    delta: dict[str, Any]
    joint_names: list[str]
    input_file: str = ""
    input_file_hash: str = ""
    model_name: str = ""
    timestamp: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        return bool(self.delta.get("is_significant", False))

    @property
    def joint_threshold_deg(self) -> float:
        return float(
            self.thresholds.get("joint_threshold_deg", DEFAULT_JOINT_THRESHOLD_DEG)
        )

    @property
    def pos_threshold_mm(self) -> float:
        return float(self.thresholds.get("pos_threshold_mm", DEFAULT_POS_THRESHOLD_MM))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _as_1d_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _as_3vec(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.size == 0:
        return None
    if arr.size != 3:
        return None
    return arr.reshape(3)


def _decode_struct(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of a scipy.io loaded struct to a plain dict."""
    if isinstance(obj, dict):
        return {k: obj[k] for k in obj if not k.startswith("__")}
    # scipy returns numpy structured arrays for MATLAB structs
    if hasattr(obj, "dtype") and obj.dtype.names:
        flat = obj.squeeze()
        return {
            name: flat[name].item() if flat[name].size == 1 else flat[name]
            for name in obj.dtype.names
        }
    raise TypeError(f"Cannot decode object of type {type(obj)!r} into a dict")


def _decode_joint_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    arr = np.asarray(raw).squeeze()
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(x).strip() for x in arr.tolist()]


def load_diff_report(mat_path: Path | str) -> InitialStateDiffReport:  # noqa: C901
    """Load a MAT report written by ``diagnose_initial_state.m``.

    The MAT file is a flat struct: ``specified``, ``actual``, ``delta``,
    plus scalar provenance fields. We use ``scipy.io.loadmat`` with
    ``squeeze_me`` and ``struct_as_record=False`` to keep code paths
    short.
    """
    from scipy.io import loadmat  # imported lazily for fast module import

    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic report not found: {path}")

    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    def _struct_to_dict(s: Any) -> dict[str, Any]:
        if hasattr(s, "_fieldnames"):
            return {n: getattr(s, n) for n in s._fieldnames}
        if isinstance(s, dict):
            return s
        raise TypeError(f"Expected MATLAB struct, got {type(s)!r}")

    if "specified" not in raw or "actual" not in raw or "delta" not in raw:
        raise ValueError(
            f"Malformed report: missing one of specified/actual/delta in {path}"
        )

    spec_raw = _struct_to_dict(raw["specified"])
    act_raw = _struct_to_dict(raw["actual"])
    delta_raw = _struct_to_dict(raw["delta"])

    specified: dict[str, np.ndarray] = {"q": _as_1d_array(spec_raw.get("q", []))}
    actual: dict[str, np.ndarray] = {"q": _as_1d_array(act_raw.get("q", []))}

    for key, decoded in (("r_butt", _as_3vec), ("r_clubhead", _as_3vec)):
        sv = decoded(spec_raw.get(key))
        av = decoded(act_raw.get(key))
        if sv is not None:
            specified[key] = sv
        if av is not None:
            actual[key] = av

    delta: dict[str, Any] = {
        "q_per_joint_deg": _as_1d_array(delta_raw.get("q_per_joint_deg", [])),
        "q_max_deg": float(np.asarray(delta_raw.get("q_max_deg", np.nan)).item()),
        "r_butt_mm": float(np.asarray(delta_raw.get("r_butt_mm", np.nan)).item()),
        "r_clubhead_mm": float(
            np.asarray(delta_raw.get("r_clubhead_mm", np.nan)).item()
        ),
        "is_significant": bool(
            np.asarray(delta_raw.get("is_significant", False)).item()
        ),
    }

    joint_names = _decode_joint_names(raw.get("joint_names"))
    if not joint_names and specified["q"].size:
        joint_names = [f"q{i + 1}" for i in range(specified["q"].size)]

    thresholds: dict[str, float] = {}
    if "thresholds" in raw:
        try:
            t = _struct_to_dict(raw["thresholds"])
            thresholds = {k: float(np.asarray(v).item()) for k, v in t.items()}
        except (TypeError, ValueError):
            thresholds = {}

    return InitialStateDiffReport(
        specified=specified,
        actual=actual,
        delta=delta,
        joint_names=joint_names,
        input_file=str(raw.get("input_file", "")),
        input_file_hash=str(raw.get("input_file_hash", "")),
        model_name=str(raw.get("model_name", "")),
        timestamp=str(raw.get("timestamp", "")),
        thresholds=thresholds,
    )


def _validate_report(report: InitialStateDiffReport) -> None:
    if not isinstance(report, InitialStateDiffReport):
        raise TypeError("Expected InitialStateDiffReport instance")
    n_spec = report.specified["q"].size
    n_act = report.actual["q"].size
    if n_spec != n_act:
        raise ValueError(
            f"Specified and actual joint counts differ: {n_spec} vs {n_act}"
        )
    if report.joint_names and len(report.joint_names) != n_spec:
        raise ValueError(
            f"joint_names length ({len(report.joint_names)}) does not match q ({n_spec})"
        )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_skeleton_overlay(report: InitialStateDiffReport) -> Figure:
    """Render the specified vs. actual skeleton in 3D with delta arrows.

    The "skeleton" here is intentionally minimal: a polyline from
    butt to clubhead. When richer joint-position data is available
    (future enhancement), additional segments can be appended without
    breaking the API.
    """
    import matplotlib.pyplot as plt

    _validate_report(report)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    spec_pts: list[np.ndarray] = []
    act_pts: list[np.ndarray] = []
    if "r_butt" in report.specified and "r_butt" in report.actual:
        spec_pts.append(report.specified["r_butt"])
        act_pts.append(report.actual["r_butt"])
    if "r_clubhead" in report.specified and "r_clubhead" in report.actual:
        spec_pts.append(report.specified["r_clubhead"])
        act_pts.append(report.actual["r_clubhead"])

    n_arrows = 0
    if spec_pts:
        draw_segments(ax, spec_pts, color="tab:blue", label="specified")
        draw_segments(ax, act_pts, color="tab:orange", label="actual")
        n_arrows = draw_delta_arrows(ax, spec_pts, act_pts, color="tab:red")
        all_pts = np.vstack(spec_pts + act_pts)
        equalize_3d_axes(ax, all_pts)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")  # type: ignore[attr-defined]
    ax.set_title("Initial pose: specified vs. constraint-resolved")
    if spec_pts:
        ax.legend(loc="best")

    # Stash the arrow count so tests can verify deltas were drawn.
    fig._delta_arrow_count = n_arrows  # type: ignore[attr-defined]
    return fig


def plot_per_joint_delta_bars(report: InitialStateDiffReport) -> Figure:
    """Horizontal bar chart of per-joint angle deltas, sorted by magnitude."""
    import matplotlib.pyplot as plt

    _validate_report(report)
    deltas = report.delta["q_per_joint_deg"]
    names = report.joint_names or [f"q{i + 1}" for i in range(deltas.size)]

    if deltas.size == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No joint deltas available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    order = np.argsort(np.abs(deltas))[::-1]
    sorted_deltas = deltas[order]
    sorted_names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(sorted_names))))
    y = np.arange(len(sorted_names))
    colors = [
        "tab:red" if abs(d) > report.joint_threshold_deg else "tab:gray"
        for d in sorted_deltas
    ]
    ax.barh(y, sorted_deltas, color=colors)
    ax.set_yticks(y)  # type: ignore[operator]
    ax.set_yticklabels(sorted_names)  # type: ignore[operator]
    ax.invert_yaxis()
    ax.axvline(
        report.joint_threshold_deg, color="tab:red", linestyle="--", linewidth=0.8
    )
    ax.axvline(
        -report.joint_threshold_deg, color="tab:red", linestyle="--", linewidth=0.8
    )
    ax.set_xlabel("delta (deg, actual - specified)")
    ax.set_title("Per-joint constraint-projection delta")

    fig._sorted_abs_deltas = np.abs(sorted_deltas)  # type: ignore[attr-defined]
    return fig


def plot_cartesian_delta_summary(report: InitialStateDiffReport) -> Figure:
    """3D scatter showing butt and clubhead Cartesian deltas."""
    import matplotlib.pyplot as plt

    _validate_report(report)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if "r_butt" in report.specified and "r_butt" in report.actual:
        pairs.append(("butt", report.specified["r_butt"], report.actual["r_butt"]))
    if "r_clubhead" in report.specified and "r_clubhead" in report.actual:
        pairs.append(
            ("clubhead", report.specified["r_clubhead"], report.actual["r_clubhead"])
        )

    for label, s, a in pairs:
        ax.scatter(s[0], s[1], s[2], color="tab:blue", marker="o", s=60)  # type: ignore[misc]
        ax.scatter(a[0], a[1], a[2], color="tab:orange", marker="^", s=60)  # type: ignore[misc]
        ax.text(s[0], s[1], s[2], f"{label} (spec)", fontsize=8)  # type: ignore[arg-type, call-arg]
        delta = a - s
        ax.quiver(
            s[0],
            s[1],
            s[2],
            delta[0],
            delta[1],
            delta[2],
            color="tab:red",
            arrow_length_ratio=0.2,
        )

    if pairs:
        all_pts = np.vstack([s for _, s, _ in pairs] + [a for _, _, a in pairs])
        equalize_3d_axes(ax, all_pts)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")  # type: ignore[attr-defined]
    ax.set_title(
        f"Cartesian deltas (butt: {report.delta['r_butt_mm']:.2f} mm, "
        f"clubhead: {report.delta['r_clubhead_mm']:.2f} mm)"
    )
    return fig


# --------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------


def summarize_for_pr_comment(report: InitialStateDiffReport) -> str:
    """Return a markdown summary for posting as a PR comment."""
    _validate_report(report)
    d = report.delta
    sig = "**SIGNIFICANT**" if report.is_significant else "negligible"
    lines: list[str] = [
        "## Initial-state constraint-projection diagnostic",
        "",
        f"- input: `{report.input_file or '(unknown)'}`",
        f"- model: `{report.model_name or '(unknown)'}`",
        f"- timestamp: `{report.timestamp or '(unknown)'}`",
        f"- input file sha256: `{report.input_file_hash or '(unknown)'}`",
        "",
        f"**Verdict:** {sig}",
        "",
        "| metric | value | threshold |",
        "|---|---|---|",
        f"| max joint delta | {d['q_max_deg']:.3f} deg | {report.joint_threshold_deg:.3f} deg |",
        f"| butt position delta | {d['r_butt_mm']:.3f} mm | {report.pos_threshold_mm:.3f} mm |",
        f"| clubhead position delta | {d['r_clubhead_mm']:.3f} mm | {report.pos_threshold_mm:.3f} mm |",
        "",
    ]

    deltas = d["q_per_joint_deg"]
    if deltas.size:
        order = np.argsort(np.abs(deltas))[::-1]
        top = order[: min(5, deltas.size)]
        lines.append("### Top joint deltas")
        lines.append("")
        lines.append("| joint | delta (deg) |")
        lines.append("|---|---|")
        for i in top:
            name = report.joint_names[i] if i < len(report.joint_names) else f"q{i + 1}"
            lines.append(f"| `{name}` | {deltas[i]:+.3f} |")
        lines.append("")

    if report.is_significant:
        lines.append(
            "> The constraint solver moved the model meaningfully from the "
            'specified pose. Downstream consumers that assume "the model '
            'starts where I told it to" will be working from the '
            "post-projection pose, not the input file."
        )
    return "\n".join(lines)


def report_to_json(report: InitialStateDiffReport) -> str:
    """Serialize a report to JSON (numpy arrays -> lists). Used by the CLI."""
    _validate_report(report)

    def _enc(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, dict):
            return {k: _enc(v) for k, v in x.items()}
        return x

    payload = {
        "specified": _enc(report.specified),
        "actual": _enc(report.actual),
        "delta": _enc(report.delta),
        "joint_names": list(report.joint_names),
        "input_file": report.input_file,
        "input_file_hash": report.input_file_hash,
        "model_name": report.model_name,
        "timestamp": report.timestamp,
        "thresholds": dict(report.thresholds),
    }
    return json.dumps(payload, indent=2, sort_keys=True)
