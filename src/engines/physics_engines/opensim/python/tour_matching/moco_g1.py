"""Pure helpers for the Moco G1 horizon ladder (MS-42 phase A, #10341).

Everything here runs without the OpenSim bindings: horizon ladder and mesh
selection, capture windowing and the marker gap policy, warm-start assembly
for chaining horizons, the G1 gate table and the OS-7 receipt contract. The
OpenSim-bound solve and replay live in :mod:`moco_tracking` and the OS-7
driver; they receive plain arrays from these functions (LoD).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.tour_metrics import SharedMetrics

Array = NDArray[np.float64]

#: Program ladder: pilot horizon, three intermediate rungs, G1 (0.85 s), then
#: the full capture. Rungs above the capture length are dropped.
LADDER_RUNGS_S: tuple[float, ...] = (0.1, 0.3, 0.6, 0.85)

#: G1 targets from the program tracker (metres; pelvis yaw in degrees).
G1_TARGETS: Mapping[str, float] = MappingProxyType(
    {
        "whole_marker_rmse_m": 0.025,
        "early_marker_rmse_m": 0.012,
        "terminal_marker_rmse_m": 0.035,
        "club_marker_rmse_m": 0.060,
        "pelvis_yaw_rmse_deg": 3.0,
    }
)

#: Keys every OS-7 receipt must carry (numbers quoted in docs come from here).
REQUIRED_RECEIPT_KEYS: tuple[str, ...] = (
    "deliverable",
    "model_sha256",
    "trc_sha256",
    "horizon_s",
    "mesh_intervals",
    "solver_status",
    "iterations",
    "wall_clock_s",
    "solution_metrics",
    "replay_metrics",
    "gates",
    "per_horizon",
    "qualification",
)


@precondition(lambda full_s: np.isfinite(full_s) and full_s > 0.0, "full_s must be > 0")
@postcondition(lambda r: len(r) >= 1 and all(np.diff(r) > 0), "ladder must increase")
def horizon_ladder(
    full_s: float, rungs_s: Sequence[float] = LADDER_RUNGS_S
) -> tuple[float, ...]:
    """Return the increasing horizon ladder ending at the full capture length."""
    ladder = [float(r) for r in rungs_s if r < full_s - 1e-9]
    ladder.append(float(full_s))
    return tuple(ladder)


@precondition(
    lambda duration_s, per_second=100.0, minimum=10: (
        duration_s > 0.0 and per_second > 0.0 and minimum >= 1
    ),
    "duration and density must be positive",
)
def mesh_intervals_for(
    duration_s: float, per_second: float = 100.0, minimum: int = 10
) -> int:
    """Mesh intervals for a horizon: ``per_second`` per second, floored at ``minimum``."""
    return max(int(minimum), int(round(duration_s * per_second)))


@precondition(
    lambda capture, t_end_s: t_end_s > 0.0 and np.isfinite(t_end_s),
    "t_end_s must be positive",
)
def window_capture(capture: TourCapture, t_end_s: float) -> TourCapture:
    """Frames with ``time <= t_end_s`` (needs at least two frames)."""
    keep = capture.time_s <= t_end_s + 1e-9
    if np.count_nonzero(keep) < 2:
        raise ValueError(f"Window to {t_end_s} s holds fewer than two frames")
    return TourCapture(
        capture.time_s[keep],
        capture.labels,
        capture.points_m[keep],
        capture.valid[keep],
        capture.source_sha256,
    )


def trim_trailing_invalid(
    capture: TourCapture, labels: Sequence[str]
) -> tuple[TourCapture, int]:
    """Drop trailing frames where any of ``labels`` is invalid; returns (capture, dropped)."""
    columns = [capture.index(label) for label in labels]
    ok = (
        capture.valid[:, columns].all(axis=1)
        if columns
        else np.ones(capture.frames, bool)
    )
    last_good = int(np.max(np.nonzero(ok)[0])) if ok.any() else -1
    if last_good < 1:
        raise ValueError("Fewer than two frames remain after trimming trailing gaps")
    dropped = capture.frames - (last_good + 1)
    if dropped == 0:
        return capture, 0
    return (
        TourCapture(
            capture.time_s[: last_good + 1],
            capture.labels,
            capture.points_m[: last_good + 1],
            capture.valid[: last_good + 1],
            capture.source_sha256,
        ),
        dropped,
    )


def _gap_runs(invalid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """(start, length) of each run of ``True`` in a boolean vector."""
    idx = np.flatnonzero(invalid)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(r[0]), int(r.size)) for r in np.split(idx, breaks)]


@precondition(
    lambda capture, max_gap_frames: max_gap_frames >= 0, "max_gap_frames must be >= 0"
)
def fill_marker_gaps(
    capture: TourCapture, max_gap_frames: int
) -> tuple[TourCapture, dict[str, int]]:
    """Linearly interpolate interior gaps of at most ``max_gap_frames`` frames.

    Leading and trailing gaps, and gaps longer than the limit, are left
    invalid. Returns the filled capture and ``{label: frames_filled}``; the
    input capture is not modified. Filled samples exist only so Moco's marker
    splines are defined; metrics must use the original validity mask.
    """
    points = np.array(capture.points_m, copy=True)
    valid = np.array(capture.valid, copy=True)
    report: dict[str, int] = {}
    for m, label in enumerate(capture.labels):
        filled = 0
        for start, length in _gap_runs(~capture.valid[:, m]):
            end = start + length  # exclusive
            if start == 0 or end >= capture.frames or length > max_gap_frames:
                continue
            t0, t1 = capture.time_s[start - 1], capture.time_s[end]
            w = (capture.time_s[start:end] - t0) / (t1 - t0)
            points[start:end, m] = (1.0 - w)[:, None] * capture.points_m[
                start - 1, m
            ] + w[:, None] * capture.points_m[end, m]
            valid[start:end, m] = True
            filled += length
        if filled:
            report[label] = filled
    return (
        TourCapture(
            capture.time_s, capture.labels, points, valid, capture.source_sha256
        ),
        report,
    )


@precondition(
    lambda capture, min_valid_fraction: 0.0 < min_valid_fraction <= 1.0,
    "min_valid_fraction must be in (0, 1]",
)
def retain_markers(
    capture: TourCapture, min_valid_fraction: float
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Labels whose valid fraction meets the floor, and the fraction of those dropped."""
    fraction = capture.valid.mean(axis=0)
    kept = tuple(
        label
        for label, f in zip(capture.labels, fraction, strict=True)
        if f >= min_valid_fraction
    )
    dropped = {
        label: float(f)
        for label, f in zip(capture.labels, fraction, strict=True)
        if f < min_valid_fraction
    }
    if not kept:
        raise ValueError("No marker satisfies the validity floor")
    return kept, dropped


def read_sto(path: Path) -> tuple[Array, dict[str, Array]]:
    """Read an OpenSim ``.sto``/``.mot`` table: (time, {column: values}).

    Pure text parsing; ``inDegrees`` is not applied (Moco state files are in
    radians). Raises ValueError when the ``endheader`` line is missing.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_end = next(
            i for i, line in enumerate(lines) if line.strip() == "endheader"
        )
    except StopIteration as error:
        raise ValueError(f"{path} has no endheader line") from error
    labels = lines[header_end + 1].split("\t")
    rows = np.array(
        [
            [float(v) for v in line.split("\t")]
            for line in lines[header_end + 2 :]
            if line.strip()
        ],
        dtype=float,
    )
    if rows.ndim != 2 or rows.shape[1] != len(labels):
        raise ValueError(f"{path}: data width disagrees with its label row")
    return rows[:, 0], {label: rows[:, i] for i, label in enumerate(labels) if i > 0}


def write_sto(
    path: Path, time_s: Array, columns: Mapping[str, Array], *, in_degrees: bool
) -> Path:
    """Write an OpenSim ``.sto``/``.mot`` table; every column must match ``time_s``."""
    t = np.asarray(time_s, dtype=float)
    for name, values in columns.items():
        if np.asarray(values).shape != t.shape:
            raise ValueError(f"Column {name} does not match the time vector")
    path = Path(path)
    header = [
        path.stem,
        "version=1",
        f"nRows={t.size}",
        f"nColumns={len(columns) + 1}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "endheader",
        "\t".join(["time", *columns]),
    ]
    matrix = np.column_stack(
        [t, *[np.asarray(v, dtype=float) for v in columns.values()]]
    )
    rows = ["\t".join(f"{v:.10g}" for v in row) for row in matrix]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    return path


@precondition(
    lambda columns, frames: frames >= 1 and frames % 2 == 1,
    "frames must be an odd positive window length",
)
def smooth_columns(columns: Mapping[str, Array], frames: int) -> dict[str, Array]:
    """Centred Hann moving average of each column (edge-padded, length kept).

    ``frames == 1`` returns copies. Used to take the 360 Hz IK jitter out of
    the warm-start speeds and the IK-state reference: the OS-3b IK shows
    per-frame jumps of 0.045 rad (finite-difference accelerations above
    1500 rad/s^2) that a dynamic solve cannot reconcile.
    """
    if frames == 1:
        return {name: np.array(values, dtype=float) for name, values in columns.items()}
    window = np.hanning(frames + 2)[1:-1]
    window /= window.sum()
    half = frames // 2
    out: dict[str, Array] = {}
    for name, values in columns.items():
        q = np.asarray(values, dtype=float)
        padded = np.pad(q, half, mode="edge")
        out[name] = np.convolve(padded, window, mode="valid")
    return out


def _interp(t: Array, src_t: Array, src: Array) -> Array:
    return np.interp(t, src_t, src)


@precondition(
    lambda grid_t, **_: (
        grid_t.ndim == 1 and grid_t.size >= 2 and np.all(np.diff(grid_t) > 0)
    ),
    "grid_t must be an increasing vector",
)
def assemble_warm_start(
    grid_t: Array,
    *,
    state_names: Sequence[str],
    control_names: Sequence[str],
    previous: tuple[Array, Mapping[str, Array], Mapping[str, Array]] | None,
    ik: tuple[Array, Mapping[str, Array]],
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Assemble a warm start on ``grid_t`` from a shorter solution and IK.

    Inside the previous solution's time range the previous states and
    controls are interpolated; beyond it, coordinate values come from IK,
    speeds from the IK finite difference and controls are held at the last
    previous value (zero when there is no previous solution). Every state in
    ``state_names`` must be a ``.../value`` or ``.../speed`` coordinate state
    whose value column exists in the IK table.
    """
    ik_t, ik_values = ik
    ik_t = np.asarray(ik_t, dtype=float)
    prev_end = -np.inf
    if previous is not None:
        prev_t = np.asarray(previous[0], dtype=float)
        prev_end = float(prev_t[-1])
    inside = grid_t <= prev_end + 1e-12

    states: dict[str, Array] = {}
    for name in state_names:
        value_name = (
            name if name.endswith("/value") else name[: -len("/speed")] + "/value"
        )
        if value_name not in ik_values:
            raise KeyError(f"IK table lacks {value_name}")
        q_ik = np.asarray(ik_values[value_name], dtype=float)
        if name.endswith("/value"):
            column = _interp(grid_t, ik_t, q_ik)
        elif name.endswith("/speed"):
            column = _interp(grid_t, ik_t, np.gradient(q_ik, ik_t))
        else:
            raise ValueError(f"Unsupported state column: {name}")
        if previous is not None and name in previous[1] and inside.any():
            column[inside] = _interp(
                grid_t[inside], prev_t, np.asarray(previous[1][name])
            )
        states[name] = column

    controls: dict[str, Array] = {}
    for name in control_names:
        column = np.zeros_like(grid_t)
        if previous is not None and name in previous[2]:
            u_prev = np.asarray(previous[2][name], dtype=float)
            column[inside] = _interp(grid_t[inside], prev_t, u_prev)
            column[~inside] = u_prev[-1]
        controls[name] = column
    return states, controls


def gate_table(
    metrics: SharedMetrics, targets: Mapping[str, float] = G1_TARGETS
) -> dict:
    """Per-metric value/target/pass table and the overall verdict for a horizon."""
    values = {
        "whole_marker_rmse_m": metrics.whole_marker_rmse_m,
        "early_marker_rmse_m": metrics.early_marker_rmse_m,
        "terminal_marker_rmse_m": metrics.terminal_marker_rmse_m,
        "club_marker_rmse_m": metrics.club_marker_rmse_m,
        "pelvis_yaw_rmse_deg": float(np.degrees(metrics.pelvis_yaw_rmse_rad)),
    }
    gates = {
        key: {
            "value": float(values[key]),
            "target": float(targets[key]),
            "passed": bool(np.isfinite(values[key]) and values[key] <= targets[key]),
        }
        for key in values
    }
    return {"gates": gates, "all_passed": all(g["passed"] for g in gates.values())}


def validate_os7_receipt(document: Mapping[str, object]) -> bool:
    """Fail closed when a required OS-7 receipt key is missing."""
    missing = [key for key in REQUIRED_RECEIPT_KEYS if key not in document]
    if missing:
        raise KeyError(f"OS-7 receipt is missing required keys: {', '.join(missing)}")
    return True


__all__ = [
    "G1_TARGETS",
    "LADDER_RUNGS_S",
    "REQUIRED_RECEIPT_KEYS",
    "assemble_warm_start",
    "fill_marker_gaps",
    "gate_table",
    "horizon_ladder",
    "mesh_intervals_for",
    "read_sto",
    "retain_markers",
    "smooth_columns",
    "trim_trailing_invalid",
    "validate_os7_receipt",
    "window_capture",
    "write_sto",
]
