"""Shoulder, upper-trunk and pelvis turn lines with X-factor (issue #12042).

Three distinct, named horizontal-plane lines are measured from capture markers
or from model forward-kinematics points:

- ``shoulder_girdle``: the shoulder-back markers (or the shoulder joint
  centres).  The markers ride the scapula/acromion, so this line includes
  scapular rotation on top of the trunk.
- ``upper_trunk``: the BackLeft/BackRight markers (or thorax-fixed model
  points).  This is the rib-cage proxy and is what "X-factor" is built on.
- ``pelvis``: the WaistLeft/WaistRight markers (or the hip joint centres).

Frame convention (ADR-0041): world Z up, golfer faces -X, target line -Y.  A
C3D file is Y-up in metres and is mapped ``(x, -z, y)`` before it gets here.
A line's *yaw* is ``atan2(y, x)`` of the right-to-left vector projected onto
the horizontal XY plane.  The *turn* is that yaw relative to the same line at
address, sign-flipped so that **positive is the backswing** (clockwise seen
from above for a right-handed golfer), unwrapped through +-180 degrees.

X-factor is the upper-trunk turn minus the pelvis turn.  The shoulder-girdle
variant is reported separately as ``x_factor_shoulder_girdle``.

Data validity: both markers of every pair must be finite.  Gaps no longer than
``MAX_FILL_GAP_S`` seconds are linearly interpolated per marker; longer gaps
and leading/trailing gaps stay NaN and carry a reason.  Unavailable never
means zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.swing_comparison.events import SwingEvents

LINE_NAMES: tuple[str, ...] = ("shoulder_girdle", "upper_trunk", "pelvis")

#: Longest marker dropout (seconds) that is bridged by linear interpolation.
MAX_FILL_GAP_S: float = 0.10
#: A pair is only preferred over its fallback when at least this share of
#: frames is finite after short-gap filling.
MIN_PREFERRED_VALID_FRACTION: float = 0.5
#: A line shorter than this horizontally (metres) has no defined yaw.
MIN_HORIZONTAL_LENGTH_M: float = 1e-3
#: Plausible right-to-left line length (metres); catches mm/cm inputs.
PLAUSIBLE_LINE_LENGTH_M: tuple[float, float] = (0.02, 1.5)

FRAME_CONVENTION = (
    "Z up, golfer faces -X, target line -Y; yaw = atan2(y, x) of the right-to-left "
    "line in the XY plane; turn = -(yaw - yaw_at_address), unwrapped, positive in "
    "the backswing"
)

#: Capture marker pairs, ordered by preference: ((left, right), ...).
MARKER_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "shoulder_girdle": (
        ("LShoulderBack", "RShoulderBack"),
        ("LShoulderTop", "RShoulderTop"),
    ),
    "upper_trunk": (("BackLeft", "BackRight"),),
    "pelvis": (("WaistLeft", "WaistRight"), ("WaistLBack", "WaistRBack")),
}

#: Default model point names (left, right) accepted by ``model_turn_lines``.
MODEL_POINT_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "shoulder_girdle": (("shoulder_l", "shoulder_r"),),
    "upper_trunk": (("thorax_l", "thorax_r"),),
    "pelvis": (("hip_l", "hip_r"), ("WaistLeft", "WaistRight")),
}


@dataclass(frozen=True)
class LineTurn:
    """Turn of one left/right line about vertical, relative to address.

    Attributes:
        name: Line name (one of ``LINE_NAMES`` or a derived name).
        t: Time stamps (N,) in seconds.
        turn_deg: Turn (N,) in degrees, positive in the backswing; NaN where
            the line is unavailable.
        status: ``"ok"`` (all frames finite), ``"partial"`` or ``"unavailable"``.
        reason: Why the line is not fully ``"ok"`` (``None`` when ok).
        points: The (left, right) point names actually used, if any.
        filled_frames: Count of interpolated marker-frames (short gaps).
        valid_fraction: Share of frames with a finite turn.
    """

    name: str
    t: np.ndarray
    turn_deg: np.ndarray
    status: str
    reason: str | None
    points: tuple[str, str] | None = None
    filled_frames: int = 0
    valid_fraction: float = 0.0
    address_idx: int = field(default=0)

    def index_at(self, time_s: float) -> int:
        """Return the frame index nearest to ``time_s``."""
        return _nearest_index(self.t, time_s)

    def value_at(self, time_s: float) -> float:
        """Return the turn (deg) at the frame nearest to ``time_s`` (NaN if none)."""
        return float(self.turn_deg[self.index_at(time_s)])

    def max_backswing_deg(self, start_s: float, end_s: float) -> float:
        """Return the maximum turn within ``[start_s, end_s]`` (NaN if none)."""
        i0, i1 = self.index_at(start_s), self.index_at(end_s)
        window = self.turn_deg[min(i0, i1) : max(i0, i1) + 1]
        if window.size == 0 or not np.isfinite(window).any():
            return float("nan")
        return float(np.nanmax(window))


@dataclass(frozen=True)
class TurnMetrics:
    """Shoulder-girdle, upper-trunk and pelvis turn lines plus X-factors."""

    shoulder_girdle: LineTurn
    upper_trunk: LineTurn
    pelvis: LineTurn
    x_factor: LineTurn
    x_factor_shoulder_girdle: LineTurn

    def lines(self) -> dict[str, LineTurn]:
        """Return all lines keyed by name."""
        return {
            "shoulder_girdle": self.shoulder_girdle,
            "upper_trunk": self.upper_trunk,
            "pelvis": self.pelvis,
            "x_factor": self.x_factor,
            "x_factor_shoulder_girdle": self.x_factor_shoulder_girdle,
        }


def _nearest_index(t: np.ndarray, time_s: float) -> int:
    require(np.isfinite(time_s), "time_s must be finite", time_s)
    i = int(np.searchsorted(t, time_s))
    if i <= 0:
        return 0
    if i >= len(t):
        return len(t) - 1
    return i if (t[i] - time_s) < (time_s - t[i - 1]) else i - 1


def _validated_time(t: Any) -> np.ndarray:
    if not isinstance(t, (np.ndarray, Sequence)):
        raise TypeError("t must be a numpy array or a sequence of seconds")
    arr = np.asarray(t, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 4:
        raise ValueError(f"t must be 1-D with at least 4 frames, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or not np.all(np.diff(arr) > 0):
        raise ValueError("t must be finite and strictly increasing")
    return arr


def _validated_points(name: str, pts: Any, n: int) -> np.ndarray:
    if not isinstance(pts, np.ndarray):
        raise TypeError(f"{name}: expected a numpy array, got {type(pts).__name__}")
    if pts.shape != (n, 3):
        raise ValueError(f"{name}: expected shape ({n}, 3), got {pts.shape}")
    return np.asarray(pts, dtype=np.float64)


def fill_short_gaps(points: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, int]:
    """Linearly bridge NaN runs of at most ``max_gap_frames`` between valid frames.

    Rows with any NaN are invalid.  Leading and trailing gaps, and gaps longer
    than ``max_gap_frames``, are left NaN (no extrapolation).

    Returns:
        ``(filled copy, number of frames filled)``.
    """
    require(max_gap_frames >= 0, "max_gap_frames must be >= 0", max_gap_frames)
    out = np.array(points, dtype=np.float64, copy=True)
    valid = np.isfinite(out).all(axis=1)
    out[~valid] = np.nan
    idx = np.flatnonzero(valid)
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        gap = int(b - a - 1)
        if 0 < gap <= max_gap_frames:
            w = (np.arange(a + 1, b) - a) / float(b - a)
            out[a + 1 : b] = out[a] + w[:, None] * (out[b] - out[a])
            filled += gap
    return out, filled


def _unwrap_with_nan(angle: np.ndarray) -> np.ndarray:
    out = np.full_like(angle, np.nan)
    ok = np.isfinite(angle)
    if ok.any():
        out[ok] = np.unwrap(angle[ok])
    return out


def _unavailable(
    name: str,
    t: np.ndarray,
    reason: str,
    address_idx: int,
    points: tuple[str, str] | None = None,
) -> LineTurn:
    return LineTurn(
        name=name,
        t=t,
        turn_deg=np.full(len(t), np.nan),
        status="unavailable",
        reason=reason,
        points=points,
        address_idx=address_idx,
    )


def line_turn(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    t: np.ndarray,
    address_time_s: float,
    *,
    points: tuple[str, str] | None = None,
    max_gap_s: float = MAX_FILL_GAP_S,
) -> LineTurn:
    """Compute the turn of the right-to-left line relative to address.

    Args:
        name: Name recorded on the result.
        left: Left point positions (N, 3), metres, world frame (Z up).
        right: Right point positions (N, 3), metres, same frame.
        t: Time stamps (N,), seconds, strictly increasing.
        address_time_s: Time of the address reference frame (nearest frame).
        points: Optional (left, right) point names for provenance.
        max_gap_s: Longest gap bridged by interpolation, in seconds.

    Returns:
        A ``LineTurn``; NaN (never zero) wherever a marker or the address
        reference is unavailable.

    Raises:
        TypeError: for non-array inputs.
        ValueError: for wrong shapes, bad time base, negative ``max_gap_s`` or
            positions whose line length is implausible for metres.
    """
    t_arr = _validated_time(t)
    n = len(t_arr)
    left = _validated_points(f"{name} left", left, n)
    right = _validated_points(f"{name} right", right, n)
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError(f"max_gap_s must be finite and >= 0, got {max_gap_s}")
    address_idx = _nearest_index(t_arr, address_time_s)

    dt = float(np.median(np.diff(t_arr)))
    max_gap_frames = int(np.floor(max_gap_s / dt + 1e-9))
    l_fill, n_l = fill_short_gaps(left, max_gap_frames)
    r_fill, n_r = fill_short_gaps(right, max_gap_frames)
    vec = l_fill - r_fill
    horiz = np.hypot(vec[:, 0], vec[:, 1])
    usable = np.isfinite(vec).all(axis=1) & (horiz >= MIN_HORIZONTAL_LENGTH_M)
    if not usable.any():
        return _unavailable(
            name, t_arr, "no_frame_with_both_points_finite", address_idx, points
        )

    length = float(np.median(np.linalg.norm(vec[usable], axis=1)))
    lo, hi = PLAUSIBLE_LINE_LENGTH_M
    if not lo <= length <= hi:
        raise ValueError(
            f"{name}: median line length {length:.4g} is outside the plausible "
            f"{lo}-{hi} m range; positions must be in metres"
        )

    yaw = np.full(n, np.nan)
    yaw[usable] = np.arctan2(vec[usable, 1], vec[usable, 0])
    yaw = _unwrap_with_nan(yaw)
    if not np.isfinite(yaw[address_idx]):
        return _unavailable(
            name, t_arr, "address_frame_unavailable", address_idx, points
        )
    turn = -np.degrees(yaw - yaw[address_idx]) + 0.0

    frac = float(np.isfinite(turn).mean())
    status = "ok" if frac == 1.0 else "partial"
    reason = (
        None
        if status == "ok"
        else (
            f"{int((~np.isfinite(turn)).sum())} frames unavailable "
            f"(gap longer than {max_gap_s:g} s or edge gap)"
        )
    )
    return LineTurn(
        name=name,
        t=t_arr,
        turn_deg=turn,
        status=status,
        reason=reason,
        points=points,
        filled_frames=n_l + n_r,
        valid_fraction=frac,
        address_idx=address_idx,
    )


def _choose_pair(
    name: str,
    pairs: Sequence[tuple[str, str]],
    points: Mapping[str, np.ndarray],
    t: np.ndarray,
    address_time_s: float,
    max_gap_s: float,
) -> LineTurn:
    """Return the first preferred pair with enough valid frames, else the best."""
    best: LineTurn | None = None
    for left_name, right_name in pairs:
        if left_name not in points or right_name not in points:
            continue
        cand = line_turn(
            name,
            points[left_name],
            points[right_name],
            t,
            address_time_s,
            points=(left_name, right_name),
            max_gap_s=max_gap_s,
        )
        if cand.valid_fraction >= MIN_PREFERRED_VALID_FRACTION:
            return cand
        if best is None or cand.valid_fraction > best.valid_fraction:
            best = cand
    if best is None:
        missing = ", ".join(f"{a}/{b}" for a, b in pairs)
        return _unavailable(
            name,
            t,
            f"missing_points: none of {missing} present",
            _nearest_index(t, address_time_s),
        )
    return best


def _difference(name: str, a: LineTurn, b: LineTurn, reason_prefix: str) -> LineTurn:
    """Return ``a - b`` as a LineTurn, NaN where either input is NaN."""
    diff = a.turn_deg - b.turn_deg
    frac = float(np.isfinite(diff).mean())
    if frac == 0.0:
        missing = [x.name for x in (a, b) if x.status == "unavailable"]
        return _unavailable(
            name,
            a.t,
            f"{reason_prefix}: {', '.join(missing) or 'no overlapping frames'} "
            "unavailable",
            a.address_idx,
        )
    status = "ok" if frac == 1.0 else "partial"
    return LineTurn(
        name=name,
        t=a.t,
        turn_deg=diff,
        status=status,
        reason=None if status == "ok" else "inputs unavailable on some frames",
        valid_fraction=frac,
        address_idx=a.address_idx,
    )


def _assemble(
    lines: dict[str, LineTurn],
) -> TurnMetrics:
    return TurnMetrics(
        shoulder_girdle=lines["shoulder_girdle"],
        upper_trunk=lines["upper_trunk"],
        pelvis=lines["pelvis"],
        x_factor=_difference(
            "x_factor", lines["upper_trunk"], lines["pelvis"], "x_factor"
        ),
        x_factor_shoulder_girdle=_difference(
            "x_factor_shoulder_girdle",
            lines["shoulder_girdle"],
            lines["pelvis"],
            "x_factor_shoulder_girdle",
        ),
    )


def compute_turn_lines(
    points: Mapping[str, np.ndarray],
    t: np.ndarray,
    events: SwingEvents,
    *,
    pairs: Mapping[str, Sequence[tuple[str, str]]] = MARKER_PAIRS,
    max_gap_s: float = MAX_FILL_GAP_S,
) -> TurnMetrics:
    """Compute all turn lines from named points (capture markers or model points).

    Args:
        points: Name -> (N, 3) positions in metres, world frame (Z up).
        t: Time stamps (N,), seconds.
        events: Swing events; only ``address_time`` is used here.
        pairs: Per-line preferred (left, right) point-name pairs.
        max_gap_s: Longest gap bridged by interpolation, in seconds.
    """
    t_arr = _validated_time(t)
    if not isinstance(points, Mapping):
        raise TypeError("points must be a mapping of name -> (N, 3) array")
    lines = {
        name: _choose_pair(
            name, pairs[name], points, t_arr, events.address_time, max_gap_s
        )
        for name in LINE_NAMES
    }
    return _assemble(lines)


def marker_turn_lines(
    markers: Mapping[str, np.ndarray],
    t: np.ndarray,
    events: SwingEvents,
    *,
    max_gap_s: float = MAX_FILL_GAP_S,
) -> TurnMetrics:
    """Turn lines from capture markers (``MARKER_PAIRS``)."""
    return compute_turn_lines(
        markers, t, events, pairs=MARKER_PAIRS, max_gap_s=max_gap_s
    )


def model_turn_lines(
    model_points: Mapping[str, np.ndarray],
    t: np.ndarray,
    events: SwingEvents,
    *,
    pairs: Mapping[str, Sequence[tuple[str, str]]] = MODEL_POINT_PAIRS,
    max_gap_s: float = MAX_FILL_GAP_S,
) -> TurnMetrics:
    """Turn lines from model forward-kinematics points.

    ``model_points`` uses ``MODEL_POINT_PAIRS`` names by default: hip joint
    centres (``hip_l``/``hip_r``, or the model's WaistLeft/WaistRight sites),
    thorax-fixed points (``thorax_l``/``thorax_r``) and the shoulder joint
    centres (``shoulder_l``/``shoulder_r``).  Build them with
    ``spec_model_points`` or from engine site positions.
    """
    return compute_turn_lines(model_points, t, events, pairs=pairs, max_gap_s=max_gap_s)


def spec_model_points(
    spec: Mapping[str, Any],
    q: np.ndarray,
    *,
    coordinate_names: Sequence[str] | None = None,
    marker_offsets: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    hip_bodies: tuple[str, str] = ("femur_l", "femur_r"),
    shoulder_frames: tuple[str, str] = ("LS", "RS"),
    thorax_markers: tuple[str, str] = ("BackLeft", "BackRight"),
) -> dict[str, np.ndarray]:
    """Model points for ``model_turn_lines`` from the shared spec forward kinematics.

    Uses ``identifiability.body_poses_from_coordinates`` (no new FK): hip centres
    are the femur body origins, shoulder centres are the ``LS``/``RS`` frames and
    the upper-trunk points are the thorax-attached ``BackLeft``/``BackRight``
    marker attachments (``marker_offsets`` carries a receipt's calibrated
    offsets).

    Args:
        spec: Full-body spec mapping (``frames``, ``marker_attachments``, ...).
        q: Joint coordinates (N, n_coords), radians/metres, spec coordinate order
            unless ``coordinate_names`` is given.

    Returns:
        Mapping with ``hip_l/hip_r``, ``shoulder_l/shoulder_r`` and, where the
        attachments exist, ``thorax_l/thorax_r`` (each (N, 3), metres).
    """
    from src.shared.python.motion_matching.identifiability import (
        body_poses_from_coordinates,
        compute_spec_marker_positions,
    )

    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim != 2:
        raise ValueError(f"q must be (N, n_coords), got shape {q_arr.shape}")
    frames = {
        f["name"]: (f["body"], np.asarray(f["placement"])) for f in spec["frames"]
    }
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("hip_l", "hip_r", "shoulder_l", "shoulder_r")
    }
    has_thorax = all(m in spec.get("marker_attachments", {}) for m in thorax_markers)
    if has_thorax:
        out["thorax_l"], out["thorax_r"] = [], []
    for qi in q_arr:
        poses = body_poses_from_coordinates(spec, qi, coordinate_names=coordinate_names)
        for key, body in zip(("hip_l", "hip_r"), hip_bodies):
            out[key].append(poses[body][:3, 3])
        for key, frame in zip(("shoulder_l", "shoulder_r"), shoulder_frames):
            body, placement = frames[frame]
            out[key].append((poses[body] @ placement)[:3, 3])
        if has_thorax:
            mk = compute_spec_marker_positions(
                spec,
                qi,
                marker_offsets=marker_offsets,
                coordinate_names=coordinate_names,
            )
            out["thorax_l"].append(mk.get(thorax_markers[0], np.full(3, np.nan)))
            out["thorax_r"].append(mk.get(thorax_markers[1], np.full(3, np.nan)))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _num(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _line_block(line: LineTurn, events: SwingEvents) -> dict[str, Any]:
    return {
        "status": line.status,
        "reason": line.reason,
        "points": list(line.points) if line.points else None,
        "valid_fraction": round(float(line.valid_fraction), 4),
        "filled_marker_frames": int(line.filled_frames),
        "address_deg": _num(line.value_at(events.address_time)),
        "top_deg": _num(line.value_at(events.top_time)),
        "impact_deg": _num(line.value_at(events.impact_time)),
        "max_backswing_deg": _num(
            line.max_backswing_deg(events.address_time, events.impact_time)
        ),
    }


def turn_source_block(metrics: TurnMetrics, events: SwingEvents, source: str) -> dict:
    """JSON-safe per-source block (NaN -> ``None``) for all lines of ``metrics``."""
    block: dict[str, Any] = {"source": source}
    for key, line in metrics.lines().items():
        block[key] = _line_block(line, events)
    return block


def build_turn_block(
    events: SwingEvents,
    *,
    markers: TurnMetrics | None = None,
    model: TurnMetrics | None = None,
    model_source: str = "model_fk",
) -> dict[str, Any]:
    """Build the receipt ``turn`` block shared by every matched-swing writer.

    The block carries shoulder-girdle, upper-trunk, pelvis and both X-factors at
    address, top, impact and the maximum backswing, for the capture markers and
    the model side by side.  It is descriptive only: it adds no thresholds.

    Args:
        events: Swing events (capture timeline) defining the three instants.
        markers: Turn lines computed from the capture markers, if available.
        model: Turn lines computed from model FK, if available.
        model_source: Label for the model-side provenance.

    Returns:
        A JSON-safe dict validated by ``validate_turn_block``.
    """
    block: dict[str, Any] = {
        "schema": TURN_BLOCK_SCHEMA,
        "frame_convention": FRAME_CONVENTION,
        "event_times_s": {
            "address": float(events.address_time),
            "top": float(events.top_time),
            "impact": float(events.impact_time),
        },
        "definitions": {
            "shoulder_girdle": "ShoulderBack markers (scapular/acromial) or shoulder joint centres",
            "upper_trunk": "BackLeft/BackRight markers or thorax-fixed model points",
            "pelvis": "WaistLeft/WaistRight markers or hip joint centres",
            "x_factor": "upper_trunk turn minus pelvis turn",
            "x_factor_shoulder_girdle": "shoulder_girdle turn minus pelvis turn",
        },
        "markers": (
            turn_source_block(markers, events, "capture_markers")
            if markers is not None
            else None
        ),
        "model": (
            turn_source_block(model, events, model_source)
            if model is not None
            else None
        ),
    }
    validate_turn_block(block)
    return block


TURN_BLOCK_SCHEMA = "turn_block/v1"
_BLOCK_LINES = (*LINE_NAMES, "x_factor", "x_factor_shoulder_girdle")
_BLOCK_VALUES = ("address_deg", "top_deg", "impact_deg", "max_backswing_deg")
_STATUSES = ("ok", "partial", "unavailable")


def validate_turn_block(block: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``block`` is a well-formed turn block.

    Checks structure only (schema tag, event times, all five lines on each
    present side, statuses, numeric-or-null values, and that an unavailable line
    carries a reason and no numbers).  It applies no acceptance thresholds.
    """
    if not isinstance(block, Mapping):
        raise TypeError("turn block must be a mapping")
    if block.get("schema") != TURN_BLOCK_SCHEMA:
        raise ValueError(f"turn block schema must be {TURN_BLOCK_SCHEMA!r}")
    events = block.get("event_times_s")
    if not isinstance(events, Mapping) or set(events) != {"address", "top", "impact"}:
        raise ValueError("turn block needs event_times_s with address/top/impact")
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in events.values()):
        raise ValueError("turn block event times must be finite numbers")
    if block.get("markers") is None and block.get("model") is None:
        raise ValueError("turn block needs at least one of markers or model")
    for side in ("markers", "model"):
        data = block.get(side)
        if data is None:
            continue
        if not isinstance(data, Mapping) or not data.get("source"):
            raise ValueError(f"turn block {side} needs a source label")
        for line in _BLOCK_LINES:
            entry = data.get(line)
            if not isinstance(entry, Mapping):
                raise ValueError(f"turn block {side}.{line} is missing")
            if entry.get("status") not in _STATUSES:
                raise ValueError(f"turn block {side}.{line}.status invalid")
            for key in _BLOCK_VALUES:
                if key not in entry:
                    raise ValueError(f"turn block {side}.{line}.{key} is missing")
                val = entry[key]
                if val is not None and not (
                    isinstance(val, (int, float)) and np.isfinite(val)
                ):
                    raise ValueError(
                        f"turn block {side}.{line}.{key} must be finite or null"
                    )
            if entry["status"] == "unavailable":
                if not entry.get("reason"):
                    raise ValueError(
                        f"turn block {side}.{line} unavailable without reason"
                    )
                if any(entry[k] is not None for k in _BLOCK_VALUES):
                    raise ValueError(
                        f"turn block {side}.{line} unavailable but has values"
                    )
