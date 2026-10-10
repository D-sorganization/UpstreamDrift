"""Thorax / shoulder-girdle turn split for the trajectory IK (#12042, slice 7).

The capture's upper-trunk line (``BackLeft``/``BackRight``) and shoulder-girdle
line (``LShoulderBack``/``RShoulderBack``) turn by different amounts: at the top
of capture-A the shoulder girdle adds about 11 deg of scapular rotation on top
of the thorax.  With every marker weighted equally the MuJoCo IK put that turn
into the thorax instead (upper trunk +18 deg, scapulae about 3 deg), because
thorax yaw is poorly identified by marker positions: a 15 deg change of thorax
yaw moves the whole-window marker RMS by under 1 mm.

Two terms resolve the split, measured in
``docs/research/simscape_matching_reference/simscape_matching_reference.tex``
(section "Thorax and Shoulder-Girdle Turn Split"):

* a thorax-orientation residual: the ``Spine`` frame's calibrated
  BackRight->BackLeft axis is pulled onto the capture's BackRight->BackLeft
  direction (an axis target, like the club-face residual of OSV-10); and
* a higher weight on the two shoulder-girdle markers, the only observation of
  scapular protraction, which the arm and club markers otherwise outvote.

Both act only in the trajectory, consistency, shooting and ZMP re-solves; the
address calibration and segment scaling stay marker-only.  The split is off by
default (``--thorax-weight 0 --shoulder-girdle-weight 1``): it brings the IK
turn within bounds but regresses the forward-dynamics replay, see the
constants below.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Model frame carrying the upper-trunk markers (thorax proxy).
THORAX_FRAME = "Spine"
#: Capture markers of the upper-trunk line, left then right.
THORAX_LINE: tuple[str, str] = ("BackLeft", "BackRight")
#: Capture markers of the shoulder-girdle line (scapular / acromial).
SHOULDER_GIRDLE_MARKERS: tuple[str, str] = ("LShoulderBack", "RShoulderBack")
#: Thorax-orientation residual weight that meets the IK turn bounds on
#: capture-A and capture-B (unit-vector residual, squared weight).
THORAX_AXIS_WEIGHT = 0.3
#: Shoulder-girdle marker weight paired with ``THORAX_AXIS_WEIGHT`` (others 1).
SHOULDER_GIRDLE_MARKER_WEIGHT = 5.0
#: Pipeline defaults: the split is opt-in.  The IK-passing setting above makes
#: the forward-dynamics replay worse on both captures (driver 61.6 -> 69.1 mm,
#: iron 47.0 -> 71.6 mm), so the canonical receipts keep the marker-only
#: thorax until the follow-through regression is resolved.
DEFAULT_THORAX_WEIGHT = 0.0
DEFAULT_SHOULDER_GIRDLE_WEIGHT = 1.0
#: Shortest capture line that still defines a direction, metres.
MIN_LINE_LENGTH_M = 0.05
#: Default cosine taper after the detected impact for windowed terms, seconds.
IMPACT_TAPER_S = 0.05
#: Least fraction of valid frames for a club marker to vote on impact.
MIN_CLUB_VALID_FRACTION = 0.5


@dataclass(frozen=True)
class SplitWindow:
    """Which split terms stop at the detected impact (plus ``taper_s``).

    The default applies both terms over the whole capture, as in slice 7.
    """

    thorax: bool = False
    girdle: bool = False
    taper_s: float = IMPACT_TAPER_S

    @property
    def active(self) -> bool:
        return self.thorax or self.girdle


AxisTarget = tuple[tuple[float, float, float], tuple[float, float, float], float]


def _unit(vector: np.ndarray) -> tuple[float, float, float] | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < MIN_LINE_LENGTH_M:
        return None
    unit = vector / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _check_weight(weight: float, name: str) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(weight).__name__}")
    value = float(weight)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {weight}")
    return value


def thorax_body_axis(
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    frame: str = THORAX_FRAME,
) -> tuple[float, float, float]:
    """Unit BackRight->BackLeft axis in the thorax frame, from the attachments.

    Raises:
        ValueError: when a line marker is missing, rides another body, or the
            two attachments coincide.
    """
    left, right = THORAX_LINE
    for label in (left, right):
        if label not in attachments:
            raise ValueError(f"attachment for {label} is missing")
        if attachments[label][0] != frame:
            raise ValueError(
                f"{label} rides {attachments[label][0]!r}, not the thorax "
                f"frame {frame!r}"
            )
    axis = _unit(
        np.asarray(attachments[left][1], dtype=float)
        - np.asarray(attachments[right][1], dtype=float)
    )
    if axis is None:
        raise ValueError("BackLeft and BackRight attachments coincide")
    return axis


def thorax_axis_targets(
    points: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    weight: float = THORAX_AXIS_WEIGHT,
    factors: Sequence[float] | None = None,
) -> list[dict[str, AxisTarget] | None] | None:
    """Per-frame axis targets aligning the thorax with the upper-trunk line.

    Args:
        points: Capture markers (frames, markers, 3) in the native world.
        valid: Validity mask (frames, markers).
        labels: Marker labels for the columns of ``points``.
        attachments: Calibrated attachments ``{label: (frame, offset_m)}``.
        weight: Residual weight; 0 disables the target.
        factors: Optional per-frame scale in [0, 1] of ``weight`` (see
            :func:`window_factors`); a frame with factor 0 has no target.

    Returns:
        One entry per frame, ``None`` where either line marker is missing or
        the line is degenerate; ``None`` overall when ``weight`` is 0.
        Postcondition: every target pairs the same body axis with a unit
        world direction.
    """
    w = _check_weight(weight, "thorax weight")
    pts = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if pts.ndim != 3 or pts.shape[2] != 3 or mask.shape != pts.shape[:2]:
        raise ValueError("points must be (frames, markers, 3) matching valid")
    if len(labels) != pts.shape[1]:
        raise ValueError("labels must name every marker column")
    scale = _frame_factors(factors, pts.shape[0])
    if w == 0.0:
        return None
    left, right = (list(labels).index(name) for name in THORAX_LINE)
    body_axis = thorax_body_axis(attachments)
    out: list[dict[str, AxisTarget] | None] = []
    for frame in range(pts.shape[0]):
        direction = None
        if mask[frame, left] and mask[frame, right] and scale[frame] > 0:
            direction = _unit(pts[frame, left] - pts[frame, right])
        out.append(
            None
            if direction is None
            else {THORAX_FRAME: (body_axis, direction, w * scale[frame])}
        )
    return out


def _frame_factors(factors: Sequence[float] | None, frames: int) -> np.ndarray:
    if factors is None:
        return np.ones(frames)
    scale = np.asarray(factors, dtype=float)
    if scale.shape != (frames,):
        raise ValueError("factors must hold one value per frame")
    if not np.all((scale >= 0) & (scale <= 1)):
        raise ValueError("factors must lie in [0, 1]")
    return scale


def capture_impact_time(
    times: Sequence[float] | np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
) -> float:
    """Impact time of a capture from its club markers, never a fixed time.

    Each club marker with at least ``MIN_CLUB_VALID_FRACTION`` valid frames is
    gap-interpolated and passed to :func:`gaze.impact_index` (lowest point
    near peak speed); the median frame of those votes is the impact.

    Raises:
        ValueError: when no club marker has enough valid frames.
    """
    from src.shared.python.motion_matching.gaze import (  # noqa: PLC0415
        impact_index,
    )
    from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: PLC0415
        MARKER_SEGMENTS,
    )

    t = np.asarray(times, dtype=float)
    pts = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    votes = []
    for label in MARKER_SEGMENTS["club"]:
        if label not in labels:
            continue
        j = list(labels).index(label)
        ok = mask[:, j] & np.isfinite(pts[:, j]).all(axis=1)
        if ok.mean() < MIN_CLUB_VALID_FRACTION:
            continue
        track = np.stack(
            [np.interp(t, t[ok], pts[ok, j, axis]) for axis in range(3)], axis=1
        )
        votes.append(impact_index(t, track))
    if not votes:
        raise ValueError("no club marker has enough valid frames to find impact")
    return float(t[int(np.sort(votes)[(len(votes) - 1) // 2])])


def window_factors(
    times: Sequence[float] | np.ndarray, end_s: float, taper_s: float
) -> np.ndarray:
    """1 up to ``end_s``, a half-cosine down to 0 over ``taper_s``, then 0."""
    if not np.isfinite(taper_s) or taper_s < 0:
        raise ValueError(f"taper must be finite and >= 0, got {taper_s}")
    t = np.asarray(times, dtype=float)
    if taper_s == 0:
        return (t <= end_s).astype(float)
    x = np.clip((t - end_s) / taper_s, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def shoulder_girdle_weights(
    labels: Sequence[str], weight: float = SHOULDER_GIRDLE_MARKER_WEIGHT
) -> dict[str, float]:
    """Marker weights for the shoulder-girdle markers present in ``labels``.

    A weight of 1 (the default marker weight) returns an empty mapping.
    """
    w = _check_weight(weight, "shoulder-girdle weight")
    if w == 1.0:
        return {}
    return {label: w for label in SHOULDER_GIRDLE_MARKERS if label in labels}


def shoulder_girdle_weights_per_frame(
    labels: Sequence[str], weight: float, factors: Sequence[float]
) -> list[dict[str, float]]:
    """Per-frame shoulder-girdle weights tapering from ``weight`` to 1."""
    w = _check_weight(weight, "shoulder-girdle weight")
    scale = _frame_factors(factors, len(factors))
    return [shoulder_girdle_weights(labels, 1.0 + (w - 1.0) * float(f)) for f in scale]


def lane_axis_targets(lane: Any) -> list[dict[str, Any]] | None:
    """Union of a lane's face and thorax axis targets for the re-solves.

    Lanes without the attributes (or with non-list stand-ins) contribute
    nothing, so lightweight lanes keep working.
    """
    from src.shared.python.motion_matching.club_face_target import (  # noqa: PLC0415
        merge_axis_targets,
    )

    lists = []
    for name in ("face_targets", "thorax_targets"):
        value = getattr(lane, name, None)
        if isinstance(value, (list, tuple)):
            lists.append(value)
    if len(lists) == 1:
        # One source: hand it through unchanged, as before the split existed.
        return lists[0]
    return merge_axis_targets(*lists)


def lane_split_weights(lane: Any) -> dict[str, float] | None:
    """The lane's shoulder-girdle marker weights, or ``None`` when unset."""
    value = getattr(lane, "split_marker_weights", None)
    if isinstance(value, Mapping) and value:
        return dict(value)
    return None


def lane_split_weights_per_frame(lane: Any) -> list[dict[str, float]] | None:
    """The lane's windowed shoulder-girdle weights, or ``None`` when unset."""
    value = getattr(lane, "split_marker_weights_per_frame", None)
    return value if isinstance(value, list) else None


def split_window_from_args(args: argparse.Namespace) -> SplitWindow:
    """The :class:`SplitWindow` selected on the command line."""
    return SplitWindow(
        thorax=getattr(args, "thorax_window", "full") == "impact",
        girdle=getattr(args, "shoulder_girdle_window", "full") == "impact",
        taper_s=float(getattr(args, "split_taper_s", IMPACT_TAPER_S)),
    )


def add_turn_split_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--thorax-weight`` and ``--shoulder-girdle-weight`` to a parser."""

    def nonnegative(value: str) -> float:
        try:
            return _check_weight(float(value), "weight")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser.add_argument(
        "--thorax-weight",
        type=nonnegative,
        default=DEFAULT_THORAX_WEIGHT,
        help=(
            "thorax-orientation residual (#12042 slice 7): pulls the Spine "
            "frame's BackRight->BackLeft axis onto the capture line; 0 restores "
            "the marker-only thorax (default); 0.3 meets the IK turn bounds "
            "but regresses forward dynamics"
        ),
    )
    parser.add_argument(
        "--shoulder-girdle-weight",
        type=nonnegative,
        default=DEFAULT_SHOULDER_GIRDLE_WEIGHT,
        help=(
            "marker weight of LShoulderBack/RShoulderBack in the trajectory "
            "re-solves (#12042 slice 7); 1 (default) keeps equal weights; 5 "
            "pairs with --thorax-weight 0.3"
        ),
    )
    for flag, term in (
        ("--thorax-window", "thorax target"),
        ("--shoulder-girdle-window", "shoulder-girdle weights"),
    ):
        parser.add_argument(
            flag,
            choices=("full", "impact"),
            default="full",
            help=(
                f"'impact' applies the {term} from address to the impact "
                "detected from the club markers, then tapers it out over "
                "--split-taper-s; 'full' (default) applies it to every frame"
            ),
        )
    parser.add_argument(
        "--split-taper-s",
        type=nonnegative,
        default=IMPACT_TAPER_S,
        help="cosine taper after impact for windowed split terms, seconds",
    )


def turn_split_active(lane: Any) -> bool:
    """Whether either split term is switched on for ``lane``."""
    return (
        bool(getattr(lane, "thorax_targets", None))
        or bool(lane_split_weights(lane))
        or bool(lane_split_weights_per_frame(lane))
    )


def turn_split_report(lane: Any) -> dict[str, Any]:
    """Receipt block describing the active split terms."""
    targets = getattr(lane, "thorax_targets", None) or ()
    window = getattr(lane, "turn_split_window", None)
    impact = getattr(lane, "turn_split_impact_s", None)
    return {
        "thorax_frame": THORAX_FRAME,
        "thorax_line": list(THORAX_LINE),
        "thorax_weight": float(getattr(lane, "thorax_weight", 0.0)),
        "thorax_targeted_frames": int(sum(t is not None for t in targets)),
        "shoulder_girdle_markers": list(SHOULDER_GIRDLE_MARKERS),
        "shoulder_girdle_weight": float(getattr(lane, "shoulder_girdle_weight", 1.0)),
        "stages": ["trajectory", "consistency", "shooting", "zmp_filter"],
        "window": (
            {
                "thorax": window.thorax,
                "shoulder_girdle": window.girdle,
                "impact_s": impact,
                "taper_s": window.taper_s,
            }
            if isinstance(window, SplitWindow) and window.active
            else None
        ),
    }
