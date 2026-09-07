"""Swing events and tempo from one camera's 2-D track (#9663).

A single view cannot be reconstructed, but the hand-speed profile in the
image plane carries the same events (address, top, peak, finish) and tempo
as the 3-D one, and the hip and shoulder lines carry their image-plane
tilt. Everything is normalised by the subject's box height so the numbers
are comparable across takes and resolutions; nothing here is in metres and
the record says so.

Reuses the smoother and event logic of :mod:`.analytics` on a 2-D series:
the same code, a different unit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

from .analytics import SwingEvents, SwingSeries, detect_events
from .temporal import SmootherOptions, smooth

Array = npt.NDArray[np.float64]

HAND_ACCELERATION_SIGMA_BH = 400.0  # box heights / s^2, the 3-D prior scaled
HAND_POSITION_SIGMA_BH = 0.005
MIN_FRAMES = 3
SERIES_JOINTS = (
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
)


class Swing2DSummary(BaseModel):
    """What one view can say on its own: events, tempo, normalised speed."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "swing-2d/1.0.0"
    view: str
    frames: int
    frames_with_pose: int
    fps: float
    units: str = "subject box heights (image plane), not metres"
    peak_hand_speed_bh_per_s: float
    peak_hand_speed_frame: int
    max_shoulder_tilt_deg: float
    max_hip_tilt_deg: float
    events: SwingEvents


def _joint_track(
    payload: dict[str, Any], name: str, *, min_confidence: float
) -> tuple[Array, npt.NDArray[np.bool_]]:
    """``(T, 2)`` pixels of ``name`` and a mask of confident frames."""
    names = list(payload["detector_layout"]["keypoint_names"])
    require(name in names, "joint missing from layout", name)
    k = names.index(name)
    fps = float(payload["fps"])
    total = int(payload["frames_total"])
    track = np.full((total, 2), np.nan)
    ok = np.zeros(total, dtype=bool)
    for row in payload["frames"]:
        index = int(round(float(row["time_s"]) * fps))
        if 0 <= index < total and float(row["confidence"][k]) >= min_confidence:
            track[index] = row["keypoints_px"][k]
            ok[index] = True
    return track, ok


def _box_height(payload: dict[str, Any], *, min_confidence: float) -> float:
    """Median vertical extent of confident joints: the subject's scale in px."""
    heights = []
    for row in payload["frames"]:
        px = np.asarray(row["keypoints_px"], dtype=float)
        conf = np.asarray(row["confidence"], dtype=float)
        ys = px[conf >= min_confidence][:, 1]
        if ys.size >= 2 and ys.max() > ys.min():
            heights.append(float(ys.max() - ys.min()))
    require(bool(heights), "no frame has two confident joints")
    return float(np.median(heights))


def _line_tilt_deg(left: Array, right: Array) -> Array:
    """Image-plane angle of the left->right line, unwrapped from frame 0."""
    d = right - left
    angle = np.arctan2(d[:, 1], d[:, 0])
    angle = np.where(np.isfinite(angle), angle, 0.0)
    return np.degrees(np.unwrap(angle - angle[0]))


def _fill(track: Array, ok: npt.NDArray[np.bool_]) -> Array:
    """Linear interpolation over missing frames (edges held)."""
    out = track.copy()
    idx = np.flatnonzero(ok)
    require(idx.size >= 2, "need at least two confident frames")
    for c in range(track.shape[1]):
        out[:, c] = np.interp(np.arange(track.shape[0]), idx, track[idx, c])
    return out


def series_2d(
    payload: dict[str, Any], *, min_confidence: float = 0.5
) -> tuple[SwingSeries, float]:
    """A :class:`SwingSeries` in box-height units from one view's 2-D track."""
    fps = float(payload["fps"])
    require(fps > 0, "fps must be positive", fps)
    require(int(payload["frames_total"]) >= MIN_FRAMES, "need at least three frames")
    scale = _box_height(payload, min_confidence=min_confidence)
    tracks = {
        name: _fill(*_joint_track(payload, name, min_confidence=min_confidence))
        for name in SERIES_JOINTS
    }
    hands = 0.5 * (tracks["left_wrist"] + tracks["right_wrist"]) / scale
    options = SmootherOptions(
        acceleration_sigma=HAND_ACCELERATION_SIGMA_BH,
        measurement_sigma=HAND_POSITION_SIGMA_BH,
    )
    fit = smooth(hands, None, fps, options)
    velocity = np.gradient(fit.values, 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    unc = np.sqrt(2.0) * np.linalg.norm(fit.uncertainty, axis=1) * fps / 2.0
    hips = _line_tilt_deg(tracks["left_hip"], tracks["right_hip"])
    shoulders = _line_tilt_deg(tracks["left_shoulder"], tracks["right_shoulder"])
    series = SwingSeries(
        time_s=np.arange(hands.shape[0]) / fps,
        pelvis_turn_deg=hips,
        shoulder_turn_deg=shoulders,
        x_factor_deg=shoulders - hips,
        hand_speed_mps=speed,  # box heights / s here, see Swing2DSummary.units
        hand_speed_uncertainty_mps=unc,
    )
    return series, scale


def summarize_view_2d(
    payload: dict[str, Any], *, min_confidence: float = 0.5
) -> Swing2DSummary:
    """Events, tempo and normalised peak speed for one view."""
    series, _ = series_2d(payload, min_confidence=min_confidence)
    fps = float(payload["fps"])
    events = detect_events(series, fps)
    peak = int(np.argmax(series.hand_speed_mps))
    return Swing2DSummary(
        view=str(payload["view"]),
        frames=int(payload["frames_total"]),
        frames_with_pose=int(payload.get("frames_with_pose", len(payload["frames"]))),
        fps=fps,
        peak_hand_speed_bh_per_s=float(series.hand_speed_mps[peak]),
        peak_hand_speed_frame=peak,
        max_shoulder_tilt_deg=float(np.max(np.abs(series.shoulder_turn_deg))),
        max_hip_tilt_deg=float(np.max(np.abs(series.pelvis_turn_deg))),
        events=events,
    )


def analyze_session_2d(
    session_dir: Path,
    *,
    observations_dir: str = "observations",
    min_confidence: float = 0.5,
    views: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write ``analysis_2d/<view>.json`` for every (or the named) ingested view."""
    obs_dir = session_dir / observations_dir
    require(obs_dir.is_dir(), "session has no observations", str(obs_dir))
    out_dir = session_dir / "analysis_2d"
    out_dir.mkdir(exist_ok=True)
    written: dict[str, Path] = {}
    for path in sorted(obs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "view" not in payload or "frames" not in payload:
            continue  # the ingest index, not a view
        if views and payload["view"] not in views:
            continue
        summary = summarize_view_2d(payload, min_confidence=min_confidence)
        target = out_dir / f"{summary.view}.json"
        target.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        written[summary.view] = target
    require(bool(written), "no view could be analysed")
    return written
