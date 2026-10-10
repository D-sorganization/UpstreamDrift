"""Swing tracking of the Rajagopal golf models holding the shared club (OSV-9).

Each frame of a generated full-body swing (the generated model's own OpenSim
forward kinematics, :class:`msk_club_calibration.GeneratedSwing`) gives the
club pose and body landmarks in the Rajagopal world. Inverse kinematics of
the club-less skeleton then places the calibrated lead-hand grip frame on the
club's lead grip frame (2 mm / 0.02 rad weights: the lead weld carries the
club), the trail-hand grip frame near its grip (1 cm / 0.1 rad: the trail
weld constraint closes the remainder) and follows the landmarks (4 cm), warm
started from the previous frame and bounded by the model's coordinate ranges.

This is kinematic tracking for display and inverse analyses, not forward
dynamics. It needs OpenSim and SciPy.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python import msk_club as mc
from src.engines.physics_engines.opensim.python import msk_club_calibration as cal

LEAD_POSITION_WEIGHT_M = 0.002
LEAD_ROTATION_WEIGHT_RAD = 0.02
TRAIL_POSITION_WEIGHT_M = 0.01
TRAIL_ROTATION_WEIGHT_RAD = 0.1
NEUTRAL_WEIGHT_RAD = 4.0
SMOOTH_WEIGHT_RAD = 0.2  # pull toward the previous frame


@dataclass(frozen=True)
class TrackedFrame:
    """One tracked frame: coordinates and the residual grip errors."""

    q: dict[str, float]
    lead_grip_error_m: float
    trail_grip_gap_m: float
    landmark_rms_m: float


def _rotation_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()


class _Tracker:
    """Weighted IK residual at one frame (LOD wrapper over the probe)."""

    def __init__(
        self,
        probe: cal.PoseProbe,
        names: list[str],
        hand_frames: dict[str, np.ndarray],
    ) -> None:
        self.probe, self.names, self.hand_frames = probe, names, hand_frames
        self.rotational = np.array([not n.startswith("pelvis_t") for n in names])
        self.grips: dict[str, np.ndarray] = {}
        self.landmarks: dict[str, np.ndarray] = {}
        self.previous = np.zeros(len(names))

    def hand_grip(self, side: str) -> np.ndarray:
        return self.probe.body(mc.HAND_BODIES[side]) @ self.hand_frames[side]

    def __call__(self, q: np.ndarray) -> np.ndarray:
        self.probe.set(dict(zip(self.names, q, strict=True)))
        lead, trail = self.hand_grip("L"), self.hand_grip("R")
        parts = [
            (lead[:3, 3] - self.grips["L"][:3, 3]) / LEAD_POSITION_WEIGHT_M,
            _rotation_error(lead, self.grips["L"]) / LEAD_ROTATION_WEIGHT_RAD,
            (trail[:3, 3] - self.grips["R"][:3, 3]) / TRAIL_POSITION_WEIGHT_M,
            _rotation_error(trail, self.grips["R"]) / TRAIL_ROTATION_WEIGHT_RAD,
        ]
        for body, target in self.landmarks.items():
            residual = self.probe.body(body)[:3, 3] - target
            parts.append(residual / cal.landmark_weight(body))
        parts.append(np.asarray(q)[self.rotational] / NEUTRAL_WEIGHT_RAD)
        parts.append(
            (np.asarray(q) - self.previous)[self.rotational] / SMOOTH_WEIGHT_RAD
        )
        return np.concatenate(parts)

    def measure(self, q: np.ndarray) -> TrackedFrame:
        self.probe.set(dict(zip(self.names, q, strict=True)))
        errors = {s: self.hand_grip(s)[:3, 3] - self.grips[s][:3, 3] for s in "LR"}
        squared = [
            np.sum((self.probe.body(b)[:3, 3] - t) ** 2)
            for b, t in self.landmarks.items()
        ]
        return TrackedFrame(
            q=dict(zip(self.names, (float(v) for v in q), strict=True)),
            lead_grip_error_m=float(np.linalg.norm(errors["L"])),
            trail_grip_gap_m=float(np.linalg.norm(errors["R"])),
            landmark_rms_m=float(np.sqrt(np.mean(squared))),
        )


def track_swing(
    model_path: Path, rows: np.ndarray, *, club: str = "driver"
) -> list[TrackedFrame]:
    """Track generated swing coordinates ``rows`` (frames x coordinates).

    ``rows`` follow the generated model's coordinate order (the club-face
    fixtures' order). The model must have a committed grip calibration.
    Raises ``ValueError`` for a non-2-D ``rows`` and ``FileNotFoundError``
    for a missing model.
    """
    from defusedxml import ElementTree as SafeET
    from scipy.optimize import least_squares

    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D (frames x coordinates), got {rows.shape}")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    calibration = mc.load_calibration(model_path.stem, club)
    shared = mc.load_msk_club(club)
    generated = cal.GeneratedSwing(club)
    with tempfile.TemporaryDirectory() as tmp:
        skeleton = cal.skeleton_without_club(model_path, Path(tmp))
        names = cal.free_coordinates(SafeET.parse(str(skeleton)).getroot())
        probe = cal.PoseProbe(skeleton)
    bounds = np.array([probe.coordinate_range(n) for n in names])
    tracker = _Tracker(probe, names, calibration.hand_frames)
    q = np.clip([calibration.address_q.get(n, 0.0) for n in names], *bounds.T)
    frames: list[TrackedFrame] = []
    for row in rows:
        targets = generated.targets(row)
        tracker.grips = cal.grip_frames(shared, targets.club_in_ground)
        tracker.landmarks = targets.landmarks
        tracker.previous = q
        fit = least_squares(tracker, q, bounds=tuple(bounds.T), x_scale="jac")
        q = fit.x
        frames.append(tracker.measure(q))
    return frames


def coordinate_rows(frames: Sequence[TrackedFrame]) -> tuple[list[str], np.ndarray]:
    """``(names, frames x names)`` array of tracked coordinates."""
    if not frames:
        raise ValueError("no tracked frames")
    names = list(frames[0].q)
    return names, np.array([[f.q[n] for n in names] for f in frames])
