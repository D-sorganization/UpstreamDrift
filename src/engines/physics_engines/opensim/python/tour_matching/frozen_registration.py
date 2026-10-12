"""Training-only registration lineage around the existing rigid fit provider.

Neither explicit anchors nor a coordinate gauge proves anatomical equivalence.
This module never selects labels, fills missing observations or fits holdout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np

from src.shared.python.motion_matching.marker_calibration import Pose
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)

from . import registration as provider
from .registration import (
    CaptureRegistration,
    compute_capture_registration,
    register_points,
)

Interpretation = Literal["coordinate-gauge", "declared-unverified"]


@dataclass(frozen=True)
class FrozenCaptureRegistration:
    """A fitted rigid transform with immutable training lineage, not acceptance."""

    transform: CaptureRegistration
    training_source_sha256: str
    training_frames: tuple[int, ...]
    training_times_s: tuple[float, ...]
    anchor_labels: tuple[str, ...]
    anchor_provenance: tuple[tuple[str, str], ...]
    training_points_sha256: str
    target_geometry_sha256: str
    target_points_sha256: str
    provider_sha256: str
    interpretation: Interpretation
    training_rms_m: float

    @property
    def anatomically_qualified(self) -> bool:
        """Fitting or authored provenance does not prove landmark equivalence."""
        return False

    @property
    def identity_sha256(self) -> str:
        """Bind transform, declared frames, training evidence and implementation."""
        payload = dict(self.__dict__)
        payload["transform"] = {
            "rotation": self.transform.rotation.tolist(),
            "translation": self.transform.translation.tolist(),
            "source_frame": self.transform.source_frame,
            "target_frame": self.transform.target_frame,
        }
        payload["schema"] = "frozen-training-registration/1"
        return _digest(payload)


@dataclass(frozen=True)
class RegisteredCapture:
    """Existing capture arrays accompanied by the applied transform lineage."""

    capture: TourCapture
    registration_sha256: str
    source_frame: str
    target_frame: str


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _training_indices(capture: TourCapture, frames: Sequence[int]) -> tuple[int, ...]:
    if len(frames) == 0 or any(
        isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer))
        for f in frames
    ):
        raise ValueError("Explicit integer training frames are required")
    indices = tuple(int(f) for f in frames)
    if any(f < 0 or f >= capture.frames for f in indices) or any(
        a >= b for a, b in zip(indices, indices[1:], strict=False)
    ):
        raise ValueError(
            "Original training frames must be in range and strictly increasing"
        )
    return indices


def _anchor_arrays(
    capture: TourCapture,
    indices: tuple[int, ...],
    targets: Mapping[str, Sequence[float]],
    provenance: Mapping[str, str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if len(targets) < 3 or not set(targets).issubset(capture.labels):
        raise ValueError("At least three explicit observed anchor labels are required")
    if set(provenance) != set(targets) or any(
        not isinstance(value, str) or not value.strip() for value in provenance.values()
    ):
        raise ValueError("Every correspondence needs explicit anchor provenance")
    labels = tuple(label for label in capture.labels if label in targets)
    columns = [capture.index(label) for label in labels]
    if not capture.valid[np.ix_(indices, columns)].all():
        raise ValueError("Every declared training anchor must be observed")
    target = np.asarray([targets[label] for label in labels], dtype=float)
    if target.shape != (len(labels), 3) or not np.isfinite(target).all():
        raise ValueError("Target anchors must be finite 3-D metre positions")
    points = capture.points_m[np.ix_(indices, columns)].reshape(-1, 3)
    return labels, points, np.tile(target, (len(indices), 1))


def fit_training_registration(
    capture: TourCapture,
    training_frames: Sequence[int],
    target_points_m: Mapping[str, Sequence[float]],
    *,
    target_geometry_sha256: str,
    anchor_provenance: Mapping[str, str],
    target_frame: str,
    interpretation: Interpretation,
    source_frame: str,
) -> FrozenCaptureRegistration:
    """Fit declared static training anchors; all other observations are unused.

    Multiple frames assert one static anchor arrangement; residuals expose any
    disagreement. No noise or anatomical acceptance threshold is inferred.
    """
    if interpretation not in ("coordinate-gauge", "declared-unverified"):
        raise ValueError(
            "Registration interpretation must remain explicitly unqualified"
        )
    if capture.source_sha256 is None:
        raise ValueError("Pinned source SHA-256 identity is required")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in (target_geometry_sha256, capture.source_sha256)
    ):
        raise ValueError(
            "Pinned source and target geometry SHA-256 identities are required"
        )
    if not all(
        isinstance(value, str) and value.strip()
        for value in (source_frame, target_frame)
    ):
        raise ValueError("Explicit source and target frame identifiers are required")
    indices = _training_indices(capture, training_frames)
    labels, source, target = _anchor_arrays(
        capture, indices, target_points_m, anchor_provenance
    )
    transform = compute_capture_registration(
        source, target, source_frame=source_frame, target_frame=target_frame
    )
    residual = register_points(source, transform) - target
    implementation = hashlib.sha256(Path(__file__).read_bytes())
    implementation.update(Path(provider.__file__).read_bytes())
    return FrozenCaptureRegistration(
        transform=transform,
        training_source_sha256=capture.source_sha256,
        training_frames=indices,
        training_times_s=tuple(float(capture.time_s[f]) for f in indices),
        anchor_labels=labels,
        anchor_provenance=tuple((label, anchor_provenance[label]) for label in labels),
        training_points_sha256=_digest(source.tolist()),
        target_geometry_sha256=target_geometry_sha256,
        target_points_sha256=_digest(target.tolist()),
        provider_sha256=implementation.hexdigest(),
        interpretation=interpretation,
        training_rms_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
    )


def apply_frozen_registration(
    capture: TourCapture,
    registration: FrozenCaptureRegistration,
    *,
    source_frame: str,
) -> RegisteredCapture:
    """Apply fixed geometry only; preserve raw identity, missingness and clock."""
    if source_frame != registration.transform.source_frame:
        raise ValueError("Observation source frame differs from frozen registration")
    points = np.full_like(capture.points_m, np.nan)
    points[capture.valid] = register_points(
        capture.points_m[capture.valid], registration.transform
    )
    transformed = TourCapture(
        capture.time_s, capture.labels, points, capture.valid, capture.source_sha256
    )
    return RegisteredCapture(
        transformed,
        registration.identity_sha256,
        source_frame,
        registration.transform.target_frame,
    )


def pelvis_coordinate_gauge(
    capture: TourCapture,
    reference_pose: Pose,
    *,
    target_left_axis_local: np.ndarray,
    target_geometry_sha256: str,
    training_frame: int,
) -> FrozenCaptureRegistration:
    """Declare a training-only origin/yaw gauge, without landmark equivalence.

    The waist centroid maps to the native reference pelvis origin. Declared
    capture Y-up and horizontal waist left-minus-right map to the reference
    body's Y axis and an explicit native local left axis. This is an authored
    coordinate gauge, not measured static pose, anatomy or ground calibration.
    """
    index = _training_indices(capture, (training_frame,))[0]
    if capture.source_sha256 is None:
        raise ValueError("Pinned source SHA-256 identity is required")
    labels = MARKER_SEGMENTS["pelvis"]
    if not set(labels).issubset(capture.labels):
        raise ValueError("All four declared waist gauge markers are required")
    columns = [capture.index(label) for label in labels]
    if not capture.valid[index, columns].all():
        raise ValueError("All training waist gauge markers must be observed")
    points = capture.points_m[index, columns]
    centre = points.mean(axis=0)
    left = points[labels.index("WaistLeft")] - points[labels.index("WaistRight")]
    left[1] = 0.0
    if np.linalg.norm(left) < 1e-6:
        raise ValueError("Training waist horizontal direction is degenerate")
    left /= np.linalg.norm(left)
    target_left = np.asarray(target_left_axis_local, dtype=float)
    if (
        target_left.shape != (3,)
        or not np.isfinite(target_left).all()
        or np.linalg.norm(target_left) < 1e-6
        or abs(target_left[1]) > 1e-8
    ):
        raise ValueError(
            "Native reference left axis must be finite and perpendicular to local Y"
        )
    target_left = target_left / np.linalg.norm(target_left)
    reference = CaptureRegistration(*reference_pose)
    source_frame = "capture-world:" + capture.source_sha256
    target_frame = "native-ground:" + target_geometry_sha256
    gauge = compute_capture_registration(
        np.array([centre, centre + [0, 1, 0], centre + left]),
        np.array(
            [
                reference.translation,
                reference.translation + reference.rotation[:, 1],
                reference.translation + reference.rotation @ target_left,
            ]
        ),
        source_frame=source_frame,
        target_frame=target_frame,
    )
    targets = dict(zip(labels, register_points(points, gauge), strict=True))
    basis = "Training-only centroid/Y-up/left-axis coordinate gauge; no donor landmark equivalence"
    return fit_training_registration(
        capture,
        (index,),
        targets,
        target_geometry_sha256=target_geometry_sha256,
        anchor_provenance=dict.fromkeys(labels, basis),
        target_frame=target_frame,
        source_frame=source_frame,
        interpretation="coordinate-gauge",
    )
