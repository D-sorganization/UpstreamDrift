"""Club marker registration from the club markers alone (#15296).

The static-seed registration (``club_markers.DERIVATION_METHOD``) places the
club cluster through a body-driven calibration, so it inherits the body
model's error. This module places the same cluster in the ``Clubhead`` frame
(origin at the head centre on the shaft axis, +x from the butt to the head)
from the six club markers only. It is a kinematic registration, not a
measurement of the club: see ``derive_club_only_offsets`` for what is assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.execution.club_markers import (
    HEAD_TRIAD,
    ClubMarkers,
    Offset,
)
from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.tour_capture_contract import (
    capture_kind,
    load_tour_capture,
)

CLUB_ONLY_METHOD = (
    "club markers alone: shaft axis through the two on-shaft markers, model head "
    "soled on the ground at address, roll set by the head-triad azimuth setting"
)
_TILT = math.radians(10.0)  # Clubhead frame about the shaft in the club body frame
_HEAD_AXES = (
    (0.0, math.sin(_TILT), -math.cos(_TILT)),
    (1.0, 0.0, 0.0),
    (0.0, -math.cos(_TILT), -math.sin(_TILT)),
)
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ClubOnlySettings:
    """Choices and tolerances of the club-only registration (lengths in m)."""

    shaft_axis_labels: tuple[str, str] = ("Marker_3:3:3", "Marker_3:3:1")  # butt, tip
    head_labels: tuple[str, ...] = HEAD_TRIAD
    other_labels: tuple[str, ...] = ("Marker_3:3:2",)  # on the club, off the shaft
    address_frames: int = 10  # leading frames with the club at rest at address
    min_frames: int = 30  # complete frames needed for the rigidity check
    still_tolerance_m: float = 0.002  # largest marker move inside the address window
    rigid_rms_tolerance_m: float = 0.006  # RMS change of inter-marker distances
    min_shaft_elevation_deg: float = 10.0
    min_roll_radius_m: float = 0.010  # head-triad centroid distance from the axis
    ground_height_m: float = 0.0
    hover_m: float = 0.0  # head lowest point above the ground at address
    # Azimuth of the head-triad centroid about the shaft, from +y toward +z. Not
    # identifiable from club markers: this keeps the roll of ``driver.json``.
    head_centroid_azimuth_deg: float = 3.92
    head_half_size_m: Vec3 = DRIVER.head_half_size_m  # model ellipsoid semi-axes
    head_axes: tuple[Vec3, Vec3, Vec3] = _HEAD_AXES  # those axes, Clubhead frame
    max_axis_marker_distance_m: float = 0.020  # on-shaft marker to the club axis

    def __post_init__(self) -> None:
        positive = (
            self.still_tolerance_m,
            self.rigid_rms_tolerance_m,
            self.min_shaft_elevation_deg,
            self.min_roll_radius_m,
            self.max_axis_marker_distance_m,
            *self.head_half_size_m,
        )
        finite = (*positive, self.ground_height_m, self.head_centroid_azimuth_deg)
        if not all(math.isfinite(v) for v in finite) or min(positive) <= 0:
            raise ValueError("every tolerance and head size must be finite and > 0")
        if not math.isfinite(self.hover_m) or self.hover_m < 0:
            raise ValueError("hover_m must be finite and >= 0")
        if not 2 <= self.address_frames <= self.min_frames:
            raise ValueError("need 2 <= address_frames <= min_frames")
        if len({*self.shaft_axis_labels, *self.head_labels}) != 5:
            raise ValueError("need two shaft axis labels and three other head labels")
        axes = np.asarray(self.head_axes, dtype=float)
        if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3)):
            raise ValueError("head_axes must be three orthonormal rows")


def head_support_m(rotation: np.ndarray, settings: ClubOnlySettings) -> float:
    """Height of the head centre above the model head ellipsoid's lowest point.

    ``rotation`` holds the ``Clubhead`` axes in a z-up world as columns.
    """
    vertical = np.asarray(settings.head_axes) @ np.asarray(rotation)[2]
    return float(np.linalg.norm(vertical * settings.head_half_size_m))


def _pair_distances(points: np.ndarray) -> np.ndarray:
    pairs = list(combinations(range(points.shape[-2]), 2))
    first, second = (points[..., [p[i] for p in pairs], :] for i in (0, 1))
    return np.linalg.norm(first - second, axis=-1)


def _address_cluster(
    labels: Sequence[str], points_m: np.ndarray, settings: ClubOnlySettings
) -> np.ndarray:
    """Mean address positions after the input and rigidity checks."""
    points = np.asarray(points_m, dtype=float)
    needed = {*settings.shaft_axis_labels, *settings.head_labels}
    if missing := sorted(needed - set(labels)):
        raise ValueError(f"club markers missing: {missing}")
    if points.ndim != 3 or points.shape[1:] != (len(labels), 3):
        raise ValueError("points_m must be shaped (frames, len(labels), 3)")
    if np.isinf(points).any():
        raise ValueError("inf samples: mark a missing sample NaN")
    window = points[: settings.address_frames]
    if len(window) < settings.address_frames or np.isnan(window).any():
        raise ValueError("address window must be complete and free of NaN")
    complete = points[~np.isnan(points).any(axis=(1, 2))]
    if len(complete) < settings.min_frames:
        raise ValueError(f"need {settings.min_frames} complete frames")
    address = window.mean(axis=0)
    moved = float(np.linalg.norm(window - address, axis=-1).max())
    if moved > settings.still_tolerance_m:
        raise ValueError(f"club not at rest at address: moves {moved:.4f} m")
    change = _pair_distances(complete) - _pair_distances(address)
    rms = float(np.sqrt(np.mean(change**2)))
    if rms > settings.rigid_rms_tolerance_m:
        raise ValueError(f"club cluster is not rigid: distance RMS {rms:.4f} m")
    return address


def derive_club_only_offsets(
    labels: Sequence[str],
    points_m: np.ndarray,
    settings: ClubOnlySettings | None = None,
) -> dict[str, Offset]:
    """``Clubhead``-frame offsets of ``labels`` from their trajectories alone.

    ``points_m`` is ``(frames, len(labels), 3)`` in a z-up world, NaN where a
    sample is missing, with the club at rest at address in the leading
    ``address_frames``. The rule, every choice a setting:

    1. Shape: mean marker positions over the address window.
    2. Shaft axis: the line through the two ``shaft_axis_labels``, +x toward
       the head. Assumes both sit on the shaft with equal standoff.
    3. Roll: the head-triad centroid lies at ``head_centroid_azimuth_deg``.
    4. Along-shaft datum: the origin is slid along the axis until the model
       head ellipsoid, centred on it, has its lowest point ``hover_m`` above
       ``ground_height_m``. Assumes a soled head and a level capture ground.

    Raises ``ValueError`` for missing labels, inf samples, an incomplete or
    moving address window, too few complete frames, inter-marker distances
    that change by more than ``rigid_rms_tolerance_m`` (RMS), a shaft flatter
    than ``min_shaft_elevation_deg``, a head triad not beyond the axis pair or
    a centroid nearer the axis than ``min_roll_radius_m``. Postconditions: the
    axis pair has y = z = 0 and the address shape is kept exactly.
    """
    settings = settings or ClubOnlySettings()
    address = _address_cluster(labels, points_m, settings)
    butt, tip = (address[labels.index(k)] for k in settings.shaft_axis_labels)
    centroid = address[[labels.index(k) for k in settings.head_labels]].mean(axis=0)
    axis = (tip - butt) / np.linalg.norm(tip - butt)
    if abs(axis[2]) < math.sin(math.radians(settings.min_shaft_elevation_deg)):
        raise ValueError("shaft elevation at address is below the minimum")
    if (centroid - tip) @ axis <= 0:
        raise ValueError("head triad is not beyond the tip-side shaft marker")
    radial = (centroid - butt) - ((centroid - butt) @ axis) * axis
    radius = float(np.linalg.norm(radial))
    if radius < settings.min_roll_radius_m:
        raise ValueError("head triad centroid is on the shaft axis: roll undefined")
    azimuth = math.radians(settings.head_centroid_azimuth_deg)
    radial = radial / radius
    y_axis = math.cos(azimuth) * radial - math.sin(azimuth) * np.cross(axis, radial)
    rotation = np.column_stack([axis, y_axis, np.cross(axis, y_axis)])
    centre_height = (
        settings.ground_height_m + settings.hover_m + head_support_m(rotation, settings)
    )
    origin = butt + ((centre_height - butt[2]) / axis[2]) * axis
    local = (address - origin) @ rotation
    return {
        k: (float(v[0]), float(v[1]), float(v[2]))
        for k, v in zip(labels, local, strict=True)
    }


def axis_marker_distances_m(
    offsets: Mapping[str, Sequence[float]], settings: ClubOnlySettings
) -> dict[str, float]:
    """Distance of each shaft axis marker from the ``Clubhead`` x axis (m)."""
    return {k: math.hypot(*offsets[k][1:3]) for k in settings.shaft_axis_labels}


def club_only_markers_from_capture(
    capture_path: Path, *, club: str, settings: ClubOnlySettings | None = None
) -> ClubMarkers:
    """Club-only registration of the settings' labels in a canonical capture.

    Missing samples become NaN and the capture is turned z-up. The source
    block records the method, the capture's name and SHA-256 and the settings.
    """
    settings = settings or ClubOnlySettings()
    capture = load_tour_capture(Path(capture_path))
    labels = sorted(
        {*settings.head_labels, *settings.other_labels, *settings.shaft_axis_labels}
    )
    columns = [capture.labels.index(k) for k in labels]
    points = np.where(
        capture.valid[:, columns, None], capture.points_m[:, columns], np.nan
    )
    offsets = derive_club_only_offsets(
        labels, capture_to_native_world(points), settings
    )
    digest = str(capture.source_sha256)
    source = {
        "method": CLUB_ONLY_METHOD,
        "capture": capture_kind(digest),
        "capture_sha256": digest,
        "settings": json.loads(json.dumps(asdict(settings))),  # as a file holds it
    }
    return ClubMarkers(club, offsets, source)
