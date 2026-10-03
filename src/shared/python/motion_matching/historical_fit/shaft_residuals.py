"""Optional image bearings composed by the canonical native spline estimator.

Typed context is a caller declaration, not authentication. Persistence and jobs
must rebind evidence through the capture library before admitting these terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import re
import numpy as np
from .shaft_geometry import AuthoredShaftAxis
from .shaft_observations import ShaftAxisEvidence

if TYPE_CHECKING:
    from .contracts import CameraProjection
    from src.shared.python.motion_matching.pipeline.plant import MatchingPlant


def _digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Image source identity requires prefixed lowercase SHA256")


@dataclass(frozen=True)
class ImageSourceIdentity:
    """Declared capture/source/camera/clock context; never a binding capability."""

    capture_id: str
    capture_sha256: str
    source_sha256: str
    camera_id: str
    source_clock_sha256: str

    def __post_init__(self) -> None:
        for name in ("capture_id", "camera_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError("Image source identity requires nonempty trimmed IDs")
        for name in ("capture_sha256", "source_sha256", "source_clock_sha256"):
            _digest(getattr(self, name))


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True, kw_only=True)
class ImageResidualAssessment:
    """Validated raw image diagnostic identity; no objective or admission claim."""

    evidence_sha256: str
    frame_indices: tuple[int, ...]
    source_times: tuple[float, ...]
    raw_rms_pixels: float | None

    def __post_init__(self) -> None:
        _digest(self.evidence_sha256)
        indices, times = tuple(self.frame_indices), tuple(self.source_times)
        if not indices or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices
        ):
            raise ValueError("Assessment frame indices must be nonnegative integers")
        if len(indices) != len(times) or any(not _finite(t) for t in times):
            raise ValueError("Assessment times must match finite indexed frames")
        if any(b <= a for a, b in zip(indices, indices[1:], strict=False)) or any(
            b <= a for a, b in zip(times, times[1:], strict=False)
        ):
            raise ValueError("Assessment indices and source times must increase")
        if self.raw_rms_pixels is not None and (
            not _finite(self.raw_rms_pixels) or self.raw_rms_pixels < 0
        ):
            raise ValueError("Raw image RMS must be finite nonnegative or absent")
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "source_times", times)


@dataclass(frozen=True, kw_only=True)
class ShaftAxisAssessment(ImageResidualAssessment):
    """Unweighted perpendicular pixel RMS, separate from Euclidean body RMS.

    Two distances per observed fragment, regardless of authored uncertainty or
    confidence. Abstentions retain two objective zero slots but no raw readings.
    The unoriented angular error is in degrees and does not measure shaft length.
    """

    perpendicular_errors_pixels: tuple[tuple[float, float] | None, ...]
    angular_errors_deg: tuple[float | None, ...]
    observed_segment_count: int

    def __post_init__(self) -> None:
        super().__post_init__()
        errors = tuple(
            None if row is None else tuple(row)
            for row in self.perpendicular_errors_pixels
        )
        angles = tuple(self.angular_errors_deg)
        if len(errors) != len(self.frame_indices) or len(angles) != len(errors):
            raise ValueError("Shaft diagnostic rows must align with source frames")
        for row, angle in zip(errors, angles, strict=True):
            if row is None:
                if angle is not None:
                    raise ValueError("Abstention must have no raw bearing angle")
            elif (
                len(row) != 2
                or any(not _finite(value) for value in row)
                or angle is None
                or not _finite(angle)
                or not 0 <= angle <= 90
            ):
                raise ValueError(
                    "Shaft diagnostics require finite pixel pairs and angles in [0,90]"
                )
        observed = [row for row in errors if row is not None]
        if (
            isinstance(self.observed_segment_count, bool)
            or not isinstance(self.observed_segment_count, int)
            or self.observed_segment_count != len(observed)
        ):
            raise ValueError("Observed shaft segment count differs from raw rows")
        expected = float(np.sqrt(np.mean(np.square(observed)))) if observed else None
        if (expected is None) != (self.raw_rms_pixels is None) or (
            expected is not None
            and not np.isclose(expected, self.raw_rms_pixels, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Raw shaft RMS differs from unweighted pixel diagnostics")
        object.__setattr__(self, "perpendicular_errors_pixels", errors)
        object.__setattr__(self, "angular_errors_deg", angles)


@runtime_checkable
class ImageResidualTerm(Protocol):
    """Pose-space optional image residual; solver owns basis/Jacobian composition."""

    @property
    def source_times(self) -> tuple[float, ...]: ...
    def validate(
        self, native: MatchingPlant, identity: ImageSourceIdentity
    ) -> None: ...
    def residual(
        self, native: MatchingPlant, camera: CameraProjection, poses: np.ndarray
    ) -> np.ndarray: ...
    def assess(
        self, native: MatchingPlant, camera: CameraProjection, poses: np.ndarray
    ) -> ImageResidualAssessment: ...


@dataclass(frozen=True)
class AdditionalImageResiduals:
    """Immutable opt-in terms with explicit unauthenticated source context."""

    source_identity: ImageSourceIdentity
    terms: tuple[ImageResidualTerm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, ImageSourceIdentity):
            raise ValueError("Typed image source identity required")
        terms = tuple(self.terms)
        if not terms or any(not isinstance(term, ImageResidualTerm) for term in terms):
            raise ValueError("Nonempty typed image residual terms required")
        object.__setattr__(self, "terms", terms)

    def validate(self, native: MatchingPlant) -> None:
        for term in self.terms:
            term.validate(native, self.source_identity)


@dataclass(frozen=True)
class ShaftAxisResidualTerm:
    """Visible image fragments constrain an authored infinite native shaft line."""

    evidence: ShaftAxisEvidence
    axis: AuthoredShaftAxis
    source_clock_sha256: str
    unknown_visibility_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ShaftAxisEvidence) or not isinstance(
            self.axis, AuthoredShaftAxis
        ):
            raise ValueError("Typed shaft evidence and authored axis required")
        _digest(self.source_clock_sha256)
        weight = self.unknown_visibility_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise ValueError(
                "Unknown visibility weight must be explicitly finite in [0,1]"
            )

    @property
    def source_times(self) -> tuple[float, ...]:
        return tuple(
            float(frame.frame.presentation_time) for frame in self.evidence.frames
        )

    def validate(self, native: MatchingPlant, identity: ImageSourceIdentity) -> None:
        if self.axis.native_model_sha != native.plant_sha:
            raise ValueError("Shaft axis native model identity differs")
        actual = ImageSourceIdentity(
            self.evidence.capture_id,
            self.evidence.capture_sha256,
            self.evidence.source_sha256,
            self.evidence.frames[0].frame.camera_id,
            self.source_clock_sha256,
        )
        if actual != identity:
            raise ValueError(
                "Shaft image source identity/camera/clock differs from context"
            )

    def _raw(
        self,
        native: MatchingPlant,
        camera: CameraProjection,
        pose: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, float] | None:
        segment = self.evidence.frames[frame_index].segment
        if segment.points_px is None:
            return None
        origin, normal, direction, length = project_authored_shaft_line(
            native, camera, self.axis, pose
        )
        observed = np.asarray(segment.points_px)
        errors = (observed - origin) @ normal
        bearing = observed[1] - observed[0]
        cosine = abs(float(bearing @ direction)) / (np.linalg.norm(bearing) * length)
        angle = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
        return errors, angle

    def _poses(self, poses: np.ndarray, native: MatchingPlant) -> None:
        if self.axis.native_model_sha != native.plant_sha:
            raise ValueError("Shaft axis native model identity differs")
        if (
            poses.shape != (len(self.evidence.frames), len(native.coordinate_order))
            or not np.isfinite(poses).all()
        ):
            raise ValueError(
                "Shaft poses must match finite source rows and native order"
            )

    def residual(
        self, native: MatchingPlant, camera: CameraProjection, poses: np.ndarray
    ) -> np.ndarray:
        self._poses(poses, native)
        rows = []
        for index, pose in enumerate(poses):
            raw = self._raw(native, camera, pose, index)
            if raw is None:
                rows.append(np.zeros(2))
                continue
            segment = self.evidence.frames[index].segment
            visibility = (
                self.unknown_visibility_weight
                if segment.visibility is None
                else segment.visibility
            )
            assert segment.confidence is not None and segment.sigma_px is not None
            rows.append(
                raw[0] * np.sqrt(segment.confidence * visibility) / segment.sigma_px
            )
        return np.asarray(rows).ravel()

    def assess(
        self, native: MatchingPlant, camera: CameraProjection, poses: np.ndarray
    ) -> ShaftAxisAssessment:
        self._poses(poses, native)
        raw = [
            self._raw(native, camera, pose, index) for index, pose in enumerate(poses)
        ]
        distances = tuple(
            None if item is None else (float(item[0][0]), float(item[0][1]))
            for item in raw
        )
        observed = [item[0] for item in raw if item is not None]
        rms = float(np.sqrt(np.mean(np.square(observed)))) if observed else None
        return ShaftAxisAssessment(
            evidence_sha256=self.evidence.sha256,
            frame_indices=tuple(frame.frame_index for frame in self.evidence.frames),
            source_times=self.source_times,
            perpendicular_errors_pixels=distances,
            angular_errors_deg=tuple(None if item is None else item[1] for item in raw),
            observed_segment_count=len(observed),
            raw_rms_pixels=rms,
        )


def project_authored_shaft_line(
    native: MatchingPlant,
    camera: CameraProjection,
    axis: AuthoredShaftAxis,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return image origin, unit normal, direction and length of one rigid axis.

    Uses the canonical native markers and saved projection. This authenticates
    neither caller context nor historical shaft/camera geometry. Degenerate
    projections reject instead of fabricating a bearing.
    """
    if axis.native_model_sha != native.plant_sha:
        raise ValueError("Shaft axis native model identity differs")
    points = native.marker_positions(
        pose,
        {
            "shaft_a": (axis.body, axis.point_a_m),
            "shaft_b": (axis.body, axis.point_b_m),
        },
    )
    projected = camera.project(points)
    direction = projected[1] - projected[0]
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 1e-10:
        raise ValueError("Projected shaft axis is degenerate")
    normal = np.array([-direction[1], direction[0]]) / length
    return projected[0], normal, direction, float(length)
