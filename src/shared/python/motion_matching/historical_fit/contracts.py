"""Validated image evidence and explicitly unqualified native fit records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    project_pinhole,
    reprojection_residual_from_points,
)


def _array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CameraProjection:
    """A fixed, explicitly supplied pinhole camera; its calibration is not inferred."""

    intrinsics: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        for name in ("intrinsics", "rotation", "translation"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        if (
            self.intrinsics.shape != (3, 3)
            or self.rotation.shape != (3, 3)
            or self.translation.shape != (3,)
        ):
            raise ValueError(
                "Camera requires 3x3 intrinsics/rotation and 3-vector translation"
            )
        if (
            self.intrinsics[0, 0] <= 0
            or self.intrinsics[1, 1] <= 0
            or not np.allclose(self.intrinsics[2], [0, 0, 1])
        ):
            raise ValueError(
                "Camera requires positive focal lengths and a homogeneous bottom row"
            )
        if not np.allclose(
            self.rotation @ self.rotation.T, np.eye(3), atol=1e-8
        ) or not np.isclose(np.linalg.det(self.rotation), 1.0):
            raise ValueError("Camera rotation must be proper orthonormal")

    def _depth(self, points: np.ndarray) -> None:
        if not np.all((points @ self.rotation.T + self.translation)[:, 2] > 1e-8):
            raise ValueError("Model markers must be in front of the supplied camera")

    def project(self, points: np.ndarray) -> np.ndarray:
        self._depth(points)
        return project_pinhole(
            points,
            self.intrinsics,
            rotation_world_to_camera=self.rotation,
            translation_world_to_camera=self.translation,
        )

    def residual(
        self, points: np.ndarray, observed: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        self._depth(points)
        return reprojection_residual_from_points(
            points,
            observed,
            self.intrinsics,
            confidence,
            rotation_world_to_camera=self.rotation,
            translation_world_to_camera=self.translation,
        )


@dataclass(frozen=True)
class ImageFitInputs:
    """Pixels and confidence in attachment order, with source PTS and native pose scales."""

    source_times: np.ndarray
    observed_pixels: np.ndarray
    confidence: np.ndarray
    seed: np.ndarray
    coordinate_scales: np.ndarray
    free_coordinates: tuple[str, ...]
    knot_times: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_times",
            "observed_pixels",
            "confidence",
            "seed",
            "coordinate_scales",
        ):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        times, observed, confidence = (
            self.source_times,
            self.observed_pixels,
            self.confidence,
        )
        if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError(
                "Source times must contain at least two strictly increasing PTS values"
            )
        if (
            observed.ndim != 3
            or observed.shape[0] != len(times)
            or observed.shape[2] != 2
            or confidence.shape != observed.shape[:2]
        ):
            raise ValueError(
                "Image evidence must have shape (frames, markers, 2) with matching confidence"
            )
        if np.any((confidence < 0) | (confidence > 1)) or not np.any(confidence > 0):
            raise ValueError(
                "At least one observed landmark with confidence in [0, 1] is required"
            )
        if (
            self.seed.ndim != 1
            or self.coordinate_scales.shape != self.seed.shape
            or not np.all(self.coordinate_scales > 0)
        ):
            raise ValueError(
                "Seed must be a vector with matching positive native coordinate scales"
            )
        if not self.free_coordinates or len(set(self.free_coordinates)) != len(
            self.free_coordinates
        ):
            raise ValueError("Free coordinate names must be nonempty and unique")
        knots = _array(
            times if self.knot_times is None else self.knot_times, "knot_times"
        )
        if (
            knots.ndim != 1
            or len(knots) < 2
            or not np.all(np.diff(knots) > 0)
            or knots[0] != times[0]
            or knots[-1] != times[-1]
        ):
            raise ValueError(
                "Spline knots must increase and span the exact source interval"
            )
        object.__setattr__(self, "knot_times", knots)


@dataclass(frozen=True)
class ImageFitConfig:
    max_iterations: int = 100
    prior_weight: float = 0.1
    smoothness_weight: float = 0.01
    closure_weight: float = 100.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("Fit iteration budget must be a positive integer")
        for value in (self.prior_weight, self.smoothness_weight, self.closure_weight):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Fit weights must be finite and nonnegative")


@dataclass(frozen=True)
class ImageFitResult:
    """Kinematic research output; source-clock derivatives cannot certify torques."""

    source_times: np.ndarray
    q: np.ndarray
    rms_pixels: float
    initial_rms_pixels: float
    pixel_errors: np.ndarray
    observed_point_count: int
    model_sha: str
    coordinate_order: tuple[str, ...]
    converged: bool
    optimizer_message: str
    knot_times: np.ndarray
    spline_coefficients: np.ndarray
    free_coordinates: tuple[str, ...]
    qualification: str = field(default="monocular_research_hypothesis", init=False)
    physical_time_qualified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name in ("source_times", "q", "knot_times", "spline_coefficients"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        errors = self.pixel_errors.copy()
        errors.setflags(write=False)
        object.__setattr__(self, "pixel_errors", errors)

    def evaluate_source_times(self, source_times: np.ndarray) -> np.ndarray:
        """Evaluate the preserved fitted spline inside its source-clock interval."""
        times = np.asarray(source_times, dtype=float)
        if (
            times.ndim != 1
            or not np.isfinite(times).all()
            or np.any(times < self.knot_times[0])
            or np.any(times > self.knot_times[-1])
        ):
            raise ValueError("Evaluation must stay inside the fitted source interval")
        trajectory = CubicHermiteSplineTrajectory(
            self.knot_times, len(self.free_coordinates)
        )
        free = trajectory.evaluate(self.spline_coefficients, times).q
        poses = np.tile(self.q[0], (len(times), 1))
        indices = [self.coordinate_order.index(name) for name in self.free_coordinates]
        poses[:, indices] = free
        return poses
