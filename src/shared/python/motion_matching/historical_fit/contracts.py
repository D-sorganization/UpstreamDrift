"""Validated image evidence and explicitly unqualified native fit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.shared.python.estimation import (
    AuthoredHermiteInitialization,
    CubicHermiteSplineTrajectory,
    project_pinhole,
    reprojection_residual_from_points,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
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
    initial_samples: np.ndarray | None = None

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
        if self.initial_samples is not None:
            samples = _array(self.initial_samples, "Warm-start samples")
            if samples.shape != (len(times), len(self.seed)):
                raise ValueError(
                    "Warm-start samples must match source frames and coordinates"
                )
            object.__setattr__(self, "initial_samples", samples)


def _validate_coordinate_bounds(bounds: tuple[tuple[str, float, float], ...]) -> None:
    """Require unique named two-sided finite limits in native coordinate units."""
    if not isinstance(bounds, tuple):
        raise ValueError("Coordinate bounds must be immutable named tuples")
    names: set[str] = set()
    for bound in bounds:
        if not isinstance(bound, tuple) or len(bound) != 3:
            raise ValueError("Coordinate bounds require name, lower and upper")
        name, lower, upper = bound
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Coordinate bounds require unique nonempty names")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                for value in (lower, upper)
            )
            or lower > upper
        ):
            raise ValueError("Coordinate bounds require ordered finite endpoints")
        names.add(name)


@dataclass(frozen=True)
class ImageFitConfig:
    max_iterations: int = 100
    prior_weight: float = 0.1
    smoothness_weight: float = 0.01
    closure_weight: float = 100.0
    constraint_options: ConstraintOptions | None = None
    interior_fractions: tuple[float, ...] = ()
    coordinate_bounds: tuple[tuple[str, float, float], ...] = ()

    initialization_policy: Literal["strict", "authored_range_project_zero_slopes"] = (
        "strict"
    )

    def __post_init__(self) -> None:
        if self.initialization_policy not in (
            "strict",
            "authored_range_project_zero_slopes",
        ):
            raise ValueError("Unknown initialization policy")
        if self.initialization_policy != "strict" and not self.coordinate_bounds:
            raise ValueError(
                "Authored initialization requires explicit coordinate bounds"
            )
        _validate_coordinate_bounds(self.coordinate_bounds)
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("Fit iteration budget must be a positive integer")
        for value in (self.prior_weight, self.smoothness_weight, self.closure_weight):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Fit weights must be finite and nonnegative")
        fractions = self.interior_fractions
        if (
            not isinstance(fractions, tuple)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 < value < 1
                for value in fractions
            )
            or tuple(sorted(set(fractions))) != fractions
        ):
            raise ValueError(
                "Interior fractions must be finite, unique and increasing in (0, 1)"
            )
        if fractions and self.constraint_options is None:
            raise ValueError("Interior fractions require explicit constraint options")
        if self.constraint_options is not None:
            from src.shared.python.motion_matching.constraint_kinematics import (
                ConstraintOptions,
            )

            if not isinstance(self.constraint_options, ConstraintOptions):
                raise ValueError(
                    "Fit constraint options must be a ConstraintOptions record"
                )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ImageFitConfig:
        """Decode only the declared JSON configuration; nested records stay typed."""
        if not isinstance(record, Mapping):
            raise ValueError("Fit configuration must be an object")
        values = dict(record)
        bounds = values.get("coordinate_bounds", ())
        if not isinstance(bounds, (tuple, list)) or any(
            not isinstance(bound, (tuple, list)) for bound in bounds
        ):
            raise ValueError("Coordinate bounds must be an array of named limits")
        values["coordinate_bounds"] = tuple(tuple(bound) for bound in bounds)
        fractions = values.get("interior_fractions", ())
        if not isinstance(fractions, (tuple, list)):
            raise ValueError("Interior fractions must be an array")
        values["interior_fractions"] = tuple(fractions)
        nested = values.get("constraint_options")
        try:
            if nested is not None:
                from src.shared.python.motion_matching.constraint_kinematics import (
                    ConstraintOptions,
                )
                from src.shared.python.motion_matching.contact_law import GroundPlane

                if not isinstance(nested, Mapping) or not isinstance(
                    nested.get("ground"), Mapping
                ):
                    raise ValueError(
                        "Fit constraint configuration must contain a ground object"
                    )
                options = dict(nested)
                options["ground"] = GroundPlane(**dict(options["ground"]))
                pinned = options.get("pinned_spheres", ())
                if not isinstance(pinned, (tuple, list)):
                    raise ValueError("Pinned sphere names must be an array")
                options["pinned_spheres"] = tuple(pinned)
                values["constraint_options"] = ConstraintOptions(**options)
            return cls(**values)
        except (TypeError, KeyError) as exc:
            raise ValueError("Malformed fit configuration") from exc


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
    constraint_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    constraint_residuals: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    constraint_row_labels: tuple[str, ...] = ()
    optimizer_ran: bool = True
    initialization: AuthoredHermiteInitialization | None = None
    qualification: str = field(default="monocular_research_hypothesis", init=False)
    physical_time_qualified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_times",
            "q",
            "knot_times",
            "spline_coefficients",
            "constraint_times",
            "constraint_residuals",
        ):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        if not isinstance(self.optimizer_ran, bool) or (
            self.converged and not self.optimizer_ran
        ):
            raise ValueError("Convergence requires an executed optimizer")
        if self.initialization is not None and not isinstance(
            self.initialization, AuthoredHermiteInitialization
        ):
            raise ValueError("Initialization must be a typed authored receipt")
        errors = self.pixel_errors.copy()
        errors.setflags(write=False)
        object.__setattr__(self, "pixel_errors", errors)
        times, labels = self.constraint_times, self.constraint_row_labels
        if (
            times.ndim != 1
            or not np.all(np.diff(times) > 0)
            or np.any(times < self.source_times[0])
            or np.any(times > self.source_times[-1])
        ):
            raise ValueError(
                "Constraint report times must increase inside the source interval"
            )
        if (
            not isinstance(labels, tuple)
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
            or bool(len(times)) != bool(labels)
        ):
            raise ValueError(
                "Constraint report labels must be immutable, unique nonempty strings"
            )
        if self.constraint_residuals.shape != (
            len(self.constraint_times),
            len(self.constraint_row_labels),
        ):
            raise ValueError("Constraint report times, labels and residuals must agree")

    @property
    def maximum_constraint_residual(self) -> float:
        """Maximum scaled dimensionless residual at tested times, never all-time proof."""
        return float(np.max(np.abs(self.constraint_residuals), initial=0.0))

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
