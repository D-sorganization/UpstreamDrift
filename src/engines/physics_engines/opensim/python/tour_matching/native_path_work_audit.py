"""Source-bound, sampled workless-identity audits for native path geometry.

This diagnostic records a finite set of observations. It never grants replay or
physiological admission and makes no global error-bound claim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

from .muscle_qualification import compute_path_length_finite_difference_moment_arm
from .native_prepared_state import DeclaredColdStart


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _interval(value: tuple[float, float], field: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field} must be a finite increasing interval")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError(f"{field} must be a finite increasing interval")
    return float(lower), float(upper)


@dataclass(frozen=True)
class NativePathWorkIdentity:
    """Exact source/component identity and the intersected coordinate chart."""

    source_sha256: str
    path_component: str
    coordinate_path: str
    point_paths: tuple[str, ...]
    source_component_paths: tuple[str, ...]
    chart_bounds: tuple[float, float]
    active_bounds: tuple[float, float]
    function_bounds: tuple[float, float]
    domain_provenance: str = "caller-declared"

    @classmethod
    def from_declared_cold_start(
        cls,
        declaration: DeclaredColdStart,
        *,
        path_component: str,
        coordinate_path: str,
        point_paths: tuple[str, ...],
        source_component_paths: tuple[str, ...],
        active_bounds: tuple[float, float],
        function_bounds: tuple[float, float],
    ) -> NativePathWorkIdentity:
        """Reuse the prepared-state declaration's source and coordinate chart."""
        chart_bounds = declaration.chart_bounds.get(coordinate_path)
        if chart_bounds is None:
            raise ValueError("path coordinate is absent from the declared native chart")
        return cls(
            source_sha256=declaration.source_sha256,
            path_component=path_component,
            coordinate_path=coordinate_path,
            point_paths=point_paths,
            source_component_paths=source_component_paths,
            chart_bounds=chart_bounds,
            active_bounds=active_bounds,
            function_bounds=function_bounds,
        )

    def __post_init__(self) -> None:
        _digest(self.source_sha256, "source_sha256")
        if not self.path_component.startswith("/"):
            raise ValueError("path_component must be an absolute native path")
        if not self.coordinate_path.startswith("/"):
            raise ValueError("coordinate_path must be an absolute native path")
        if (
            not self.point_paths
            or len(set(self.point_paths)) != len(self.point_paths)
            or any(not path.startswith("/") for path in self.point_paths)
        ):
            raise ValueError("point_paths must be unique absolute native paths")
        if (
            not self.source_component_paths
            or len(set(self.source_component_paths)) != len(self.source_component_paths)
            or any(not path.startswith("/") for path in self.source_component_paths)
        ):
            raise ValueError(
                "source_component_paths must be unique absolute native paths"
            )
        if not {self.path_component, self.coordinate_path, *self.point_paths}.issubset(
            self.source_component_paths
        ):
            raise ValueError(
                "path, coordinate and point identities must exist in the source component inventory"
            )
        object.__setattr__(
            self, "chart_bounds", _interval(self.chart_bounds, "chart_bounds")
        )
        object.__setattr__(
            self, "active_bounds", _interval(self.active_bounds, "active_bounds")
        )
        object.__setattr__(
            self, "function_bounds", _interval(self.function_bounds, "function_bounds")
        )
        if self.domain_provenance not in ("caller-declared", "native-loaded-source"):
            raise ValueError("domain_provenance is not a supported evidence origin")

    @property
    def surveyed_bounds(self) -> tuple[float, float]:
        """Intersection of declared chart, path activity and function domains."""
        lower = max(
            self.chart_bounds[0], self.active_bounds[0], self.function_bounds[0]
        )
        upper = min(
            self.chart_bounds[1], self.active_bounds[1], self.function_bounds[1]
        )
        if lower >= upper:
            raise ValueError(
                "declared chart, active path and function domains do not overlap"
            )
        return lower, upper


def inspect_native_path_work_identity(
    model: Any,
    declaration: DeclaredColdStart,
    *,
    path_component: str,
    moving_point_path: str,
    coordinate_path: str,
    osim: Any,
) -> NativePathWorkIdentity:
    """Derive coordinate, ConditionalPathPoint and function bounds natively."""
    source_bytes = declaration.model_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != declaration.source_sha256:
        raise ValueError(
            "path-work source bytes differ from the cold-start source identity"
        )
    components = tuple(model.getComponentsList())
    by_path = {component.getAbsolutePathString(): component for component in components}
    try:
        geometry = by_path[path_component]
        point = by_path[moving_point_path]
        coordinate = by_path[coordinate_path]
    except KeyError as error:
        raise ValueError(
            "path, point or coordinate is absent from the loaded source"
        ) from error
    native_point = osim.MovingPathPoint.safeDownCast(point)
    native_coordinate = osim.Coordinate.safeDownCast(coordinate)
    if native_point is None or native_coordinate is None:
        raise ValueError(
            "loaded source component has an unexpected native concrete class"
        )
    if not moving_point_path.startswith(path_component.rstrip("/") + "/"):
        raise ValueError("moving point is not owned by the declared path component")
    point_set = geometry.getPathPointSet()
    ordered_points = tuple(
        point_set.get(index).getAbsolutePathString()
        for index in range(point_set.getSize())
    )
    try:
        point_index = ordered_points.index(moving_point_path)
    except ValueError as error:
        raise ValueError(
            "moving point is absent from the declared path point set"
        ) from error

    native_bounds = (
        float(native_coordinate.getRangeMin()),
        float(native_coordinate.getRangeMax()),
    )
    declared_bounds = declaration.chart_bounds.get(coordinate_path)
    if declared_bounds is None:
        raise ValueError("moving path coordinate is absent from the declared chart")
    chart_bounds = (
        max(native_bounds[0], declared_bounds[0]),
        min(native_bounds[1], declared_bounds[1]),
    )
    if chart_bounds[0] >= chart_bounds[1]:
        raise ValueError("declared chart does not overlap the native coordinate range")

    active_bounds = native_bounds
    if point_index > 0:
        conditional = osim.ConditionalPathPoint.safeDownCast(
            point_set.get(point_index - 1)
        )
        if conditional is not None:
            active_coordinate = conditional.getCoordinate()
            if (
                active_coordinate is not None
                and active_coordinate.getAbsolutePathString() == coordinate_path
            ):
                active_bounds = (
                    float(conditional.get_range(0)),
                    float(conditional.get_range(1)),
                )

    function_domains: list[tuple[float, float]] = []
    for coordinate_for_axis, function in (
        (native_point.getXCoordinate(), native_point.get_x_location()),
        (native_point.getYCoordinate(), native_point.get_y_location()),
        (native_point.getZCoordinate(), native_point.get_z_location()),
    ):
        if (
            coordinate_for_axis is None
            or coordinate_for_axis.getAbsolutePathString() != coordinate_path
        ):
            continue
        spline = osim.SimmSpline.safeDownCast(function)
        if spline is not None:
            count = int(spline.getNumberOfPoints())
            knots = tuple(float(spline.getX(index)) for index in range(count))
            if count < 2 or not all(math.isfinite(value) for value in knots):
                raise ValueError("moving path spline has an invalid knot domain")
            function_domains.append((min(knots), max(knots)))
        elif (
            osim.LinearFunction.safeDownCast(function) is not None
            or osim.Constant.safeDownCast(function) is not None
        ):
            function_domains.append(native_bounds)
        else:
            raise ValueError("moving path function has no reviewed coordinate domain")
    if not function_domains:
        raise ValueError(
            "moving path point has no function bound to the declared coordinate"
        )
    function_bounds = (
        max(domain[0] for domain in function_domains),
        min(domain[1] for domain in function_domains),
    )
    if function_bounds[0] >= function_bounds[1]:
        raise ValueError("moving path functions have no shared coordinate domain")
    return NativePathWorkIdentity(
        source_sha256=declaration.source_sha256,
        path_component=path_component,
        coordinate_path=coordinate_path,
        point_paths=ordered_points,
        source_component_paths=tuple(by_path),
        chart_bounds=chart_bounds,
        active_bounds=active_bounds,
        function_bounds=function_bounds,
        domain_provenance="native-loaded-source",
    )


@dataclass(frozen=True)
class NativePathWorkSample:
    """One achieved native stencil and its independent force/path comparisons."""

    q_rad: float
    step_rad: float
    achieved_span_rad: float
    path_length_moment_arm_m: float
    native_moment_arm_m: float
    force_moment_arm_m: float
    workless_residual_m: float
    force_residual_m: float


@dataclass(frozen=True)
class NativePathWorkAudit:
    """Sampled evidence bound to source and component paths, never a global proof."""

    identity: NativePathWorkIdentity
    tolerance_m: float
    samples: tuple[NativePathWorkSample, ...]
    status: str
    evidence_scope: str = "sampled-domain-only"


def _sample_full_path_derivative(
    sample_path: Callable[[float], tuple[float, float]], q: float, step: float
) -> tuple[float, float]:
    achieved: dict[float, float] = {}

    def length_at(value: float) -> float:
        achieved_q, length = sample_path(value)
        if (
            not math.isfinite(achieved_q)
            or not math.isclose(achieved_q, value, rel_tol=0.0, abs_tol=1e-12)
            or not math.isfinite(length)
        ):
            raise ValueError(
                "native path sample did not achieve a finite requested state"
            )
        achieved[value] = achieved_q
        return length

    arm = compute_path_length_finite_difference_moment_arm(length_at, q, step)
    span = achieved[q + step] - achieved[q - step]
    return arm, span


def audit_native_path_work(
    identity: NativePathWorkIdentity,
    source_bytes: bytes,
    q_values: Sequence[float],
    step_sizes: Sequence[float],
    sample_path: Callable[[float], tuple[float, float]],
    native_moment_arm: Callable[[float], float],
    force_moment_arm: Callable[[float], float],
    *,
    tolerance_m: float = 1e-7,
) -> NativePathWorkAudit:
    """Compare native moment arms to full-path derivatives over explicit samples.

    ``sample_path`` returns the achieved coordinate and path length. The force
    callback must come from an independently observed native force response.
    The caller supplies component/socket provenance from the loaded source.
    """
    if hashlib.sha256(source_bytes).hexdigest() != identity.source_sha256:
        raise ValueError("path-work source bytes differ from the declared identity")
    if (
        isinstance(tolerance_m, bool)
        or not math.isfinite(tolerance_m)
        or tolerance_m <= 0.0
    ):
        raise ValueError("tolerance_m must be finite and positive")
    if not q_values or not step_sizes:
        raise ValueError(
            "path-work audit requires coordinates and finite-difference steps"
        )

    lower, upper = identity.surveyed_bounds
    observations: list[NativePathWorkSample] = []
    for q in q_values:
        if isinstance(q, bool) or not math.isfinite(q) or not lower <= q <= upper:
            raise ValueError("sample coordinate is outside the intersected chart")
        for step in step_sizes:
            if (
                isinstance(step, bool)
                or not math.isfinite(step)
                or step <= 0.0
                or not lower <= q - step < q + step <= upper
            ):
                raise ValueError(
                    "finite-difference stencil exceeds the intersected chart"
                )
            path_arm, span = _sample_full_path_derivative(sample_path, q, step)
            native_arm = native_moment_arm(q)
            force_arm = force_moment_arm(q)
            if not math.isfinite(native_arm) or not math.isfinite(force_arm):
                raise ValueError("native and force-defined moment arms must be finite")
            workless_residual = abs(path_arm - native_arm)
            force_residual = abs(force_arm - native_arm)
            observations.append(
                NativePathWorkSample(
                    q_rad=float(q),
                    step_rad=float(step),
                    achieved_span_rad=span,
                    path_length_moment_arm_m=path_arm,
                    native_moment_arm_m=float(native_arm),
                    force_moment_arm_m=float(force_arm),
                    workless_residual_m=workless_residual,
                    force_residual_m=force_residual,
                )
            )

    passed = all(
        row.workless_residual_m <= tolerance_m and row.force_residual_m <= tolerance_m
        for row in observations
    )
    return NativePathWorkAudit(
        identity=identity,
        tolerance_m=float(tolerance_m),
        samples=tuple(observations),
        status=(
            "workless_identity_satisfied_at_tested_samples"
            if passed
            else "workless_identity_not_satisfied"
        ),
    )
