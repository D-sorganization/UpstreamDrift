"""Pure, unauthenticated raw shaft diagnostics; no solver, SDK or FK implementation."""

from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
from typing import Any
import numpy as np
from .contracts import CameraProjection
from .shaft_observations import (
    ShaftAxisSegment,
)
from src.shared.python.motion_matching.marker_kinematics import (
    MarkerLinearization,
    MarkerLinearizer,
)


def _number(value: Any) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError("Finite real numbers excluding bool required")
    return float(value)


def _names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("Nonempty ordered names required")
    result = tuple(value)
    if any(not isinstance(x, str) or not x or x.strip() != x for x in result) or len(
        set(result)
    ) != len(result):
        raise ValueError("Distinct trimmed coordinate/marker names required")
    return result


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Finite numeric pair required")
    return (_number(value[0]), _number(value[1]))


@dataclass(frozen=True)
class FragmentDerivativeOptions:
    """Copied native order/units and finite authored limits; None is unbounded.

    Postcondition: unique radian selections, positive steps and one immutable
    two-sided bound or explicit None per native coordinate. Authored scalar
    limits are neither clinical nor coupled ROM.
    """

    coordinate_order: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    selected_coordinates: tuple[str, ...]
    steps: tuple[float, ...] = (1e-5, 1e-6, 1e-7)
    bounds: tuple[tuple[float, float] | None, ...] = ()
    marker_labels: tuple[str, str] = ("shaft_a", "shaft_b")

    def __post_init__(self) -> None:
        order = _names(self.coordinate_order)
        selected = _names(self.selected_coordinates)
        units = tuple(self.coordinate_units)
        labels = _names(self.marker_labels)
        if len(units) != len(order) or any(unit not in ("m", "rad") for unit in units):
            raise ValueError("Explicit native m/rad units required")
        if any(
            name not in order or units[order.index(name)] != "rad" for name in selected
        ):
            raise ValueError("Selected diagnostic coordinates must be known radians")
        if not isinstance(self.steps, (tuple, list)) or not self.steps:
            raise ValueError("Nonempty declared finite difference steps required")
        steps = tuple(_number(x) for x in self.steps)
        if any(x <= 0 for x in steps) or len(set(steps)) != len(steps):
            raise ValueError("Unique positive steps required")
        if not isinstance(self.bounds, (tuple, list)) or len(self.bounds) != len(order):
            raise ValueError("Authored bounds must match native coordinate order")
        bounds = tuple(None if x is None else _pair(x) for x in self.bounds)
        if any(x is not None and x[0] > x[1] for x in bounds) or len(labels) != 2:
            raise ValueError("Two distinct shaft markers and ordered limits required")
        for key, value in (
            ("coordinate_order", order),
            ("coordinate_units", units),
            ("selected_coordinates", selected),
            ("steps", steps),
            ("bounds", bounds),
            ("marker_labels", labels),
        ):
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class FragmentDerivativeCheck:
    """Copied raw px/rad derivative vectors with truthful central-FD status.

    Postcondition: available finite pairs have consistent error; bound-limited
    central values/error remain None. No calibration is implied.
    """

    coordinate: str
    step: float
    units: str
    analytic: tuple[float, float] | None
    central: tuple[float, float] | None
    max_abs_error: float | None
    status: str

    def __post_init__(self) -> None:
        _names((self.coordinate,))
        step = _number(self.step)
        if step <= 0 or self.units != "px_per_rad":
            raise ValueError("Positive radian step and raw px_per_rad units required")
        analytic = None if self.analytic is None else _pair(self.analytic)
        central = None if self.central is None else _pair(self.central)
        if self.status == "available":
            if analytic is None or central is None or self.max_abs_error is None:
                raise ValueError("Available derivative requires both numeric pairs")
            error = _number(self.max_abs_error)
            expected = max(abs(a - b) for a, b in zip(analytic, central, strict=True))
            if error < 0 or not np.isclose(error, expected, rtol=0, atol=1e-12):
                raise ValueError("Derivative error differs from declared vectors")
        elif self.status == "central_step_outside_authored_bound":
            if (
                analytic is None
                or central is not None
                or self.max_abs_error is not None
            ):
                raise ValueError("Bound-limited difference must remain explicitly null")
        else:
            raise ValueError("Unknown derivative availability status")
        for key, value in (
            ("step", step),
            ("analytic", analytic),
            ("central", central),
        ):
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class FragmentDerivativeAssessment:
    """Immutable local check, not source admission or calibration.

    Postcondition: exact rectangular named rows when available; unavailable
    raw assessments contain no fabricated numeric rows.
    """

    options: FragmentDerivativeOptions
    status: str
    raw_distances_px: tuple[float, float] | None
    checks: tuple[FragmentDerivativeCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.options, FragmentDerivativeOptions) or not isinstance(
            self.checks, (tuple, list)
        ):
            raise ValueError("Typed options/checks required")
        checks = tuple(self.checks)
        if any(not isinstance(row, FragmentDerivativeCheck) for row in checks):
            raise ValueError("Typed derivative rows required")
        distances = (
            None if self.raw_distances_px is None else _pair(self.raw_distances_px)
        )
        if self.status == "available":
            expected = [
                (name, h)
                for name in self.options.selected_coordinates
                for h in self.options.steps
            ]
            if (
                distances is None
                or [(row.coordinate, row.step) for row in checks] != expected
            ):
                raise ValueError(
                    "Available assessment requires exact rectangular named rows"
                )
        elif self.status in (
            "explicit_abstention",
            "degenerate_projection",
            "marker_capability_unavailable",
        ):
            if distances is not None or checks:
                raise ValueError(
                    "Unavailable assessment must not fabricate numerical rows"
                )
        else:
            raise ValueError("Unknown assessment availability status")
        object.__setattr__(self, "raw_distances_px", distances)
        object.__setattr__(self, "checks", checks)


def _pose(q: Any, options: FragmentDerivativeOptions) -> np.ndarray:
    values = np.asarray(q, dtype=object)
    if values.shape != (len(options.coordinate_order),):
        raise ValueError("Pose shape differs from exact native order")
    result = np.array([_number(x) for x in values], dtype=float)
    if any(
        bound is not None and not bound[0] <= result[i] <= bound[1]
        for i, bound in enumerate(options.bounds)
    ):
        raise ValueError("Base pose violates declared authored bounds")
    return result


def _row(
    provider: MarkerLinearizer, q: np.ndarray, options: FragmentDerivativeOptions
) -> MarkerLinearization:
    row = provider.marker_linearization(q.copy())
    if (
        not isinstance(row, MarkerLinearization)
        or row.coordinate_order != options.coordinate_order
        or row.marker_labels != options.marker_labels
    ):
        raise ValueError("Public marker identities/order differ from options")
    return row


def _line(
    camera: CameraProjection, row: MarkerLinearization, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reviewed normal derivative; existing public line projection has no Jacobian."""
    pixels = camera.project(row.positions)
    delta = pixels[1] - pixels[0]
    length = np.linalg.norm(delta)
    if length <= 1e-8:
        return None
    jac = np.einsum(
        "pij,pjk->pik", camera.project_jacobian(row.positions), row.jacobian
    )
    direction = delta / length
    d_direction = (
        (np.eye(2) - np.outer(direction, direction)) @ (jac[1] - jac[0]) / length
    )
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    normal = turn @ direction
    d_normal = turn @ d_direction
    return (observed - pixels[0]) @ normal, (
        observed - pixels[0]
    ) @ d_normal - normal @ jac[0]


def _check(
    provider: MarkerLinearizer,
    camera: CameraProjection,
    q: np.ndarray,
    observed: np.ndarray,
    options: FragmentDerivativeOptions,
    coordinate: str,
    step: float,
    analytic: np.ndarray,
) -> FragmentDerivativeCheck:
    column = options.coordinate_order.index(coordinate)
    bound = options.bounds[column]
    args = (coordinate, step, "px_per_rad", tuple(analytic))
    if bound is not None and (
        q[column] - step < bound[0] or q[column] + step > bound[1]
    ):
        return FragmentDerivativeCheck(
            *args, None, None, "central_step_outside_authored_bound"
        )
    plus = q.copy()
    minus = q.copy()
    plus[column] += step
    minus[column] -= step
    high = _line(camera, _row(provider, plus, options), observed)
    low = _line(camera, _row(provider, minus, options), observed)
    if high is None or low is None:
        raise ValueError(
            "Perturbed projected shaft degeneracy prevents central comparison"
        )
    central = (high[0] - low[0]) / (2 * step)
    return FragmentDerivativeCheck(
        *args, tuple(central), float(np.max(np.abs(central - analytic))), "available"
    )


def assess_fragment_derivatives(
    provider: MarkerLinearizer,
    camera: CameraProjection,
    q: Any,
    fragment: ShaftAxisSegment,
    options: FragmentDerivativeOptions,
) -> FragmentDerivativeAssessment:
    """Assess one source-pixel fragment through public marker/camera providers.

    The caller authenticates PNG/clock/model/axis and camera separately; typed
    inputs are not admission capabilities. Confidence/sigma do not weight this
    raw diagnostic. Postcondition: finite copied tuples or explicit unavailable
    status; no clipping, FK implementation, solver or provider use on abstention.
    """
    if (
        not isinstance(options, FragmentDerivativeOptions)
        or not isinstance(fragment, ShaftAxisSegment)
        or not isinstance(camera, CameraProjection)
    ):
        raise ValueError("Typed diagnostic options, fragment and camera required")
    pose = _pose(q, options)
    if fragment.points_px is None:
        return FragmentDerivativeAssessment(options, "explicit_abstention", None, ())
    if not isinstance(provider, MarkerLinearizer):
        return FragmentDerivativeAssessment(
            options, "marker_capability_unavailable", None, ()
        )
    try:
        row = _row(provider, pose, options)
    except NotImplementedError:
        return FragmentDerivativeAssessment(
            options, "marker_capability_unavailable", None, ()
        )
    observed = np.asarray(fragment.points_px)
    line = _line(camera, row, observed)
    if line is None:
        return FragmentDerivativeAssessment(options, "degenerate_projection", None, ())
    distances, jacobian = line
    checks = tuple(
        _check(
            provider,
            camera,
            pose,
            observed,
            options,
            name,
            h,
            jacobian[:, options.coordinate_order.index(name)],
        )
        for name in options.selected_coordinates
        for h in options.steps
    )
    return FragmentDerivativeAssessment(options, "available", tuple(distances), checks)


def optional_fragment_derivatives(
    provider: MarkerLinearizer,
    camera: CameraProjection,
    q: Any,
    fragment: ShaftAxisSegment,
    options: FragmentDerivativeOptions,
    enabled: bool = False,
) -> FragmentDerivativeAssessment | None:
    """Return None without input/provider access when explicitly disabled.

    Enabled mode delegates to the same raw check and never changes objective
    rows, body denominators or source-evidence roles.
    """
    if type(enabled) is not bool:
        raise ValueError("Explicit boolean diagnostic opt-in required")
    return (
        assess_fragment_derivatives(provider, camera, q, fragment, options)
        if enabled
        else None
    )
