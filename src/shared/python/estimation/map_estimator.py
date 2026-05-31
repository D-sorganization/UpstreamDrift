"""Single-trial MAP estimator for canonical-core trajectory fitting.

The estimator owns the decision-vector plumbing for CC-19: a cubic-Hermite
spline trajectory and one shared parameter block spanning the whole trial. The
residual math is injected as a callable so CC-18 residual implementations can
plug in through their public surface without this module depending on engine
internals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.contracts import require
from src.shared.python.simulation_backends.provenance import ProvenanceStamp

ParameterKind = Literal["length", "inertia", "generic"]
ResidualFn = Callable[["SplineTrajectoryEvaluation", Mapping[str, float]], np.ndarray]
JacobianFn = Callable[
    ["SplineTrajectoryEvaluation", Mapping[str, float], "MapDecisionLayout"],
    np.ndarray,
]


@dataclass(frozen=True)
class SharedParameterSpec:
    """One scalar parameter shared by every frame in a single-trial fit."""

    name: str
    initial: float
    kind: ParameterKind = "generic"
    lower: float | None = None
    upper: float | None = None
    prior: float | None = None
    prior_scale: float | None = None
    locked: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must be non-empty")
        _require_finite_scalar(self.initial, f"{self.name}.initial")
        if self.lower is not None:
            _require_finite_scalar(self.lower, f"{self.name}.lower")
        if self.upper is not None:
            _require_finite_scalar(self.upper, f"{self.name}.upper")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError(f"{self.name} lower bound must be < upper bound")
        if self.kind == "inertia" and (self.lower is None or self.upper is None):
            raise ValueError("inertia parameters must be bounded around a prior")
        if self.prior_scale is not None and self.prior_scale <= 0.0:
            raise ValueError(f"{self.name}.prior_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe parameter specification payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SharedParameterSpec:
        """Build a parameter specification from a JSON-safe payload."""
        data = dict(payload)
        return cls(
            name=str(data["name"]),
            initial=float(data["initial"]),
            kind=data.get("kind", "generic"),
            lower=_optional_float(data.get("lower")),
            upper=_optional_float(data.get("upper")),
            prior=_optional_float(data.get("prior")),
            prior_scale=_optional_float(data.get("prior_scale")),
            locked=bool(data.get("locked", False)),
        )


@dataclass(frozen=True)
class SharedParameterBlock:
    """Ordered, deterministic scalar parameter block for MAP estimation."""

    specs: tuple[SharedParameterSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("shared parameter names must be unique")

    @classmethod
    def from_specs(cls, specs: Sequence[SharedParameterSpec]) -> SharedParameterBlock:
        """Build a parameter block from ordered specs."""
        return cls(tuple(specs))

    @property
    def size(self) -> int:
        """Number of scalar parameters in the shared block."""
        return len(self.specs)

    @property
    def free_specs(self) -> tuple[SharedParameterSpec, ...]:
        """Unlocked parameters that are part of an estimator decision vector."""
        return tuple(spec for spec in self.specs if not spec.locked)

    @property
    def free_size(self) -> int:
        """Number of unlocked scalar parameters."""
        return len(self.free_specs)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """All parameter names in deterministic block order."""
        return tuple(spec.name for spec in self.specs)

    @property
    def free_parameter_names(self) -> tuple[str, ...]:
        """Unlocked parameter names in deterministic decision-vector order."""
        return tuple(spec.name for spec in self.free_specs)

    def index(self, name: str) -> int:
        """Return the all-parameter block index for ``name``."""
        try:
            return self.parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def free_index(self, name: str) -> int:
        """Return the unlocked-parameter index for ``name``."""
        try:
            return self.free_parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def initial_vector(self) -> np.ndarray:
        """Return initial parameter values in block order."""
        return np.array([spec.initial for spec in self.specs], dtype=float)

    def free_initial_vector(self) -> np.ndarray:
        """Return initial values for unlocked parameters only."""
        return np.array([spec.initial for spec in self.free_specs], dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower/upper bounds in block order."""
        lower = [
            -np.inf if spec.lower is None else float(spec.lower) for spec in self.specs
        ]
        upper = [
            np.inf if spec.upper is None else float(spec.upper) for spec in self.specs
        ]
        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def free_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower/upper bounds for unlocked parameters only."""
        lower = [
            -np.inf if spec.lower is None else float(spec.lower)
            for spec in self.free_specs
        ]
        upper = [
            np.inf if spec.upper is None else float(spec.upper)
            for spec in self.free_specs
        ]
        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def to_mapping(self, vector: np.ndarray) -> dict[str, float]:
        """Map an ordered vector back to named scalar parameters."""
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.size,):
            raise ValueError(f"parameter vector must have shape {(self.size,)}")
        return {
            spec.name: float(values[index]) for index, spec in enumerate(self.specs)
        }

    def expand_free_vector(self, free_vector: np.ndarray) -> np.ndarray:
        """Expand unlocked values into the full block, preserving locked initials."""
        values = np.asarray(free_vector, dtype=float)
        if values.shape != (self.free_size,):
            raise ValueError(
                f"free parameter vector must have shape {(self.free_size,)}"
            )
        expanded = self.initial_vector()
        free_index = 0
        for index, spec in enumerate(self.specs):
            if spec.locked:
                continue
            expanded[index] = values[free_index]
            free_index += 1
        return expanded

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe block payload."""
        return {"parameters": [spec.to_dict() for spec in self.specs]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SharedParameterBlock:
        """Build a parameter block from a JSON-safe payload."""
        specs = payload.get("parameters")
        if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
            raise ValueError("parameter block payload must contain 'parameters'")
        return cls.from_specs([SharedParameterSpec.from_dict(spec) for spec in specs])

    def prior_residuals(self, vector: np.ndarray) -> np.ndarray:
        """Return Gaussian prior residuals for specs with prior and scale."""
        values = np.asarray(vector, dtype=float)
        residuals = []
        for index, spec in enumerate(self.specs):
            if spec.prior is None or spec.prior_scale is None:
                continue
            residuals.append((values[index] - spec.prior) / spec.prior_scale)
        return np.array(residuals, dtype=float)

    def prior_jacobian(self) -> np.ndarray:
        """Return Jacobian of prior residuals with respect to block values."""
        rows = []
        for index, spec in enumerate(self.specs):
            if spec.prior is None or spec.prior_scale is None:
                continue
            row = np.zeros(self.size, dtype=float)
            row[index] = 1.0 / spec.prior_scale
            rows.append(row)
        if not rows:
            return np.zeros((0, self.size), dtype=float)
        return np.vstack(rows)


@dataclass(frozen=True)
class SplineTrajectoryEvaluation:
    """Spline trajectory values and analytic coefficient derivatives."""

    times: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    q_basis: np.ndarray
    v_basis: np.ndarray
    a_basis: np.ndarray


@dataclass(frozen=True)
class CubicHermiteSplineTrajectory:
    """Cubic-Hermite trajectory with analytic first and second derivatives."""

    knot_times: np.ndarray
    n_dof: int

    def __post_init__(self) -> None:
        times = np.asarray(self.knot_times, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("knot_times must be a 1D array with at least two knots")
        if not np.all(np.isfinite(times)):
            raise ValueError("knot_times must be finite")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("knot_times must be strictly increasing")
        if self.n_dof <= 0:
            raise ValueError("n_dof must be positive")
        object.__setattr__(self, "knot_times", times)

    @property
    def n_knots(self) -> int:
        """Number of spline knots."""
        return int(self.knot_times.size)

    @property
    def coefficient_size(self) -> int:
        """Flat coefficient count: knot positions plus knot velocities."""
        return 2 * self.n_knots * self.n_dof

    def initial_coefficients_from_samples(
        self,
        sample_times: np.ndarray,
        sample_q: np.ndarray,
    ) -> np.ndarray:
        """Build deterministic spline coefficients from observed samples."""
        times = np.asarray(sample_times, dtype=float)
        q = np.asarray(sample_q, dtype=float)
        if q.shape != (times.size, self.n_dof):
            raise ValueError(f"sample_q must have shape {(times.size, self.n_dof)}")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(times)):
            raise ValueError("sample times and positions must be finite")
        knot_q = np.column_stack(
            [np.interp(self.knot_times, times, q[:, dof]) for dof in range(self.n_dof)]
        )
        knot_v = np.gradient(knot_q, self.knot_times, axis=0, edge_order=1)
        return self.pack(knot_q, knot_v)

    def pack(self, knot_q: np.ndarray, knot_v: np.ndarray) -> np.ndarray:
        """Flatten knot positions and knot velocities into decision order."""
        q = np.asarray(knot_q, dtype=float)
        v = np.asarray(knot_v, dtype=float)
        expected = (self.n_knots, self.n_dof)
        if q.shape != expected or v.shape != expected:
            raise ValueError(f"knot arrays must have shape {expected}")
        return np.concatenate([q.ravel(), v.ravel()])

    def unpack(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return knot positions and velocities from flat coefficients."""
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.shape != (self.coefficient_size,):
            raise ValueError(f"coefficients must have shape {(self.coefficient_size,)}")
        split = self.n_knots * self.n_dof
        shape = (self.n_knots, self.n_dof)
        return coeffs[:split].reshape(shape), coeffs[split:].reshape(shape)

    def evaluate(
        self,
        coefficients: np.ndarray,
        times: np.ndarray,
    ) -> SplineTrajectoryEvaluation:
        """Evaluate q, v, a and analytic coefficient bases at sample times."""
        coeffs = np.asarray(coefficients, dtype=float)
        sample_times = np.asarray(times, dtype=float)
        if sample_times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if not np.all(np.isfinite(sample_times)):
            raise ValueError("times must be finite")
        knot_q, knot_v = self.unpack(coeffs)
        q = np.zeros((sample_times.size, self.n_dof), dtype=float)
        v = np.zeros_like(q)
        a = np.zeros_like(q)
        q_basis = np.zeros((sample_times.size, self.n_dof, self.coefficient_size))
        v_basis = np.zeros_like(q_basis)
        a_basis = np.zeros_like(q_basis)
        for sample_index, sample_time in enumerate(sample_times):
            segment = self._segment_index(float(sample_time))
            t0 = self.knot_times[segment]
            t1 = self.knot_times[segment + 1]
            h = float(t1 - t0)
            s = (float(sample_time) - float(t0)) / h
            basis = _hermite_basis(s, h)
            left_q = knot_q[segment]
            right_q = knot_q[segment + 1]
            left_v = knot_v[segment]
            right_v = knot_v[segment + 1]
            q[sample_index] = (
                basis.q0 * left_q
                + basis.q1 * right_q
                + basis.v0 * left_v
                + basis.v1 * right_v
            )
            v[sample_index] = (
                basis.dq0 * left_q
                + basis.dq1 * right_q
                + basis.dv0 * left_v
                + basis.dv1 * right_v
            )
            a[sample_index] = (
                basis.ddq0 * left_q
                + basis.ddq1 * right_q
                + basis.ddv0 * left_v
                + basis.ddv1 * right_v
            )
            self._fill_basis_row(
                q_basis[sample_index],
                v_basis[sample_index],
                a_basis[sample_index],
                segment,
                basis,
            )
        return SplineTrajectoryEvaluation(
            sample_times, q, v, a, q_basis, v_basis, a_basis
        )

    def _segment_index(self, sample_time: float) -> int:
        clamped = min(
            max(sample_time, float(self.knot_times[0])), float(self.knot_times[-1])
        )
        right = int(np.searchsorted(self.knot_times, clamped, side="right"))
        return min(max(right - 1, 0), self.n_knots - 2)

    def _fill_basis_row(
        self,
        q_basis: np.ndarray,
        v_basis: np.ndarray,
        a_basis: np.ndarray,
        segment: int,
        basis: _HermiteBasis,
    ) -> None:
        velocity_offset = self.n_knots * self.n_dof
        for dof in range(self.n_dof):
            left_q_col = segment * self.n_dof + dof
            right_q_col = (segment + 1) * self.n_dof + dof
            left_v_col = velocity_offset + left_q_col
            right_v_col = velocity_offset + right_q_col
            q_basis[dof, left_q_col] = basis.q0
            q_basis[dof, right_q_col] = basis.q1
            q_basis[dof, left_v_col] = basis.v0
            q_basis[dof, right_v_col] = basis.v1
            v_basis[dof, left_q_col] = basis.dq0
            v_basis[dof, right_q_col] = basis.dq1
            v_basis[dof, left_v_col] = basis.dv0
            v_basis[dof, right_v_col] = basis.dv1
            a_basis[dof, left_q_col] = basis.ddq0
            a_basis[dof, right_q_col] = basis.ddq1
            a_basis[dof, left_v_col] = basis.ddv0
            a_basis[dof, right_v_col] = basis.ddv1


@dataclass(frozen=True)
class MapDecisionLayout:
    """Column layout of the MAP decision vector."""

    trajectory_size: int
    parameter_names: tuple[str, ...]

    @property
    def size(self) -> int:
        """Total decision-vector width."""
        return self.trajectory_size + len(self.parameter_names)

    def parameter_column(self, name: str) -> int:
        """Return the absolute decision-vector column for a parameter."""
        try:
            index = self.parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.trajectory_size + index


@dataclass(frozen=True)
class MapEstimatorOptions:
    """Numerical options for the single-trial MAP solve."""

    max_iterations: int = 50
    xtol: float = 1e-10
    ftol: float = 1e-10
    gtol: float = 1e-10
    method: Literal["trf", "lm"] = "trf"


@dataclass(frozen=True)
class MapEstimatorProblem:
    """Complete single-trial MAP problem definition."""

    trajectory: CubicHermiteSplineTrajectory
    evaluation_times: np.ndarray
    initial_coefficients: np.ndarray
    shared_parameters: SharedParameterBlock
    residual: ResidualFn
    jacobian: JacobianFn | None = None
    options: MapEstimatorOptions = MapEstimatorOptions()
    provenance: ProvenanceStamp | None = None


@dataclass(frozen=True)
class MapEstimatorResult:
    """Deterministic result of a single-trial MAP solve."""

    success: bool
    coefficients: np.ndarray
    parameters: dict[str, float]
    residual: np.ndarray
    objective: float
    n_iterations: int
    message: str
    provenance: ProvenanceStamp | None = None


def solve_single_trial_map(problem: MapEstimatorProblem) -> MapEstimatorResult:
    """Solve a single-trial MAP problem with spline and shared parameters."""
    _validate_problem(problem)
    x0 = _pack_decision(problem.initial_coefficients, problem.shared_parameters)
    lower, upper = _decision_bounds(problem)
    layout = MapDecisionLayout(
        trajectory_size=problem.trajectory.coefficient_size,
        parameter_names=tuple(spec.name for spec in problem.shared_parameters.specs),
    )

    def residual_for_solver(x: np.ndarray) -> np.ndarray:
        return _objective_residual(problem, x)

    jacobian_for_solver = None
    if problem.jacobian is not None:

        def jacobian_for_solver(x: np.ndarray) -> np.ndarray:
            return _objective_jacobian(problem, layout, x)

    method = problem.options.method
    if method == "lm" and _has_finite_bounds(lower, upper):
        method = "trf"
    result = least_squares(
        residual_for_solver,
        x0,
        jac=jacobian_for_solver if jacobian_for_solver is not None else "2-point",
        bounds=(lower, upper),
        method=method,
        max_nfev=problem.options.max_iterations,
        xtol=problem.options.xtol,
        ftol=problem.options.ftol,
        gtol=problem.options.gtol,
    )
    residual = residual_for_solver(result.x)
    coefficients, parameter_values = _unpack_decision(problem, result.x)
    return MapEstimatorResult(
        success=bool(result.success),
        coefficients=coefficients,
        parameters=problem.shared_parameters.to_mapping(parameter_values),
        residual=residual,
        objective=_squared_norm_objective(residual),
        n_iterations=int(result.nfev),
        message=str(result.message),
        provenance=problem.provenance,
    )


def _validate_problem(problem: MapEstimatorProblem) -> None:
    require(problem.trajectory is not None, "trajectory must be provided")
    eval_times = np.asarray(problem.evaluation_times, dtype=float)
    require(eval_times.ndim == 1, "evaluation_times must be a 1D array")
    require(bool(np.all(np.isfinite(eval_times))), "evaluation_times must be finite")
    coeffs = np.asarray(problem.initial_coefficients, dtype=float)
    require(
        coeffs.shape == (problem.trajectory.coefficient_size,),
        "initial_coefficients shape must match trajectory",
    )
    require(bool(np.all(np.isfinite(coeffs))), "initial_coefficients must be finite")
    require(problem.options.max_iterations > 0, "max_iterations must be positive")


def _pack_decision(
    coefficients: np.ndarray,
    parameter_block: SharedParameterBlock,
) -> np.ndarray:
    return np.concatenate(
        [np.asarray(coefficients, dtype=float), parameter_block.initial_vector()]
    )


def _unpack_decision(
    problem: MapEstimatorProblem,
    decision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(decision, dtype=float)
    split = problem.trajectory.coefficient_size
    return x[:split], x[split:]


def _decision_bounds(problem: MapEstimatorProblem) -> tuple[np.ndarray, np.ndarray]:
    trajectory_size = problem.trajectory.coefficient_size
    param_lower, param_upper = problem.shared_parameters.bounds()
    lower = np.concatenate([np.full(trajectory_size, -np.inf), param_lower])
    upper = np.concatenate([np.full(trajectory_size, np.inf), param_upper])
    return lower, upper


def _objective_residual(
    problem: MapEstimatorProblem, decision: np.ndarray
) -> np.ndarray:
    coefficients, parameter_values = _unpack_decision(problem, decision)
    evaluation = problem.trajectory.evaluate(coefficients, problem.evaluation_times)
    parameters = problem.shared_parameters.to_mapping(parameter_values)
    data_residual = np.asarray(problem.residual(evaluation, parameters), dtype=float)
    if data_residual.ndim != 1:
        raise ValueError("residual callable must return a 1D array")
    if not np.all(np.isfinite(data_residual)):
        raise ValueError("residual callable returned non-finite values")
    prior_residual = problem.shared_parameters.prior_residuals(parameter_values)
    return np.concatenate([data_residual, prior_residual])


def _objective_jacobian(
    problem: MapEstimatorProblem,
    layout: MapDecisionLayout,
    decision: np.ndarray,
) -> np.ndarray:
    if problem.jacobian is None:
        raise ValueError("jacobian callable must be provided")
    coefficients, parameter_values = _unpack_decision(problem, decision)
    evaluation = problem.trajectory.evaluate(coefficients, problem.evaluation_times)
    parameters = problem.shared_parameters.to_mapping(parameter_values)
    data_jacobian = np.asarray(
        problem.jacobian(evaluation, parameters, layout), dtype=float
    )
    if data_jacobian.ndim != 2 or data_jacobian.shape[1] != layout.size:
        raise ValueError(f"jacobian callable must return (*, {layout.size})")
    prior_parameter_jacobian = problem.shared_parameters.prior_jacobian()
    if prior_parameter_jacobian.shape[0] == 0:
        return data_jacobian
    prior_jacobian = np.zeros(
        (prior_parameter_jacobian.shape[0], layout.size),
        dtype=float,
    )
    prior_jacobian[:, layout.trajectory_size :] = prior_parameter_jacobian
    return np.vstack([data_jacobian, prior_jacobian])


def _squared_norm_objective(residual: np.ndarray) -> float:
    return 0.5 * float(np.vdot(residual, residual))


def _has_finite_bounds(lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))


def _require_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class _HermiteBasis:
    q0: float
    q1: float
    v0: float
    v1: float
    dq0: float
    dq1: float
    dv0: float
    dv1: float
    ddq0: float
    ddq1: float
    ddv0: float
    ddv1: float


def _hermite_basis(s: float, h: float) -> _HermiteBasis:
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    dh00 = (6.0 * s**2 - 6.0 * s) / h
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = (-6.0 * s**2 + 6.0 * s) / h
    dh11 = 3.0 * s**2 - 2.0 * s
    ddh00 = (12.0 * s - 6.0) / h**2
    ddh10 = (6.0 * s - 4.0) / h
    ddh01 = (-12.0 * s + 6.0) / h**2
    ddh11 = (6.0 * s - 2.0) / h
    return _HermiteBasis(
        q0=h00,
        q1=h01,
        v0=h * h10,
        v1=h * h11,
        dq0=dh00,
        dq1=dh01,
        dv0=dh10,
        dv1=dh11,
        ddq0=ddh00,
        ddq1=ddh01,
        ddv0=ddh10,
        ddv1=ddh11,
    )
