"""Engine-agnostic momentum and energy ledger contract and analytic fixtures (#15689)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.shared.python.core.constants import GRAVITY_FLOAT

ENERGY_LEDGER_SCHEMA: str = "energy-ledger/1.0.0"
Vector3 = tuple[float, float, float]


class UnavailableType:
    """Sentinel type for unsupported ledger components (never zero-fill)."""

    def __repr__(self) -> str:
        """Return canonical sentinel representation."""
        return "UNAVAILABLE"


UNAVAILABLE: UnavailableType = UnavailableType()
LedgerScalar = float | UnavailableType
LedgerVector3 = Vector3 | UnavailableType

_SCALAR_FLAGS = (
    "kinetic",
    "potential",
    "actuator_work",
    "dissipation",
    "external_work",
)
_VECTOR_FIELDS = (
    ("linear_momentum_ns", "linear_momentum"),
    ("net_external_force_n", "linear_momentum"),
    ("angular_momentum_nms", "angular_momentum"),
    ("net_external_torque_nm", "angular_momentum"),
)


def _is_finite(v: object) -> bool:
    """Return True if v is a finite real number (excluding bool)."""
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _check_field(label: str, v: object, avail: bool, is_vec: bool) -> None:
    """Enforce availability flag vs UNAVAILABLE sentinel."""
    if not avail:
        if v is not UNAVAILABLE:
            raise ValueError(f"{label} is unavailable and must be UNAVAILABLE")
    elif v is UNAVAILABLE:
        raise ValueError(f"{label} is available and cannot be UNAVAILABLE")
    elif is_vec:
        if not isinstance(v, tuple) or len(v) != 3 or not all(_is_finite(c) for c in v):
            raise ValueError(f"{label} must be a 3-tuple of finite floats")
    elif not _is_finite(v):
        raise ValueError(f"{label} must be a finite float")


@dataclass(frozen=True)
class LedgerAvailability:
    """Explicit per-component availability flags for an engine ledger."""

    kinetic: bool = True
    potential: bool = True
    actuator_work: bool = True
    dissipation: bool = True
    external_work: bool = True
    linear_momentum: bool = True
    angular_momentum: bool = True

    def __post_init__(self) -> None:
        """Validate that all availability flags are booleans."""
        if not all(
            isinstance(getattr(self, k), bool) for k in self.__dataclass_fields__
        ):
            raise TypeError("all availability flags must be booleans")


@dataclass(frozen=True)
class EnergyLedgerStep:
    """Single timestamped momentum and mechanical energy ledger sample."""

    time_s: float
    kinetic_j: LedgerScalar
    potential_j: LedgerScalar
    actuator_work_j: LedgerScalar = UNAVAILABLE
    dissipation_j: LedgerScalar = UNAVAILABLE
    external_work_j: LedgerScalar = UNAVAILABLE
    linear_momentum_ns: LedgerVector3 = UNAVAILABLE
    angular_momentum_nms: LedgerVector3 = UNAVAILABLE
    net_external_force_n: LedgerVector3 = UNAVAILABLE
    net_external_torque_nm: LedgerVector3 = UNAVAILABLE

    def __post_init__(self) -> None:
        """Validate time finiteness and kinetic energy non-negativity."""
        if not _is_finite(self.time_s):
            raise ValueError("time_s must be a finite float")
        if not isinstance(self.kinetic_j, UnavailableType):
            if not _is_finite(self.kinetic_j) or float(self.kinetic_j) < -1e-15:
                raise ValueError("kinetic_j must be a non-negative finite float")


@dataclass(frozen=True)
class EnergyLedgerSeries:
    """Validated time-ordered momentum and energy ledger series."""

    schema_version: str
    engine: str
    availability: LedgerAvailability
    steps: tuple[EnergyLedgerStep, ...]
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        """Validate schema, ordering, and sentinel/availability consistency."""
        if self.schema_version != ENERGY_LEDGER_SCHEMA:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        m = self.mass_kg
        if m is not None and (not _is_finite(m) or m <= 0):
            raise ValueError("mass_kg must be a positive finite float")
        av, prev_t = self.availability, None
        for idx, step in enumerate(self.steps):
            if prev_t is not None and step.time_s <= prev_t:
                raise ValueError(f"non-increasing time_s at step {idx}")
            prev_t = step.time_s
            for flag_name in _SCALAR_FLAGS:
                attr = f"{flag_name}_j"
                _check_field(attr, getattr(step, attr), getattr(av, flag_name), False)
            for attr, flag_name in _VECTOR_FIELDS:
                _check_field(attr, getattr(step, attr), getattr(av, flag_name), True)


@dataclass(frozen=True)
class ConservationAuditResult:
    """Result of auditing an EnergyLedgerSeries against physical laws."""

    passed: bool
    schema_version: str
    engine: str
    step_count: int
    max_abs_energy_residual_j: LedgerScalar
    max_abs_step_energy_residual_j: LedgerScalar
    final_energy_drift_j: LedgerScalar
    max_abs_linear_momentum_residual_ns: LedgerScalar
    max_abs_angular_momentum_residual_nms: LedgerScalar
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate audit result consistency invariants."""
        if self.passed != (len(self.violations) == 0):
            raise ValueError("passed must equal (len(violations) == 0)")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative")


@runtime_checkable
class EnergyLedgerAdapter(Protocol):
    """Protocol normalizing raw engine outputs into EnergyLedgerSeries."""

    ENGINE: ClassVar[str]

    def availability(self) -> LedgerAvailability:
        """Declare which ledger terms this adapter supports."""

    def to_energy_ledger(self, raw_trace: object) -> EnergyLedgerSeries:
        """Convert raw engine trajectory outputs into an EnergyLedgerSeries."""


def normalize_engine_ledger(
    adapter: EnergyLedgerAdapter, raw_trace: object
) -> EnergyLedgerSeries:
    """Normalize raw engine output via an adapter and verify identity consistency."""
    series = adapter.to_energy_ledger(raw_trace)
    if series.engine != adapter.ENGINE:
        raise ValueError(f"engine mismatch: {adapter.ENGINE!r} != {series.engine!r}")
    if series.availability != adapter.availability():
        raise ValueError("series availability does not match adapter availability")
    return series


def _step_energy_terms(step: EnergyLedgerStep) -> tuple[float, float, float]:
    """Extract (mechanical_energy, net_work_minus_diss, dissipation) from a step."""
    ke, pe = float(step.kinetic_j), float(step.potential_j)  # type: ignore[arg-type]
    wa, we = float(step.actuator_work_j), float(step.external_work_j)  # type: ignore[arg-type]
    diss = float(step.dissipation_j)  # type: ignore[arg-type]
    return ke + pe, wa + we - diss, diss


def _audit_energy(
    steps: tuple[EnergyLedgerStep, ...], tol_j: float, violations: list[str]
) -> tuple[float, float, float]:
    """Compute cumulative and step-wise mechanical energy ledger residuals."""
    e0, w0, d0 = _step_energy_terms(steps[0])
    max_c, max_s, fin, pe, pw, pd = 0.0, 0.0, 0.0, e0, w0, d0
    for idx in range(1, len(steps)):
        ec, wc, dc = _step_energy_terms(steps[idx])
        rc, rs = (ec - e0) - (wc - w0), (ec - pe) - (wc - pw)
        if dc - pd < -tol_j:
            violations.append(f"negative dissipation at step {idx}: {dc - pd:.3e} J")
        max_c, max_s, fin = max(max_c, abs(rc)), max(max_s, abs(rs)), rc
        pe, pw, pd = ec, wc, dc
    if max_c > tol_j or max_s > tol_j:
        violations.append(f"energy residual {max_c:.3e} J > {tol_j:.3e} J")
    return max_c, max_s, fin


def _audit_mom(
    steps: tuple[EnergyLedgerStep, ...],
    cfg: tuple[bool, bool, float, bool],
    violations: list[str],
) -> LedgerScalar:
    """Audit dM/dt = F_ext via trapezoidal impulse quadrature."""
    avail, is_lin, tol, req = cfg
    kind = "linear" if is_lin else "angular"
    label = f"{kind} momentum"
    m_attr = f"{kind}_momentum_n{'s' if is_lin else 'ms'}"
    f_attr = f"net_external_{'force_n' if is_lin else 'torque_nm'}"
    unit = "N*s" if is_lin else "N*m*s"
    if not avail or len(steps) < 2:
        if not avail and req:
            violations.append(f"{label} is UNAVAILABLE while require_momentum=True")
        return UNAVAILABLE
    p0: Vector3 = getattr(steps[0], m_attr)
    c_imp, max_res = [0.0, 0.0, 0.0], 0.0
    for idx in range(1, len(steps)):
        sa, sb = steps[idx - 1], steps[idx]
        dt = sb.time_s - sa.time_s
        pa, pb = getattr(sa, m_attr), getattr(sb, m_attr)
        fa, fb = getattr(sa, f_attr), getattr(sb, f_attr)
        imp = [0.5 * dt * (fa[i] + fb[i]) for i in range(3)]
        c_imp = [c_imp[i] + imp[i] for i in range(3)]
        r_s = math.sqrt(sum((pb[i] - pa[i] - imp[i]) ** 2 for i in range(3)))
        r_c = math.sqrt(sum((pb[i] - p0[i] - c_imp[i]) ** 2 for i in range(3)))
        max_res = max(max_res, r_s, r_c)
    if max_res > tol:
        violations.append(f"{label} residual {max_res:.3e} > {tol:.3e} {unit}")
    return max_res


def audit_energy_ledger(
    series: EnergyLedgerSeries,
    *,
    energy_tolerance_j: float = 1e-9,
    momentum_tolerance_ns: float = 1e-9,
    angular_momentum_tolerance_nms: float = 1e-9,
    require_momentum: bool = False,
) -> ConservationAuditResult:
    """Fail-closed audit of energy conservation and impulse-momentum balance."""
    te, tp, tl = (
        energy_tolerance_j,
        momentum_tolerance_ns,
        angular_momentum_tolerance_nms,
    )
    if any(not _is_finite(t) or t <= 0.0 for t in (te, tp, tl)):
        raise ValueError("all tolerances must be positive finite floats")
    av, steps, violations = series.availability, series.steps, []
    if len(steps) < 2:
        violations.append("at least 2 steps are required for conservation audit")
    u = UNAVAILABLE
    e_vals: tuple[LedgerScalar, LedgerScalar, LedgerScalar] = (u, u, u)
    if not all(getattr(av, f) for f in _SCALAR_FLAGS):
        violations.append("required energy/work ledger components are UNAVAILABLE")
    elif len(steps) >= 2:
        e_vals = _audit_energy(steps, te, violations)
    rm = require_momentum
    lin_r = _audit_mom(steps, (av.linear_momentum, True, tp, rm), violations)
    ang_r = _audit_mom(steps, (av.angular_momentum, False, tl, rm), violations)
    sch, eng = series.schema_version, series.engine
    n, v_t, res_cls = len(steps), tuple(violations), ConservationAuditResult
    return res_cls(not violations, sch, eng, n, *e_vals, lin_r, ang_r, v_t)


def build_free_fall_analytic_fixture(
    *,
    mass_kg: float = 2.0,
    initial_position_m: Vector3 = (0.0, 0.0, 10.0),
    initial_velocity_mps: Vector3 = (1.5, -0.5, 4.0),
    gravity_mps2: float = GRAVITY_FLOAT,
    duration_s: float = 1.0,
    dt_s: float = 0.01,
) -> EnergyLedgerSeries:
    """Build exact analytic free-fall rigid-body trajectory (T+V=const, dP/dt=m*g)."""
    m, g = mass_kg, gravity_mps2
    if min(m, g, duration_s, dt_s) <= 0.0:
        raise ValueError("mass_kg, gravity_mps2, duration_s, and dt_s must be positive")
    x0, y0, z0 = initial_position_m
    vx0, vy0, vz0 = initial_velocity_mps
    fz, f_v = -m * g, (0.0, 0.0, -m * g)
    steps: list[EnergyLedgerStep] = []
    add, step_cls = steps.append, EnergyLedgerStep
    for k in range(max(1, int(round(duration_s / dt_s))) + 1):
        t = k * dt_s
        rx, ry, rz = x0 + vx0 * t, y0 + vy0 * t, z0 + vz0 * t - 0.5 * g * t * t
        vx, vy, vz = vx0, vy0, vz0 - g * t
        px, py, pz = m * vx, m * vy, m * vz
        ke, pe = 0.5 * m * (vx * vx + vy * vy + vz * vz), m * g * rz
        p_v, t_v = (px, py, pz), (ry * fz, -rx * fz, 0.0)
        l_v = (ry * pz - rz * py, rz * px - rx * pz, rx * py - ry * px)
        add(step_cls(t, ke, pe, 0.0, 0.0, 0.0, p_v, l_v, f_v, t_v))
    eng, av, ser_cls = "analytic-free-fall", LedgerAvailability(), EnergyLedgerSeries
    return ser_cls(ENERGY_LEDGER_SCHEMA, eng, av, tuple(steps), m)


def build_driven_pendulum_analytic_fixture(
    *,
    mass_kg: float = 1.5,
    length_m: float = 0.8,
    torque_nm: float = 1.2,
    initial_angle_rad: float = 0.25,
    initial_angular_velocity_rps: float = 0.0,
    gravity_mps2: float = GRAVITY_FLOAT,
    duration_s: float = 1.0,
    dt_s: float = 0.002,
) -> EnergyLedgerSeries:
    """Build frictionless torque-driven pendulum trajectory satisfying d(T+V)=W_act."""
    if min(mass_kg, length_m, duration_s, dt_s) <= 0.0:
        raise ValueError("mass_kg, length_m, duration_s, and dt_s must be positive")
    inertia, ml = mass_kg * length_m * length_m, mass_kg * length_m
    times = [k * dt_s for k in range(max(1, int(round(duration_s / dt_s))) + 1)]

    def _rhs(_t: float, state: NDArray[np.float64]) -> list[float]:
        """Return ODE derivative [dtheta/dt, domega/dt, dW_act/dt]."""
        th, om = float(state[0]), float(state[1])
        al = (torque_nm - ml * gravity_mps2 * math.sin(th)) / inertia
        return [om, al, torque_nm * om]

    y0 = np.array([initial_angle_rad, initial_angular_velocity_rps, 0.0], dtype=float)
    kw = {"method": "DOP853", "t_eval": times, "rtol": 1e-12, "atol": 1e-14}
    sol = solve_ivp(_rhs, (0.0, times[-1]), y0, **kw)  # type: ignore[arg-type, call-overload]
    if not sol.success:
        raise RuntimeError(f"pendulum ODE integration failed: {sol.message}")
    steps: list[EnergyLedgerStep] = []
    add, step_cls = steps.append, EnergyLedgerStep
    for k, t in enumerate(times):
        th, om, w_act = float(sol.y[0, k]), float(sol.y[1, k]), float(sol.y[2, k])
        al = (torque_nm - ml * gravity_mps2 * math.sin(th)) / inertia
        c, s = math.cos(th), math.sin(th)
        ke, pe = 0.5 * inertia * om * om, ml * gravity_mps2 * (1.0 - c)
        p_v = (ml * om * c, ml * om * s, 0.0)
        f_v = (ml * (al * c - om * om * s), ml * (al * s + om * om * c), 0.0)
        l_v, t_v = (0.0, 0.0, inertia * om), (0.0, 0.0, inertia * al)
        add(step_cls(t, ke, pe, w_act, 0.0, 0.0, p_v, l_v, f_v, t_v))
    av, ser_cls = LedgerAvailability(), EnergyLedgerSeries
    return ser_cls(
        ENERGY_LEDGER_SCHEMA, "analytic-driven-pendulum", av, tuple(steps), mass_kg
    )
