"""Start-pose conformance of each lift pack (LIFT-3, #11743).

Per (engine, lift) this reduces the pack's start pose to six checks:

* ``bar_start``: pulls start with the bar resting on the floor (bar centre one
  plate radius above the sole); squat and bench start with the bar carried at
  the shoulders / over the chest (no rack body is modelled in any pack, so
  rack support itself is not observable);
* ``feet_flat_in_contact``: both foot origins level and the lowest foot point
  on the ground.  Foot pitch is not observable from body origins, so this is
  necessary rather than sufficient;
* ``hand_bar_closure``: worst hand-to-bar closure residual <= 5 mm;
* ``joint_limits``: no start default and no phase reference outside the
  joint's range;
* ``static_grf``: vertical ground reaction within 2 % of body plus bar weight.

The check functions are pure (numbers in, :class:`CheckResult` out).  Only
:func:`evaluate_start_conformance` touches an engine adapter, and it reuses
the LIFT-1 adapters, phases and geometry instead of loading anything itself.
Unavailable is never zero: a quantity an engine cannot supply is
``status="unavailable"`` with ``value=None`` and a reason.  Limits are fixed;
a failing pack is recorded as failing, never re-tuned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .analysis import FLOOR_PULLS, PLATE_RADIUS_M, lifter_mass
from .model import CoordinateInfo, EngineAdapter, require_lift
from .names import engine_coordinate_name

BAR_FLOOR_TOL_M = 0.01
BAR_SHOULDER_TOL_M = 0.15
BENCH_ARM_REACH_M = 0.80
FEET_LEVEL_TOL_M = 0.001
FEET_GROUND_TOL_M = 0.005
CLOSURE_LIMIT_M = 0.005
GRF_REL_TOL = 0.02
LIMIT_SLACK_RAD = 1e-9
GRAVITY_MPS2 = 9.80665

CHECK_NAMES = (
    "bar_start",
    "feet_flat_in_contact",
    "hand_bar_closure",
    "joint_limits",
    "static_grf",
)
_STATUSES = ("pass", "fail", "unavailable")
#: Engines whose world has a ground plane at z = 0; Pinocchio's world origin
#: is the pelvis and its URDF carries no ground.
GROUND_Z: Mapping[str, float | None] = {
    "mujoco": 0.0,
    "opensim": 0.0,
    "drake": 0.0,
    "pinocchio": None,
}
_ENGINE_ERRORS = (
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    OSError,
    ImportError,
)


@dataclass(frozen=True)
class CheckResult:
    """One check: measured value against a fixed limit."""

    name: str
    status: str
    value: float | None
    limit: float | None
    unit: str
    reason: str | None = None
    detail: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {self.status!r}")
        if self.status == "unavailable":
            if self.value is not None:
                raise ValueError("an unavailable check must carry value=None")
            if not self.reason:
                raise ValueError("an unavailable check needs a reason")


@dataclass(frozen=True)
class StartConformance:
    """All checks for one engine and lift."""

    engine: str
    lift: str
    checks: tuple[CheckResult, ...]

    @property
    def failing(self) -> tuple[CheckResult, ...]:
        """Checks with ``status == "fail"``."""
        return tuple(c for c in self.checks if c.status == "fail")

    @property
    def passed(self) -> bool:
        """True when nothing failed (unavailable checks do not count as pass)."""
        return not self.failing

    def check(self, name: str) -> CheckResult:
        """Return the named check (``KeyError`` if absent)."""
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "engine": self.engine,
            "lift": self.lift,
            "checks": [{**c.__dict__, "detail": list(c.detail)} for c in self.checks],
        }


def _unavailable(name: str, limit: float | None, unit: str, reason: str) -> CheckResult:
    return CheckResult(name, "unavailable", None, limit, unit, reason)


def _finite(label: str, *values: float) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value}")


def _point(positions: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in positions:
        raise ValueError(f"body {name!r} (upper_arm bodies required) missing")
    arr = np.asarray(positions[name], dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"body {name!r} position must be a finite 3-vector")
    return arr


def check_bar_start(
    lift: str,
    bar_above_sole_m: float | None,
    positions: Mapping[str, Any] | None,
) -> CheckResult:
    """Bar on the floor for pulls; carried at shoulders / over chest otherwise.

    Pulls: ``|bar centre height above sole - 0.225 m| <= 0.01 m``.  Squat: bar
    within 0.15 m of the shoulder midpoint.  Bench: bar within 0.15 m of the
    shoulder midpoint horizontally and 0 to 0.80 m above it.
    """
    require_lift(lift)
    name = "bar_start"
    if lift in FLOOR_PULLS:
        if bar_above_sole_m is None:
            return _unavailable(name, BAR_FLOOR_TOL_M, "m", "engine reports no sole")
        _finite("bar_above_sole_m", bar_above_sole_m)
        err = abs(bar_above_sole_m - PLATE_RADIUS_M)
        status = "pass" if err <= BAR_FLOOR_TOL_M else "fail"
        return CheckResult(
            name,
            status,
            err,
            BAR_FLOOR_TOL_M,
            "m",
            detail=(f"bar centre {bar_above_sole_m:.4f} m above sole",),
        )
    if positions is None:
        raise ValueError("positions are required for squat and bench_press")
    bar = _point(positions, "barbell_shaft")
    mid = 0.5 * (_point(positions, "upper_arm_l") + _point(positions, "upper_arm_r"))
    rel = bar - mid
    horizontal = float(np.linalg.norm(rel[:2]))
    if lift == "squat":
        err = float(np.linalg.norm(rel))
        ok = err <= BAR_SHOULDER_TOL_M
    else:
        err = horizontal
        ok = horizontal <= BAR_SHOULDER_TOL_M and 0.0 <= rel[2] <= BENCH_ARM_REACH_M
    return CheckResult(
        name,
        "pass" if ok else "fail",
        err,
        BAR_SHOULDER_TOL_M,
        "m",
        detail=(f"bar - shoulder midpoint = {np.round(rel, 4).tolist()}",),
    )


def check_feet_flat(
    foot_l_z: float,
    foot_r_z: float,
    sole_z: float | None,
    ground_z: float | None,
) -> CheckResult:
    """Feet level with each other and the lowest foot point on the ground."""
    name = "feet_flat_in_contact"
    _finite("foot heights", foot_l_z, foot_r_z)
    limit = FEET_LEVEL_TOL_M
    if sole_z is None:
        return _unavailable(name, limit, "m", "engine reports no sole height")
    if ground_z is None:
        return _unavailable(
            name, limit, "m", "engine world has no ground reference to test contact"
        )
    _finite("sole_z/ground_z", sole_z, ground_z)
    level = abs(foot_l_z - foot_r_z)
    gap = abs(sole_z - ground_z)
    ok = level <= FEET_LEVEL_TOL_M and gap <= FEET_GROUND_TOL_M
    return CheckResult(
        name,
        "pass" if ok else "fail",
        max(level, gap),
        FEET_GROUND_TOL_M,
        "m",
        detail=(
            f"foot origin height difference {level * 1000:.2f} mm "
            f"(limit {FEET_LEVEL_TOL_M * 1000:.0f} mm)",
            f"lowest foot point {gap * 1000:.2f} mm from ground",
        ),
    )


def check_hand_bar_closure(
    closure_m: Mapping[str, float] | None, reason: str = "no closure residual"
) -> CheckResult:
    """Worst per-hand closure residual against the 5 mm limit."""
    name = "hand_bar_closure"
    if not closure_m:
        return _unavailable(name, CLOSURE_LIMIT_M, "m", reason)
    values = [float(v) for v in closure_m.values()]
    if any(v < 0 for v in values):
        raise ValueError("closure residuals must be non-negative")
    if any(not math.isfinite(v) for v in values):
        return _unavailable(name, CLOSURE_LIMIT_M, "m", "non-finite closure residual")
    worst = max(values)
    return CheckResult(
        name,
        "pass" if worst <= CLOSURE_LIMIT_M else "fail",
        worst,
        CLOSURE_LIMIT_M,
        "m",
        detail=tuple(f"{k}: {v * 1000:.3f} mm" for k, v in closure_m.items()),
    )


def check_joint_limits(
    coordinates: Sequence[CoordinateInfo],
    phase_targets: Mapping[str, Mapping[str, float]],
) -> CheckResult:
    """Count start defaults and phase references outside their joint range.

    *phase_targets* maps phase name to ``{native coordinate: canonical angle}``;
    the native value is the canonical angle plus the coordinate's zero offset.
    """
    by_name = {c.name: c for c in coordinates}
    out: list[str] = []
    for coord in coordinates:
        if coord.lower is None or coord.upper is None or coord.default is None:
            continue
        if (
            not coord.lower - LIMIT_SLACK_RAD
            <= coord.default
            <= (coord.upper + LIMIT_SLACK_RAD)
        ):
            out.append(
                f"start:{coord.name}={coord.default:.4f} "
                f"outside [{coord.lower:.4f}, {coord.upper:.4f}]"
            )
    for phase, targets in phase_targets.items():
        for native, angle in targets.items():
            if native not in by_name:
                raise ValueError(f"unknown coordinate {native!r} in phase {phase!r}")
            coord = by_name[native]
            if coord.lower is None or coord.upper is None:
                continue
            value = angle + coord.zero_offset
            if (
                not coord.lower - LIMIT_SLACK_RAD
                <= value
                <= (coord.upper + LIMIT_SLACK_RAD)
            ):
                out.append(
                    f"{phase}:{native}={value:.4f} "
                    f"outside [{coord.lower:.4f}, {coord.upper:.4f}]"
                )
    return CheckResult(
        "joint_limits",
        "fail" if out else "pass",
        float(len(out)),
        0.0,
        "count",
        detail=tuple(out),
    )


def expected_support_n(masses: Mapping[str, float]) -> float:
    """Body plus bar weight (bench excluded), newtons."""
    bar = sum(m for n, m in masses.items() if n.startswith("barbell_"))
    total = lifter_mass(dict(masses)) + bar
    if total <= 0:
        raise ValueError("segment masses give no lifter or bar mass")
    return total * GRAVITY_MPS2


def check_static_grf(
    value_n: float | None,
    expected_n: float,
    reason: str = "engine reports no ground contact force",
) -> CheckResult:
    """Relative error of the vertical ground force against body plus bar weight."""
    name = "static_grf"
    if not math.isfinite(expected_n) or expected_n <= 0:
        raise ValueError(f"expected_n must be positive and finite, got {expected_n}")
    if value_n is None:
        return _unavailable(name, GRF_REL_TOL, "fraction", reason)
    _finite("value_n", value_n)
    rel = abs(value_n - expected_n) / expected_n
    return CheckResult(
        name,
        "pass" if rel <= GRF_REL_TOL else "fail",
        rel,
        GRF_REL_TOL,
        "fraction",
        detail=(f"measured {value_n:.1f} N, expected {expected_n:.1f} N",),
    )


# ---- engine glue (reuses the LIFT-1 adapters; no model loading here) ------


def _closure_of(adapter: EngineAdapter, closure: Any) -> tuple[Any, str]:
    """Per-hand closure in metres from the adapter, else the bar-hold residual."""
    if isinstance(closure, dict) and closure:
        if "weld_translation_norm" in closure:
            return {"weld": closure["weld_translation_norm"]}, ""
        hands = {k: v for k, v in closure.items() if k in ("l", "r")}
        if hands:
            return hands, ""
    try:
        hold = adapter.bar_hold_wrench()
    except _ENGINE_ERRORS as exc:
        return None, f"bar-hold adapter failed: {exc}"
    resid = hold.get("hand_grip_residual_m") if hold.get("available") else None
    if resid:
        return {k: v for k, v in resid.items() if v is not None}, ""
    why = hold.get("reason") or "no hand weld constraint in this pack"
    return None, f"no closure residual from adapter ({why})"


def _phase_targets(adapter: EngineAdapter, pack: Any) -> dict[str, dict[str, float]]:
    from . import phases

    names = {c.name for c in adapter.coordinates()}
    out: dict[str, dict[str, float]] = {}
    try:
        listing = phases.pack_phases(pack, adapter.lift)
    except (KeyError, *_ENGINE_ERRORS):
        return out
    for ph in listing:
        out[ph.name] = {
            engine_coordinate_name(adapter.engine, canon): angle
            for canon, angle in ph.angles.items()
            if engine_coordinate_name(adapter.engine, canon) in names
        }
    return out


def _grf_check(contact: Mapping[str, Any], expected_n: float) -> CheckResult:
    """GRF check, with the adapter's contact counts as context.

    The adapter evaluates the start pose statically; no settling integration
    is run, so ``value_n`` is the contact force at the pack start itself.
    """
    res = check_static_grf(
        contact.get("value_n"),
        expected_n,
        str(contact.get("reason") or "engine reports no ground contact force"),
    )
    extra = tuple(
        f"{key}={contact[key]}" for key in ("n_ground_contacts",) if key in contact
    )
    return replace(res, detail=res.detail + extra)


def evaluate_start_conformance(adapter: EngineAdapter, pack: Any) -> StartConformance:
    """Run every check on *adapter*'s pack start pose."""
    ev = adapter.evaluate(None)
    sole = ev.sole_z
    bar_above = None if sole is None else float(ev.positions["barbell_shaft"][2] - sole)
    closure, closure_reason = _closure_of(adapter, ev.closure)
    contact = adapter.start_contact_force()
    expected = expected_support_n(adapter.segment_masses())
    checks = (
        check_bar_start(adapter.lift, bar_above, ev.positions),
        check_feet_flat(
            float(ev.positions["foot_l"][2]),
            float(ev.positions["foot_r"][2]),
            sole,
            GROUND_Z.get(adapter.engine),
        ),
        check_hand_bar_closure(closure, closure_reason),
        check_joint_limits(adapter.coordinates(), _phase_targets(adapter, pack)),
        _grf_check(contact, expected),
    )
    return StartConformance(adapter.engine, adapter.lift, checks)
