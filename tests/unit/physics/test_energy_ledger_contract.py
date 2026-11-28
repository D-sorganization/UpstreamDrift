"""Tests for the engine-agnostic momentum & energy ledger contract (#15689)."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest

from src.shared.python.physics import energy_ledger_contract as elc

SCH = elc.ENERGY_LEDGER_SCHEMA


def _f(val: object) -> float:
    """Assert value is a float and return it typed for mypy."""
    assert isinstance(val, float)
    return val


def _audit(
    s: elc.EnergyLedgerSeries, et: float, pt: float, rm: bool = True
) -> elc.ConservationAuditResult:
    """Run audit_energy_ledger with concise tolerance arguments."""
    kw = {
        "energy_tolerance_j": et,
        "momentum_tolerance_ns": pt,
        "angular_momentum_tolerance_nms": pt,
    }
    return elc.audit_energy_ledger(s, **kw, require_momentum=rm)


class _DummyAdapter:
    """Minimal mock engine adapter implementing EnergyLedgerAdapter."""

    ENGINE: ClassVar[str] = "dummy-engine"

    def availability(self) -> elc.LedgerAvailability:
        """Declare supported components for the dummy engine."""
        return elc.LedgerAvailability(True, True, True, False, False, True, False)

    def to_energy_ledger(self, raw_trace: object) -> elc.EnergyLedgerSeries:
        """Convert raw dictionary trace into an EnergyLedgerSeries."""
        if not isinstance(raw_trace, dict):
            raise TypeError("raw_trace must be a dict")
        u, step_cls = elc.UNAVAILABLE, elc.EnergyLedgerStep
        steps = tuple(
            step_cls(t, ke, pe, w, u, u, p, u, f)
            for t, ke, pe, w, p, f in raw_trace["samples"]
        )
        m, av = float(raw_trace["mass_kg"]), self.availability()
        return elc.EnergyLedgerSeries(SCH, self.ENGINE, av, steps, m)


@pytest.mark.unit
@pytest.mark.validates_physics
def test_free_fall_analytic_fixture_conserves_energy_and_momentum() -> None:
    """Free-falling rigid body satisfies T+V=const and dP/dt=m*g to 1e-12."""
    s = elc.build_free_fall_analytic_fixture(mass_kg=2.5, duration_s=1.0, dt_s=0.01)
    assert s.schema_version == SCH and len(s.steps) == 101
    tol = 1e-12
    r = _audit(s, tol, tol)
    assert isinstance(r, elc.ConservationAuditResult) and r.passed and not r.violations
    assert abs(_f(r.max_abs_energy_residual_j)) < tol
    assert abs(_f(r.max_abs_linear_momentum_residual_ns)) < tol
    assert abs(_f(r.max_abs_angular_momentum_residual_nms)) < tol


@pytest.mark.unit
@pytest.mark.validates_physics
def test_driven_pendulum_analytic_fixture_matches_actuator_work() -> None:
    """Frictionless torque-driven pendulum satisfies d(T + V) == W_act."""
    s = elc.build_driven_pendulum_analytic_fixture(torque_nm=15.0, duration_s=0.5)
    r = _audit(s, 1e-9, 1e-4)
    assert r.passed and abs(_f(r.max_abs_energy_residual_j)) < 1e-9
    s0, s1 = s.steps[0], s.steps[-1]
    d_mech = (
        _f(s1.kinetic_j) + _f(s1.potential_j) - _f(s0.kinetic_j) - _f(s0.potential_j)
    )
    d_w = _f(s1.actuator_work_j) - _f(s0.actuator_work_j)
    assert d_w > 0.1 and abs(d_mech - d_w) < 1e-9


@pytest.mark.unit
def test_audit_fails_closed_on_energy_leak() -> None:
    """Injecting an unphysical energy leak fails the conservation audit."""
    s = elc.build_free_fall_analytic_fixture(duration_s=0.2, dt_s=0.05)
    steps = list(s.steps)
    steps[-1] = replace(steps[-1], kinetic_j=_f(steps[-1].kinetic_j) - 0.25)
    r = _audit(replace(s, steps=tuple(steps)), 1e-9, 1e-9, False)
    assert not r.passed and len(r.violations) >= 1
    assert _f(r.max_abs_energy_residual_j) >= 0.25 - 1e-12


@pytest.mark.unit
def test_audit_fails_closed_on_momentum_mismatch() -> None:
    """Injecting an impulse-momentum mismatch fails the conservation audit."""
    s = elc.build_free_fall_analytic_fixture(duration_s=0.2, dt_s=0.05)
    steps = list(s.steps)
    p = steps[-1].linear_momentum_ns
    assert isinstance(p, tuple)
    steps[-1] = replace(steps[-1], linear_momentum_ns=(p[0] + 0.5, p[1], p[2]))
    r = _audit(replace(s, steps=tuple(steps)), 1e-9, 1e-9)
    assert not r.passed and _f(r.max_abs_linear_momentum_residual_ns) >= 0.5 - 1e-12
    assert any("momentum" in msg.lower() for msg in r.violations)


@pytest.mark.unit
def test_unavailable_sentinel_rejects_zero_filling_and_fails_closed() -> None:
    """Unsupported components must use UNAVAILABLE and never zero-fill."""
    av = elc.LedgerAvailability(True, True, True, False, False, True, False)
    z, u, ser_cls = (0.0, 0.0, 0.0), elc.UNAVAILABLE, elc.EnergyLedgerSeries
    bad = elc.EnergyLedgerStep(0.0, 1.0, 2.0, 0.0, u, u, z, u, z)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        ser_cls(SCH, "partial", av, (replace(bad, dissipation_j=0.0),))
    s = ser_cls(SCH, "partial", av, (bad, replace(bad, time_s=0.1)))
    audit = elc.audit_energy_ledger(s)
    ang_res = audit.max_abs_angular_momentum_residual_nms
    assert not audit.passed and ang_res is u
    assert any("unavailable" in msg.lower() for msg in audit.violations)


@pytest.mark.unit
def test_energy_ledger_adapter_protocol_normalizes_raw_trace() -> None:
    """EnergyLedgerAdapter normalizes raw engine outputs into EnergyLedgerSeries."""
    adapter = _DummyAdapter()
    assert isinstance(adapter, elc.EnergyLedgerAdapter)
    z, p, u = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), elc.UNAVAILABLE
    samples = [(0.0, 0.5, 1.0, 0.0, p, z), (0.1, 0.5, 1.0, 0.0, p, z)]
    s = elc.normalize_engine_ledger(adapter, {"mass_kg": 1.0, "samples": samples})
    assert s.engine == "dummy-engine" and s.schema_version == SCH
    s0 = s.steps[0]
    assert s0.dissipation_j is u and s0.angular_momentum_nms is u
