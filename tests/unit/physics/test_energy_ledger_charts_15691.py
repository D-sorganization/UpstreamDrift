"""Tests for signed component carpets & cumulative waterfalls (#15691)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.physics import energy_ledger_charts as elch
from src.shared.python.physics import energy_ledger_contract as elc

SCH = elc.ENERGY_LEDGER_SCHEMA


def _f(val: object) -> float:
    """Assert val is a finite float and return it typed for mypy."""
    assert isinstance(val, float) and np.isfinite(val)
    return val


@pytest.mark.unit
@pytest.mark.validates_physics
def test_signed_component_carpet_free_fall_sync_and_bounds() -> None:
    """Carpet synchronizes times_s, orders labels, and enforces vmin == -vmax > 0."""
    series = elc.build_free_fall_analytic_fixture(duration_s=0.5, dt_s=0.05)
    comps = ("kinetic_j", "potential_j", "actuator_work_j", "linear_momentum_z_ns")
    carpet = elch.build_signed_component_carpet(series, components=comps)
    assert isinstance(carpet, elch.SignedComponentCarpet)
    assert carpet.schema_version == SCH and carpet.engine == series.engine
    assert carpet.times_s == tuple(s.time_s for s in series.steps)
    assert carpet.component_labels == comps
    assert carpet.values.shape == (4, len(series.steps))
    assert not np.any(carpet.unavailable_mask) and np.all(np.isfinite(carpet.values))
    assert carpet.vmax > 0.0 and carpet.vmin == -carpet.vmax
    pz = carpet.component_row("linear_momentum_z_ns")
    assert pz[0] > 0.0 and pz[-1] < 0.0
    assert np.all(np.abs(carpet.values) <= carpet.vmax + 1e-15)


@pytest.mark.unit
def test_signed_component_carpet_zero_series_positive_vmax() -> None:
    """All-zero series still produces symmetric non-degenerate scale bounds."""
    av = elc.LedgerAvailability(True, True, True, True, True, True, True)
    z = (0.0, 0.0, 0.0)
    st = elc.EnergyLedgerStep
    steps = (
        st(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, z, z, z, z),
        st(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, z, z, z, z),
    )
    ser = elc.EnergyLedgerSeries(SCH, "zero-eng", av, steps)
    carpet = elch.build_signed_component_carpet(ser)
    assert carpet.vmax > 0.0 and carpet.vmin == -carpet.vmax
    assert np.all(carpet.values == 0.0) and not np.any(carpet.unavailable_mask)


@pytest.mark.unit
def test_signed_component_carpet_preserves_unavailable() -> None:
    """UNAVAILABLE w_act and t_rot stay NaN with True mask, never zero-filled."""
    av = elc.LedgerAvailability(True, True, False, True, True, False, False)
    u, st = elc.UNAVAILABLE, elc.EnergyLedgerStep
    steps = (
        st(0.0, 1.5, 2.0, u, 0.0, 0.0),
        st(0.1, 2.5, 1.0, u, 0.0, 0.0),
        st(0.2, 3.0, 0.5, u, 0.0, 0.0),
    )
    series = elc.EnergyLedgerSeries(SCH, "partial-eng", av, steps)
    t_rot: list[elc.LedgerScalar] = [0.25, u, 0.75]
    comps = ("kinetic_j", "w_act", "t_rot")
    carpet = elch.build_signed_component_carpet(
        series, components=comps, extra_components={"t_rot": t_rot}
    )
    assert carpet.component_labels == comps
    assert np.all(np.isnan(carpet.component_row("w_act")))
    assert np.all(carpet.unavailable_mask[1, :])
    t_row = carpet.component_row("t_rot")
    assert np.isclose(t_row[0], 0.25) and np.isnan(t_row[1])
    assert list(carpet.unavailable_mask[2, :]) == [False, True, False]
    assert not np.any(carpet.unavailable_mask[0, :])
    assert carpet.vmax > 0.0 and carpet.vmin == -carpet.vmax


@pytest.mark.unit
@pytest.mark.validates_physics
def test_cumulative_waterfall_free_fall_exact_closure() -> None:
    """Free-fall waterfall satisfies exact work and impulse closure (< 1e-12)."""
    series = elc.build_free_fall_analytic_fixture(duration_s=1.0, dt_s=0.01)
    wf = elch.build_cumulative_waterfall(series)
    assert isinstance(wf, elch.CumulativeWaterfall)
    assert wf.times_s == tuple(s.time_s for s in series.steps)
    assert _f(wf.closure_residual_j) < 1e-12
    assert _f(wf.impulse_closure_residual_ns) < 1e-12
    assert np.allclose(wf.w_act_j, 0.0) and np.allclose(wf.w_ext_j, 0.0)
    assert np.allclose(wf.w_diss_j, 0.0) and np.max(np.abs(wf.de_total_j)) < 1e-12
    assert np.allclose(wf.j_ext_ns, wf.dp_ns, atol=1e-12)
    assert [t.label for t in wf.work_terms] == ["W_act", "W_ext", "W_diss", "dE_total"]
    assert all(isinstance(t, elch.WaterfallTerm) and t.available for t in wf.work_terms)


@pytest.mark.unit
@pytest.mark.validates_physics
def test_cumulative_waterfall_driven_pendulum_tight_closure() -> None:
    """Driven pendulum waterfall matches W_act == dE_total and J_ext == dP."""
    series = elc.build_driven_pendulum_analytic_fixture(torque_nm=12.0, duration_s=0.4)
    wf = elch.build_cumulative_waterfall(series)
    assert _f(wf.closure_residual_j) < 1e-9
    assert _f(wf.impulse_closure_residual_ns) < 1e-4
    w_val = _f(wf.work_term("W_act").final_value)
    de_val = _f(wf.work_term("dE_total").final_value)
    assert w_val > 0.1 and abs(w_val - de_val) < 1e-9
    assert np.allclose(wf.w_act_j, wf.de_total_j, atol=1e-9)


@pytest.mark.unit
def test_cumulative_waterfall_preserves_unavailable() -> None:
    """Waterfall keeps UNAVAILABLE terms NaN and fails closed on closure."""
    base = elc.build_free_fall_analytic_fixture(duration_s=0.2, dt_s=0.05)
    u = elc.UNAVAILABLE
    av = replace(base.availability, actuator_work=False, linear_momentum=False)
    steps = tuple(
        replace(s, actuator_work_j=u, linear_momentum_ns=u, net_external_force_n=u)
        for s in base.steps
    )
    wf = elch.build_cumulative_waterfall(replace(base, availability=av, steps=steps))
    assert wf.closure_residual_j is u and wf.impulse_closure_residual_ns is u
    assert np.all(np.isnan(wf.w_act_j)) and np.all(np.isnan(wf.j_ext_ns))
    w_act = wf.work_term("W_act")
    assert not w_act.available and w_act.final_value is u
    assert wf.work_term("W_diss").available and wf.work_term("dE_total").available
