"""Tests for Slice 2 per-segment & system energy ledger evaluator (#15690)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.physics import energy_ledger_contract as elc
from src.shared.python.physics import energy_ledger_evaluator as ele


def _f(val: object) -> float:
    """Assert value is a float and return it typed for mypy."""
    assert isinstance(val, float)
    return val


def _sv(vec: elc.LedgerVector3, frac: float) -> elc.LedgerVector3:
    """Scale a 3-vector ledger field while preserving UNAVAILABLE."""
    if isinstance(vec, elc.UnavailableType):
        return elc.UNAVAILABLE
    return (vec[0] * frac, vec[1] * frac, vec[2] * frac)


def _ss(val: elc.LedgerScalar, frac: float) -> elc.LedgerScalar:
    """Scale a scalar ledger field while preserving UNAVAILABLE."""
    return (
        elc.UNAVAILABLE if isinstance(val, elc.UnavailableType) else float(val) * frac
    )


def _split(
    system: elc.EnergyLedgerSeries, fractions: tuple[float, ...]
) -> tuple[ele.SegmentEnergyRecord, ...]:
    """Partition an EnergyLedgerSeries into conservative per-segment records."""
    recs: list[ele.SegmentEnergyRecord] = []
    sch, eng, av = system.schema_version, system.engine, system.availability
    for idx, f in enumerate(fractions):
        steps = tuple(
            elc.EnergyLedgerStep(
                s.time_s,
                _ss(s.kinetic_j, f),
                _ss(s.potential_j, f),
                _ss(s.actuator_work_j, f),
                _ss(s.dissipation_j, f),
                _ss(s.external_work_j, f),
                _sv(s.linear_momentum_ns, f),
                _sv(s.angular_momentum_nms, f),
                _sv(s.net_external_force_n, f),
                _sv(s.net_external_torque_nm, f),
            )
            for s in system.steps
        )
        m = system.mass_kg * f if system.mass_kg is not None else None
        recs.append(
            ele.SegmentEnergyRecord(
                f"seg_{idx}", elc.EnergyLedgerSeries(sch, eng, av, steps, m)
            )
        )
    return tuple(recs)


@pytest.mark.unit
@pytest.mark.validates_physics
@pytest.mark.parametrize(
    "system",
    [
        elc.build_free_fall_analytic_fixture(mass_kg=2.5, duration_s=0.5, dt_s=0.01),
        elc.build_driven_pendulum_analytic_fixture(torque_nm=15.0, duration_s=0.4),
    ],
)
def test_analytic_fixtures_pass_strict_analytic(system: elc.EnergyLedgerSeries) -> None:
    """Analytic fixtures split across segments pass strict_analytic policy."""
    pol = ele.tolerance_policy_preset("strict_analytic")
    res = ele.evaluate_segment_and_system_balance(
        system, _split(system, (0.35, 0.65)), pol
    )
    assert res.passed and res.status == "pass" and res.segment_sum_violation is None
    assert _f(res.segment_sum_residual_j) < 1e-12
    assert res.failing_segment_ids == () and res.unavailable_segment_ids == ()
    assert all(seg.status == "pass" for seg in res.segment_evaluations)


@pytest.mark.unit
def test_segment_sum_mismatch_fails_closed_with_diagnostic() -> None:
    """Mismatch between sum(segment_energies) and system_energy fails closed."""
    sys_s = elc.build_free_fall_analytic_fixture(duration_s=0.2, dt_s=0.05)
    seg_a, seg_b = _split(sys_s, (0.5, 0.5))
    sb = list(seg_b.series.steps)
    sb[-1] = replace(sb[-1], potential_j=_f(sb[-1].potential_j) + 0.25)
    bad_b = replace(seg_b, series=replace(seg_b.series, steps=tuple(sb)))
    res = ele.evaluate_segment_and_system_balance(
        sys_s, (seg_a, bad_b), "strict_analytic"
    )
    assert not res.passed and res.status == "fail" and res.segment_sum_violation
    assert "segment_sum_violation" in res.segment_sum_violation
    assert _f(res.segment_sum_residual_j) >= 0.25 - 1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    ("preset", "leak_j", "mom_delta"),
    [
        ("strict_analytic", 1e-6, 1e-2),
        ("forward_replay", 0.05, 0.25),
        ("inverse_dynamics", 0.5, 0.5),
    ],
)
@pytest.mark.parametrize("mode", ["energy", "momentum"])
def test_injected_segment_faults_fail_with_named_segment(
    preset: str, leak_j: float, mom_delta: float, mode: str
) -> None:
    """Injected per-segment energy leak or momentum mismatch fails with segment ID."""
    sys_s = elc.build_free_fall_analytic_fixture(duration_s=0.2, dt_s=0.05)
    seg_ok, seg_bad = _split(sys_s, (0.4, 0.6))
    sb = list(seg_bad.series.steps)
    if mode == "energy":
        sb[-1] = replace(sb[-1], kinetic_j=_f(sb[-1].kinetic_j) + leak_j)
    else:
        p = sb[-1].linear_momentum_ns
        assert isinstance(p, tuple)
        sb[-1] = replace(sb[-1], linear_momentum_ns=(p[0] + mom_delta, p[1], p[2]))
    patched = replace(seg_bad, series=replace(seg_bad.series, steps=tuple(sb)))
    res = ele.evaluate_segment_and_system_balance(sys_s, (seg_ok, patched), preset)
    assert not res.passed and res.status == "fail"
    assert "seg_1" in res.failing_segment_ids and "seg_0" not in res.failing_segment_ids
    by_id = {ev.segment_id: ev for ev in res.segment_evaluations}
    assert by_id["seg_1"].status == "fail" and by_id["seg_0"].status == "pass"


@pytest.mark.unit
def test_unavailable_segment_terms_preserved_and_reported_cleanly() -> None:
    """UNAVAILABLE segment terms never zero-fill and report status='unavailable'."""
    sys_s = elc.build_free_fall_analytic_fixture(duration_s=0.1, dt_s=0.05)
    seg_ok, seg_half = _split(sys_s, (0.5, 0.5))
    u, sch = elc.UNAVAILABLE, sys_s.schema_version
    av_p = elc.LedgerAvailability(True, True, True, False, False, True, False)
    st_p = tuple(
        replace(
            s,
            dissipation_j=u,
            external_work_j=u,
            angular_momentum_nms=u,
            net_external_torque_nm=u,
        )
        for s in seg_half.series.steps
    )
    seg_p = ele.SegmentEnergyRecord(
        "seg_partial", elc.EnergyLedgerSeries(sch, "p", av_p, st_p)
    )
    res = ele.evaluate_segment_and_system_balance(
        sys_s, (seg_ok, seg_p), "strict_analytic"
    )
    assert (
        not res.passed
        and res.status == "unavailable"
        and "seg_partial" in res.unavailable_segment_ids
    )
    ev_p = {ev.segment_id: ev for ev in res.segment_evaluations}["seg_partial"]
    assert ev_p.status == "unavailable" and ev_p.max_abs_energy_residual_j is u
    assert (
        ev_p.max_abs_angular_momentum_residual_nms is u
        and "dissipation" in ev_p.unavailable_components
    )

    av_k = elc.LedgerAvailability(False, True, True, True, True, True, True)
    st_k = tuple(replace(s, kinetic_j=u) for s in seg_half.series.steps)
    seg_k = ele.SegmentEnergyRecord(
        "seg_no_ke", elc.EnergyLedgerSeries(sch, "k", av_k, st_k)
    )
    res_k = ele.evaluate_segment_and_system_balance(
        sys_s, (seg_ok, seg_k), "forward_replay"
    )
    assert res_k.status == "unavailable" and res_k.segment_sum_residual_j is u


@pytest.mark.unit
def test_tolerance_policy_sha256_is_deterministic() -> None:
    """Identical TolerancePolicy instances yield identical deterministic policy_sha256."""
    p1 = ele.tolerance_policy_preset("forward_replay")
    p2 = ele.tolerance_policy_preset("forward_replay")
    p3 = ele.TolerancePolicy("forward_replay", "1.0.0", 1e-3, 1e-2, 1e-2, 1e-6, True)
    assert len(p1.policy_sha256) == 64
    assert p1.policy_sha256 == p2.policy_sha256 == p3.policy_sha256
    p_diff = replace(p1, energy_tolerance_j=p1.energy_tolerance_j * 2.0)
    assert p1.policy_sha256 != p_diff.policy_sha256
    with pytest.raises(ValueError, match="unknown"):
        ele.tolerance_policy_preset("nonexistent_preset")
