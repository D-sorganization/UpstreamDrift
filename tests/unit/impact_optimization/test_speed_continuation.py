"""SHOW-C5 (#14561): speed continuation 100 -> 125 mph on the reduced plant."""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.speed_continuation import (
    LadderSpec,
    binding_report,
    pareto_front,
    run_speed_continuation,
)
from src.shared.python.impact_optimization.trajectory_optimiser import (
    OptimiserConfig,
)
from tests.unit.impact_optimization.test_trajectory_optimiser import (
    BASE,
    TIMES,
    _envelope,
    _trust,
    plant,
)

pytestmark = pytest.mark.unit


def _run(envelope, **kw):
    ladder = LadderSpec(**kw) if kw else LadderSpec()
    return run_speed_continuation(
        rollout=plant,
        baseline_torque_nm=BASE,
        times_s=TIMES,
        envelope=envelope,
        trust_region=_trust(2000.0),
        config=OptimiserConfig(n_knots=5, seed=0, max_iterations=60),
        ladder=ladder,
    )


def test_default_ladder_is_100_to_125_in_steps_of_5() -> None:
    res = _run(_envelope())
    assert [p.speed_target_mph for p in res.points] == [
        100.0,
        105.0,
        110.0,
        115.0,
        120.0,
        125.0,
    ]


def test_generous_envelope_reaches_125_within_tolerances() -> None:
    res = _run(_envelope())
    last = res.points[-1]
    assert abs(last.solution.outcome.impact.speed_mph - 125.0) <= 1.0
    assert last.solution.tolerances_met
    assert res.reached_mph == 125.0


def test_tight_joint_stops_the_ladder_and_is_named() -> None:
    env = _envelope(first=BASE[:, 0].max() * 1.001)
    res = _run(env)
    assert res.reached_mph < 125.0
    lines = binding_report(res, env)
    assert any("J0" in ln for ln in lines)
    assert any("miscalibrat" in ln for ln in lines)  # firm 125: limit is suspect


def test_pareto_front_is_non_dominated_and_ordered_by_speed() -> None:
    res = _run(_envelope())
    front = pareto_front(res.points)
    assert front and len(front) <= len(res.points)
    speeds = [p.achieved_mph for p in front]
    assert speeds == sorted(speeds)
    for a in front:
        for b in front:
            dominated = (
                b.achieved_mph >= a.achieved_mph
                and b.solution.deviation < a.solution.deviation
                and b is not a
            )
            assert not dominated


def test_points_report_face_to_path_gap() -> None:
    res = _run(_envelope())
    for p in res.points:
        imp = p.solution.outcome.impact
        assert p.solution.face_to_path_deg == pytest.approx(imp.face_deg - imp.path_deg)


@pytest.mark.parametrize(
    "kw", [{"step_mph": 0.0}, {"start_mph": 130.0, "stop_mph": 125.0}]
)
def test_invalid_ladder_rejected(kw: dict) -> None:
    with pytest.raises(ContractViolationError):
        _run(_envelope(), **kw)


def test_binding_line_carries_the_limit_provenance_note() -> None:
    """A limit that binds on a prior-derived split must say so (show-trunk)."""
    env = _envelope(first=BASE[:, 0].max() * 1.001)
    res = _run(env)
    note = "prior-derived split, not a measurement"
    lines = binding_report(res, env, limit_notes={"J0": note})
    j0 = [ln for ln in lines if "J0" in ln]
    assert j0 and all(note in ln for ln in j0)
    assert not any("over-rotation" in ln for ln in lines)


def test_unlisted_joint_has_no_provenance_note() -> None:
    env = _envelope(first=BASE[:, 0].max() * 1.001)
    res = _run(env)
    lines = binding_report(res, env, limit_notes={"J1": "x"})
    assert not any("x)" in ln for ln in lines if "J0" in ln)


def test_every_binding_line_names_its_limit_kind_as_torque() -> None:
    """Torque-only today: ROM/velocity binding is C3c (show-review item 1)."""
    env = _envelope(first=BASE[:, 0].max() * 1.001)
    lines = binding_report(_run(env), env)
    assert lines and all("torque limit" in ln for ln in lines)
    assert not any("ROM" in ln or "velocity" in ln for ln in lines)


def test_solution_exposes_achieved_speed_without_a_deep_chain() -> None:
    """LoD: callers read the speed from the solution, not through 3 hops."""
    res = _run(_envelope())
    for p in res.points:
        assert p.solution.achieved_speed_mph == p.solution.outcome.impact.speed_mph
        assert p.achieved_mph == p.solution.achieved_speed_mph


def test_rollout_outcome_exposes_speed_directly() -> None:
    out = plant(BASE)
    assert out.speed_mph == out.impact.speed_mph


def test_blocking_torque_on_an_assumed_slot_is_assumption_limited() -> None:
    """D32: a rung binding on an assumption-based limit is not a miscalibration."""
    import dataclasses

    env = dataclasses.replace(
        _envelope(first=BASE[:, 0].max() * 1.001), assumed_torque_slots=("J0",)
    )
    lines = [ln for ln in binding_report(_run(env), env) if "J0" in ln]
    blocking = [ln for ln in lines if "BLOCKS" in ln]
    assert blocking and all("assumption-limited" in ln for ln in blocking)
    assert not any("miscalibrat" in ln for ln in blocking)
    assert all("assumed" in ln for ln in blocking)  # says the limit is not cited


def _j0_peak(res) -> float:
    return float(max(abs(p.torque_nm[:, 0]).max() for p in res.points))


def test_feasible_rung_above_the_lower_source_is_source_sensitive() -> None:
    """D34: the envelope limit IS the upper cited value; Pan is the lower band."""
    peak = _j0_peak(_run(_envelope()))
    high = 1.5 * peak  # upper cited value = the envelope limit: nothing blocks
    env = _envelope(first=high)
    res = _run(env)
    lines = binding_report(res, env, source_band={"J0": (0.5 * peak, high)})
    sens = [ln for ln in lines if "source-sensitive" in ln]
    assert sens and not any("BLOCKS" in ln for ln in sens)
    assert all("lower" in ln and "upper" in ln and "J0" in ln for ln in sens)
    assert not any("miscalibrat" in ln for ln in lines)


def test_peak_below_the_lower_source_is_not_flagged() -> None:
    peak = _j0_peak(_run(_envelope()))
    env = _envelope(first=3.0 * peak)
    lines = binding_report(_run(env), env, source_band={"J0": (2.0 * peak, 3.0 * peak)})
    assert not any("source-sensitive" in ln for ln in lines)


def test_block_at_the_ceiling_keeps_the_d29_wording() -> None:
    cap = BASE[:, 0].max() * 1.001  # blocked at the upper value: blocked either way
    env = _envelope(first=cap)
    lines = binding_report(_run(env), env, source_band={"J0": (0.5 * cap, cap)})
    blocking = [ln for ln in lines if "J0" in ln and "BLOCKS" in ln]
    assert blocking and all("miscalibrat" in ln for ln in blocking)
    assert not any("source-sensitive" in ln for ln in blocking)


@pytest.mark.parametrize("band", [(0.0, 5000.0), (6000.0, 5000.0), (10.0, 4000.0)])
def test_a_band_that_is_not_below_the_envelope_limit_is_refused(band) -> None:
    env = _envelope()  # J0 limit 5000 must equal the upper cited value
    with pytest.raises(ContractViolationError):
        binding_report(_run(env), env, source_band={"J0": band})
