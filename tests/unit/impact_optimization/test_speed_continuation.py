"""SHOW-C5 (#14561): speed continuation 100 -> 125 mph on the reduced plant."""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.speed_continuation import (
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
    return run_speed_continuation(
        rollout=plant,
        baseline_torque_nm=BASE,
        times_s=TIMES,
        envelope=envelope,
        trust_region=_trust(2000.0),
        config=OptimiserConfig(n_knots=5, seed=0, max_iterations=60),
        **kw,
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
