"""Reserve attribution tests (issue #11689): waterfall and kinematic floor."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.myofullbody import attribution, redundancy

pytestmark = pytest.mark.unit


def _frames(n: int = 4):
    """One coordinate, one muscle: capacity 50 N m, passive 10 N m, demand 40 N m."""
    basis = redundancy.FrameBasis(
        active=np.array([50.0]),
        passive=np.array([10.0]),
        moment=np.array([[1.0, 0.0]]),
        phi=np.array([[1.0, 0.0], [0.0, 0.0]]),
    )
    tau = np.tile([[60.0, 5.0]], (n, 1))
    return [basis] * n, tau


def test_waterfall_separates_passive_capacity_and_structure() -> None:
    bases, tau = _frames()
    out = attribution.waterfall(bases, tau, {"legs": [0], "arms": [1]}, [0, 1], 100.0)
    legs, arms = out["legs"], out["arms"]
    assert legs["baseline"] >= 0.0
    assert legs["passive_off"] >= legs["capacity_unlimited"]
    # the second coordinate has no muscle at all: its demand is pure structure
    assert arms["capacity_unlimited"] == pytest.approx(1.0, abs=1e-6)
    assert arms["baseline"] == pytest.approx(1.0, abs=1e-6)
    # no moment arm, unlimited capacity: the first coordinate is fully coverable
    assert legs["capacity_unlimited"] < 0.01


def test_waterfall_variants_are_ordered_by_what_they_remove() -> None:
    bases, tau = _frames()
    legs = attribution.waterfall(bases, tau, {"legs": [0]}, [0], 100.0)["legs"]
    assert legs["baseline"] >= legs["capacity_x3_passive_off"] - 1e-9
    assert legs["passive_off"] >= legs["capacity_x3_passive_off"] - 1e-9


def test_kinematic_floor_is_zero_when_effort_is_representable() -> None:
    bases, tau = _frames()
    tau = np.tile([[20.0, 0.0]], (4, 1))
    floor = attribution.kinematic_floor(bases, tau, {"legs": [0], "arms": [1]}, [0, 1])
    assert floor["legs"] == pytest.approx(0.0, abs=1e-9)
    assert floor["arms"] == 0.0  # no effort in the group


def test_kinematic_floor_counts_the_unrepresentable_component() -> None:
    bases, tau = _frames()
    floor = attribution.kinematic_floor(bases, tau, {"arms": [1]}, [0, 1])
    assert floor["arms"] == pytest.approx(1.0)


def test_capacity_audit_reports_demand_against_independent_capacity() -> None:
    rng = np.random.default_rng(0)
    nm, nc = 6, 2
    bases = [
        redundancy.FrameBasis(
            rng.uniform(100, 200, nm), rng.uniform(0, 5, nm),
            rng.normal(0, 0.05, (nm, nc)), np.zeros((3, 2)),
        )
        for _ in range(5)
    ]  # fmt: skip
    tau = rng.normal(0, 30, (5, nc))
    audit = attribution.capacity_audit(bases, tau, ["a", "b"])
    assert set(audit) == {"a", "b"}
    for row in audit.values():
        assert row["peak_demand_nm"] > 0.0
        assert 0.0 <= row["frames_demand_exceeds_capacity"] <= 1.0
        assert row["median_capacity_over_demand"] > 0.0


def test_waterfall_rejects_shape_mismatch() -> None:
    bases, tau = _frames()
    with pytest.raises(ValueError):
        attribution.waterfall(bases, tau[:2], {"legs": [0]}, [0, 1], 100.0)


def test_reserve_ratios_run_in_parallel_with_the_same_numbers() -> None:
    bases, tau = _frames(6)
    variants = [("a", 1.0, 1.0, None), ("b", 1.0, 0.0, 10.0), ("c", 3.0, 0.0, None)]
    groups = {"legs": [0], "arms": [1]}
    serial = attribution.reserve_ratios(bases, tau, groups, [0, 1], variants, 100.0)
    parallel = attribution.reserve_ratios(
        bases, tau, groups, [0, 1], variants, 100.0, workers=2
    )
    assert [r["name"] for r in serial] == ["a", "b", "c"]
    for s, p in zip(serial, parallel, strict=True):
        assert s["reserve_over_effort_rms"] == pytest.approx(
            p["reserve_over_effort_rms"]
        )
        assert s["converged"] and p["converged"]


def test_phase_breakdown_splits_energy_at_the_key_frames() -> None:
    steps = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    keys = {"address": 0, "top": 20, "impact": 50, "finish": 70}
    tau = np.ones((8, 1)) * 10.0
    reserve = np.zeros((8, 1))
    reserve[5:] = 5.0  # only at and after impact
    out = attribution.phase_breakdown(steps, keys, reserve, tau, {"g": [0]}, [0], 15)
    g = out["g"]
    assert set(g) == {"backswing", "downswing", "impact_window", "follow_through"}
    assert g["backswing"]["reserve_share"] == 0.0
    assert g["follow_through"]["reserve_share"] == pytest.approx(1.0 / 3.0)
    assert g["impact_window"]["reserve_share"] == pytest.approx(2.0 / 3.0)
    total = sum(v["effort_share"] for v in g.values())
    assert total == pytest.approx(1.0)
    assert g["impact_window"]["frames"] == 3  # steps 40, 50, 60 within +-15 of 50
    with pytest.raises(ValueError):
        attribution.phase_breakdown(steps, keys, reserve, tau, {"g": [0]}, [0], -1)


def test_scaling_can_be_limited_to_the_muscles_ahead_of_torque_actuators() -> None:
    basis = redundancy.FrameBasis(
        active=np.array([50.0, 10.0]),  # one muscle, one torque actuator
        passive=np.array([0.0, 0.0]),
        moment=np.array([[1.0], [1.0]]),
        phi=np.zeros((1, 1)),
    )
    tau = np.array([[200.0]])
    rows = attribution.reserve_ratios(
        [basis], tau, {"g": [0]}, [0], [("x", 1000.0, 1.0, None)], 100.0, n_scaled=1
    )
    # muscle is effectively unlimited, so the actuator's 10 N m is not what limits
    assert rows[0]["reserve_over_effort_rms"]["g"] < 0.01
    unscaled = attribution.reserve_ratios(
        [basis], tau, {"g": [0]}, [0], [("x", 1000.0, 1.0, None)], 100.0, n_scaled=0
    )
    # nothing scaled: the 10 N m actuator and 50 N m muscle cannot reach 200
    assert unscaled[0]["reserve_over_effort_rms"]["g"] > 0.2
