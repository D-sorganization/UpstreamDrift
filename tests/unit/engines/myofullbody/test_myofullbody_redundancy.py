"""Static-optimisation tests for MyoFullBody (issue #11645)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import musculoskeletal_static_opt as so
from src.shared.python.myofullbody import redundancy

pytestmark = pytest.mark.unit


def _toy(tau: float, weight: float):
    """Two muscles on one joint: force capacity 1000 N, moment arms 4 cm and 2 cm."""
    active = np.array([1000.0, 1000.0])
    passive = np.zeros(2)
    moment = np.array([[0.04], [0.02]])
    return redundancy.solve_frame(
        active, passive, moment, np.array([tau]), reserve_weight=weight
    )


def test_toy_two_muscle_one_joint_matches_the_analytic_optimum() -> None:
    """Interior optimum is a = w tau c / (1 + w |c|^2) with c_i = r_i F_i."""
    c = np.array([40.0, 20.0])
    for tau, weight in ((10.0, 1.0), (10.0, 1e4), (3.0, 100.0)):
        expected = weight * tau * c / (1.0 + weight * float(c @ c))
        sol = _toy(tau, weight)
        np.testing.assert_allclose(sol.activation, expected, atol=1e-7)
        assert sol.success
        np.testing.assert_allclose(sol.reserve, tau - c @ sol.activation, atol=1e-9)


def test_toy_bounds_saturate_and_leave_a_reserve() -> None:
    """A demand beyond the capacity (60 N m) saturates both muscles."""
    sol = _toy(100.0, 1e3)
    np.testing.assert_allclose(sol.activation, [1.0, 1.0], atol=1e-5)
    assert sol.reserve[0] == pytest.approx(40.0, abs=1e-4)


def test_toy_stronger_muscle_is_recruited_first() -> None:
    """Minimum sum of squares weights recruitment by moment arm times force."""
    sol = _toy(20.0, 1e6)
    assert sol.activation[0] == pytest.approx(2.0 * sol.activation[1], rel=1e-4)


def test_passive_force_offsets_the_demand() -> None:
    active, passive = np.array([500.0]), np.array([100.0])
    moment = np.array([[0.05]])
    sol = redundancy.solve_frame(active, passive, moment, np.array([5.0]))
    # passive alone gives 5 N m, so no activation is needed
    assert sol.activation[0] == pytest.approx(0.0, abs=1e-6)
    assert abs(sol.reserve[0]) < 1e-6


def test_fast_solver_agrees_with_bounded_least_squares() -> None:
    rng = np.random.default_rng(3)
    nm, nc = 30, 6
    active = rng.uniform(100.0, 800.0, nm)
    passive = rng.uniform(0.0, 20.0, nm)
    moment = rng.normal(0.0, 0.03, (nm, nc))
    tau = rng.normal(0.0, 25.0, nc)
    fast = redundancy.solve_frame(active, passive, moment, tau, reserve_weight=50.0)
    ref = so.solve_frame(active, passive, moment, tau, reserve_weight=50.0)
    assert fast.cost <= ref.cost * (1.0 + 1e-5)
    np.testing.assert_allclose(fast.activation, ref.activation, atol=2e-3)
    assert fast.activation.min() >= 0.0 and fast.activation.max() <= 1.0


def test_solver_contract() -> None:
    with pytest.raises(ValueError):
        redundancy.solve_frame(
            np.ones(2), np.zeros(2), np.ones((2, 1)), np.ones(1), reserve_weight=0.0
        )
    with pytest.raises(ValueError):
        redundancy.solve_frame(np.ones(2), np.zeros(3), np.ones((2, 1)), np.ones(1))
    with pytest.raises(ValueError):
        redundancy.solve_frame(
            np.array([np.nan, 1.0]), np.zeros(2), np.ones((2, 1)), np.ones(1)
        )


def test_coordinate_groups_partition_the_non_root_coordinates() -> None:
    order = (
        "TranslationInputX", "HipInputX", "SpineInputX", "TorsoInput", "LEInput",
        "RScapInputY", "LWInputX", "NeckInputZ", "hip_flexion_r", "mtp_angle_l",
    )  # fmt: skip
    groups = redundancy.coordinate_groups(order)
    assert groups["trunk"] == [2, 3]
    assert groups["arms"] == [4, 5, 6]
    assert groups["neck"] == [7]
    assert groups["legs"] == [8, 9]
    with pytest.raises(ValueError):
        redundancy.coordinate_groups(("Mystery",))


def test_group_metrics_and_fail_closed_status() -> None:
    tau = np.array([[10.0, 0.0], [10.0, 0.0]])
    reserve = np.array([[0.5, 0.0], [0.5, 0.0]])
    metrics = redundancy.group_reserve_metrics(
        reserve, tau, {"legs": [4], "arms": [5]}, [4, 5]
    )
    assert metrics["legs"]["reserve_over_effort_rms"] == pytest.approx(0.05)
    assert metrics["arms"]["reserve_over_effort_rms"] == 0.0
    ok = redundancy.qualification(metrics, True, True)
    assert ok["status"] == "QUALIFIED_SOFTWARE_ONLY"
    assert redundancy.qualification(metrics, False, True)["status"] == "NOT_QUALIFIED"
    assert redundancy.qualification(metrics, True, False)["status"] == "NOT_QUALIFIED"
    bad = {"legs": {**metrics["legs"], "reserve_over_effort_rms": 0.4}}
    assert redundancy.qualification(bad, True, True)["status"] == "NOT_QUALIFIED"
    assert redundancy.qualification({}, True, True)["status"] == "NOT_QUALIFIED"
    with pytest.raises(ValueError):
        redundancy.qualification(metrics, True, True, limit=0.0)


def test_neck_without_muscles_is_a_declared_scope_limit_not_a_pass() -> None:
    metrics = {
        "legs": {"reserve_over_effort_rms": 0.01},
        "neck": {"reserve_over_effort_rms": 1.0},
    }
    result = redundancy.qualification(metrics, True, True)
    assert result["status"] == "QUALIFIED_SOFTWARE_ONLY"
    assert result["uncovered_groups_scope_limitation"] == ["neck"]
