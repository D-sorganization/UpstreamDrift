"""Unit and regression tests for the proximal-to-distal timing analyses.

Pins reported values published on affinedrift.com,
enforces ordering invariants, and verifies internal energy-balance and
superposition checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_experiments import (
    _phase_energy_budget,
    counterfactual_split,
    rollout_program,
)
from scripts.research.proximal_distal_energy.e1d_parameter_sensitivity import (
    build_parameter_cases,
    evaluate_parameter_case,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
    segment_kinetic_energies,
    wrist_interface_powers,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    drive_only_program,
    passive_program,
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend


@pytest.fixture
def params() -> GolfModelParams:
    return GolfModelParams.default()


@pytest.fixture
def inertials(params: GolfModelParams) -> PlanarInertials:
    return PlanarInertials.from_params(params)


@pytest.mark.unit
def test_headline_ordering_invariants(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    """Assert ordering invariants: early_drive < passive < best_drive <= best_restrain."""
    # 60 N.m shoulder torque
    t_p, q_p, v_p, _ = rollout_program(params, passive_program(60.0))
    _, speed_p, _ = find_impact(t_p, q_p, v_p, inertials)

    t_e, q_e, v_e, _ = rollout_program(params, drive_only_program(60.0, 15.0, 0.0))
    _, speed_e, _ = find_impact(t_e, q_e, v_e, inertials)

    t_d, q_d, v_d, _ = rollout_program(params, drive_only_program(60.0, 15.0, 0.20))
    _, speed_d, _ = find_impact(t_d, q_d, v_d, inertials)

    t_r, q_r, v_r, _ = rollout_program(
        params, restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    )
    _, speed_r, _ = find_impact(t_r, q_r, v_r, inertials)

    assert speed_e < speed_p < speed_d <= speed_r


@pytest.mark.unit
def test_headline_pinned_numbers(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    """Assert exact published headline numbers within tight tolerance."""
    tau_s = 60.0
    t_p, q_p, v_p, _ = rollout_program(params, passive_program(tau_s))
    _, speed_p, _ = find_impact(t_p, q_p, v_p, inertials)

    t_e, q_e, v_e, _ = rollout_program(params, drive_only_program(tau_s, 15.0, 0.0))
    _, speed_e, _ = find_impact(t_e, q_e, v_e, inertials)

    t_d, q_d, v_d, _ = rollout_program(params, drive_only_program(tau_s, 15.0, 0.20))
    _, speed_d, _ = find_impact(t_d, q_d, v_d, inertials)

    # Percentage gains vs passive
    early_pct = (speed_e - speed_p) / speed_p * 100.0
    late_pct = (speed_d - speed_p) / speed_p * 100.0

    assert abs(early_pct - (-15.33)) < 0.5
    assert abs(late_pct - 12.42) < 0.5
    assert abs(speed_p - 34.15) < 0.5

    # 100 N.m shoulder torque best restrain
    t_r100, q_r100, v_r100, _ = rollout_program(
        params, restrain_then_drive_program(100.0, 15.0, 5.0, 0.125)
    )
    imp_r100 = find_impact(t_r100, q_r100, v_r100, inertials)
    assert imp_r100 is not None
    assert abs(imp_r100[1] - 46.91) < 0.5

    # 60 N.m best restrain energy budget checks
    t_r, q_r, v_r, u_r = rollout_program(
        params, restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    )
    imp_r = find_impact(t_r, q_r, v_r, inertials)
    assert imp_r is not None
    budget = _phase_energy_budget(inertials, t_r, q_r, v_r, u_r, imp_r[0])

    assert abs(budget["wrist_actuator_work_early_j"] - (-1.85)) < 0.5
    assert abs(budget["joint_force_transfer_late_j"] - 131.7) < 5.0
    assert abs(budget["wrist_actuator_work_late_j"] - 25.7) < 5.0


@pytest.mark.unit
def test_robertson_winter_energy_balance(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    """Assert integrated work done on club equals club kinetic energy gain."""
    prog = restrain_then_drive_program(60.0, 15.0, 5.0, 0.15)
    t, q, v, u = rollout_program(params, prog)
    imp = find_impact(t, q, v, inertials)
    assert imp is not None
    t_imp = imp[0]
    mask = t <= t_imp
    t_c, q_c, v_c, u_c = t[mask], q[mask], v[mask], u[mask]

    _, e_kin_club = segment_kinetic_energies(inertials, q_c, v_c)
    powers = wrist_interface_powers(inertials, t_c, q_c, v_c, u_c)

    integrated_work = np.trapezoid(
        powers["joint_force_power"]
        + powers["moment_power_on_club"]
        + powers["gravity_power_on_club"],
        t_c,
    )
    delta_e_club = e_kin_club[-1] - e_kin_club[0]
    assert abs(integrated_work - delta_e_club) < 1.0


@pytest.mark.unit
def test_drift_control_superposition(params: GolfModelParams) -> None:
    """Assert drift + control reconstructs total acceleration along trace."""
    backend = make_backend("ode", params)
    prog = drive_only_program(60.0, 15.0, 0.1)
    t, q, v, u = rollout_program(params, prog)
    drift, control = counterfactual_split(params, q, v, u)

    for k in range(10, 50):
        M = backend.mass_matrix(q[k])
        bias = backend.bias_forces(q[k], v[k])
        acc = np.linalg.solve(M, u[k] - bias)
        np.testing.assert_allclose(acc, drift[k] + control[k], rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_impact_detection_edge_cases(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    """Assert find_impact handles non-crossing edge cases."""
    # Short trace at static zero
    t = np.linspace(0.0, 0.1, 100)
    q = np.zeros((100, 2))
    v = np.zeros((100, 2))
    assert find_impact(t, q, v, inertials) is None


@pytest.mark.unit
def test_determinism(params: GolfModelParams) -> None:
    """Assert identical inputs produce identical outputs across runs."""
    prog = restrain_then_drive_program(60.0, 15.0, 5.0, 0.1)
    t1, q1, v1, u1 = rollout_program(params, prog)
    t2, q2, v2, u2 = rollout_program(params, prog)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(u1, u2)


@pytest.mark.unit
def test_parameter_sensitivity_case_contract() -> None:
    """Parameter cases cover each declared source of model uncertainty."""
    cases = build_parameter_cases()
    names = {case.name for case in cases}

    assert len(names) == len(cases)
    assert {
        "baseline",
        "arm_length_low",
        "arm_length_high",
        "arm_mass_low",
        "arm_mass_high",
        "club_length_low",
        "club_length_high",
        "clubhead_mass_low",
        "clubhead_mass_high",
        "plane_inclination_low",
        "plane_inclination_high",
        "joint_damping_low",
        "joint_damping_high",
    } == names


@pytest.mark.unit
def test_parameter_sensitivity_evaluates_ordering() -> None:
    """Each case reports comparable strategies and an explicit ordering test."""
    baseline = next(case for case in build_parameter_cases() if case.name == "baseline")
    result = evaluate_parameter_case(baseline)

    assert result["ordering"] == [
        "early_drive",
        "passive",
        "best_drive",
        "best_restrain",
    ]
    assert result["ordering_confirmed"] is True
    assert (
        result["strategies"]["early_drive"]["clubhead_speed_mps"]
        < result["strategies"]["passive"]["clubhead_speed_mps"]
    )
    assert result["strategies"]["best_drive"]["onset_s"] is not None
    assert result["strategies"]["best_restrain"]["wrist_restrain_nm"] in {
        5.0,
        10.0,
    }
