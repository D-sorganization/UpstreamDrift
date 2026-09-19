"""Unit tests for PF-04: Qualify Contact Modes and Native Pinocchio Force Feasibility (#10434).

Verifies:
1. Static weight balance in double support (total normal force = mg, COP inside support polygon).
2. Lift-off and landing transitions with hysteresis (no contact mode chattering).
3. Heel pivot support mode (heel grounded, toe lifted, COP on heel).
4. Moving / slipping foot (slip velocity detected, friction saturated at mu * fn).
5. Double support vs flight / unavailable support rejection (airborne golfer cannot claim ground balance).
6. Unphysical force and ankle torque rejection (rejects MN forces and kNm ankle torques).
7. Separate N and Nm residual budgets (independent linear force and moment tolerance checks).
8. Compliant vs allocated force consistency check (compares QP allocated forces to Hunt-Crossley constitutive forces).
9. Contact mode ambiguity and alternative schedule generation.
10. Sensitivity reporting (evaluates sensitivity to mass, marker offsets, geometry, and friction).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
)
from src.shared.python.motion_matching.contact_mode_qualifier import (
    ContactMode,
    ContactModeQualifier,
    FootContactState,
    HysteresisParameters,
    SensitivityReport,
    SupportModeReport,
    SupportState,
    verify_force_and_torque_limits,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def standard_qualifier() -> ContactModeQualifier:
    """Fixture providing a standard 6-sphere contact qualifier (3 spheres per foot: heel, mid, toe)."""
    # 6 contact spheres: [L_heel, L_mid, L_toe, R_heel, R_mid, R_toe]
    sphere_names = [
        "left_heel",
        "left_mid",
        "left_toe",
        "right_heel",
        "right_mid",
        "right_toe",
    ]
    sphere_radii = dict.fromkeys(sphere_names, 0.03)
    sphere_nominal_positions = {
        "left_heel": np.array([-0.05, 0.15, 0.03]),
        "left_mid": np.array([0.05, 0.15, 0.03]),
        "left_toe": np.array([0.15, 0.15, 0.03]),
        "right_heel": np.array([-0.05, -0.15, 0.03]),
        "right_mid": np.array([0.05, -0.15, 0.03]),
        "right_toe": np.array([0.15, -0.15, 0.03]),
    }
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    contact_params = ContactParameters(
        stiffness_n_m=1e5,
        dissipation_s_m=0.5,
        static_friction=0.8,
        dynamic_friction=0.7,
        viscous_friction=0.01,
        transition_velocity_m_s=0.05,
    )
    hysteresis = HysteresisParameters(
        engage_clearance_m=0.002,
        disengage_clearance_m=0.012,
        lift_off_velocity_m_s=0.05,
        touchdown_velocity_m_s=-0.05,
    )

    return ContactModeQualifier(
        sphere_names=sphere_names,
        sphere_radii=sphere_radii,
        nominal_positions=sphere_nominal_positions,
        ground=ground,
        contact_parameters=contact_params,
        hysteresis_parameters=hysteresis,
        mass_kg=75.0,
    )


def test_static_weight_balance(standard_qualifier: ContactModeQualifier) -> None:
    """Static double support balances total body weight with COP inside polygon."""
    qualifier = standard_qualifier
    mass_kg = 75.0
    gravity_n = mass_kg * 9.81  # ~735.75 N

    # Positions at ground level (z = 0.03 with radius = 0.03 -> contact point at z = 0.0)
    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    # Symmetrically distributed ground reaction forces summing to weight
    forces = {
        name: np.array([0.0, 0.0, gravity_n / len(qualifier.sphere_names)])
        for name in qualifier.sphere_names
    }

    report = qualifier.evaluate_support_mode(
        sphere_positions=positions,
        sphere_velocities=velocities,
        contact_forces=forces,
    )

    assert isinstance(report, SupportModeReport)
    assert report.support_state == SupportState.DOUBLE_SUPPORT
    assert report.left_foot.mode == ContactMode.FLAT
    assert report.right_foot.mode == ContactMode.FLAT
    assert math.isclose(report.total_normal_force_n, gravity_n, rel_tol=1e-3)
    assert report.is_weight_balanced is True
    assert report.inside_support_polygon is True
    assert report.cop_m is not None
    # COP in Y should be near 0 (centered between left and right foot)
    assert abs(report.cop_m[1]) < 0.02


def test_lift_off_and_landing_transitions(
    standard_qualifier: ContactModeQualifier,
) -> None:
    """Hysteresis prevents rapid chattering during lift-off and landing transitions."""
    qualifier = standard_qualifier
    foot = "right"

    # Step 1: In solid contact (clearance = 0)
    pos_ground = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    vel_zero = {name: np.zeros(3) for name in qualifier.sphere_names}
    r1 = qualifier.evaluate_support_mode(pos_ground, vel_zero)
    assert r1.right_foot.mode == ContactMode.FLAT

    # Step 2: Foot rising into the hysteresis band (clearance = 0.006 m, lifting velocity = 0.1 m/s)
    # Clearance is > engage_clearance (0.002) but < disengage_clearance (0.012)
    # Velocity upward triggers lift-off transition
    pos_rising = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    vel_rising = {name: np.zeros(3) for name in qualifier.sphere_names}
    for name in ["right_heel", "right_mid", "right_toe"]:
        pos_rising[name][2] += 0.006
        vel_rising[name][2] = 0.1  # lifting up

    r2 = qualifier.evaluate_support_mode(pos_rising, vel_rising, previous_state=r1)
    assert r2.right_foot.mode in (ContactMode.FLAT, ContactMode.FLIGHT)
    # Ambiguity score should be elevated inside transition band
    assert r2.ambiguity_score > 0.1

    # Step 3: Foot clearly above ground (clearance = 0.03 m)
    pos_airborne = {
        name: pos.copy() for name, pos in qualifier.nominal_positions.items()
    }
    for name in ["right_heel", "right_mid", "right_toe"]:
        pos_airborne[name][2] += 0.03

    r3 = qualifier.evaluate_support_mode(pos_airborne, vel_zero, previous_state=r2)
    assert r3.right_foot.mode == ContactMode.FLIGHT
    assert r3.support_state == SupportState.LEAD_ONLY  # only left foot grounded


def test_heel_pivot_support_mode(standard_qualifier: ContactModeQualifier) -> None:
    """Heel pivot mode: heel stays grounded while toe and midfoot lift."""
    qualifier = standard_qualifier

    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    # Left foot: heel grounded (z=0.03), midfoot lifted (z=0.05), toe lifted (z=0.08)
    positions["left_heel"][2] = 0.03
    positions["left_mid"][2] = 0.05
    positions["left_toe"][2] = 0.08

    # Right foot flat on ground
    report = qualifier.evaluate_support_mode(positions, velocities)

    assert report.left_foot.mode == ContactMode.HEEL_ONLY
    assert report.left_foot.active_spheres == ("left_heel",)
    assert report.right_foot.mode == ContactMode.FLAT


def test_moving_slipping_foot(standard_qualifier: ContactModeQualifier) -> None:
    """Foot translating with lateral velocity detects slip and evaluates friction cone saturation."""
    qualifier = standard_qualifier

    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    # Right foot sliding along X at 0.3 m/s
    for name in ["right_heel", "right_mid", "right_toe"]:
        velocities[name] = np.array([0.3, 0.0, 0.0])

    # Saturated friction forces opposing slip: fx = -mu * fn
    fn_per_sphere = 120.0
    mu = 0.8
    forces = {}
    for name in qualifier.sphere_names:
        if "right" in name:
            forces[name] = np.array([-mu * fn_per_sphere, 0.0, fn_per_sphere])
        else:
            forces[name] = np.array([0.0, 0.0, fn_per_sphere])

    report = qualifier.evaluate_support_mode(
        positions, velocities, contact_forces=forces
    )

    assert report.right_foot.is_slipping is True
    assert report.right_foot.slip_speed_m_s > 0.2
    assert math.isclose(report.right_foot.friction_saturation_ratio, 1.0, abs_tol=0.05)


def test_double_support_vs_flight_rejection(
    standard_qualifier: ContactModeQualifier,
) -> None:
    """Flight state where both feet are airborne rejects physical ground support."""
    qualifier = standard_qualifier

    # Both feet elevated well above ground (clearance 0.1 m)
    positions = {
        name: pos + np.array([0.0, 0.0, 0.1])
        for name, pos in qualifier.nominal_positions.items()
    }
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    report = qualifier.evaluate_support_mode(positions, velocities)

    assert report.support_state == SupportState.FLIGHT
    assert report.left_foot.mode == ContactMode.FLIGHT
    assert report.right_foot.mode == ContactMode.FLIGHT
    assert report.is_physically_supported is False
    assert report.total_normal_force_n == 0.0


def test_unphysical_force_and_torque_rejection() -> None:
    """Rejects unexplained mega-Newton loads and kNm ankle torques."""
    # 1. Normal human load: 1500 N, 80 Nm ankle torque -> PASS
    ok, diag = verify_force_and_torque_limits(
        contact_forces=np.array([0.0, 0.0, 1500.0]),
        joint_torques=np.array([80.0, 40.0, 20.0]),
        max_contact_force_n=5000.0,
        max_ankle_torque_nm=300.0,
    )
    assert ok is True
    assert diag["force_exceeded"] is False
    assert diag["torque_exceeded"] is False

    # 2. Mega-Newton load (1.2 MN) -> FAIL
    ok_mn, diag_mn = verify_force_and_torque_limits(
        contact_forces=np.array([0.0, 0.0, 1.2e6]),
        joint_torques=np.array([80.0, 40.0, 20.0]),
        max_contact_force_n=5000.0,
        max_ankle_torque_nm=300.0,
    )
    assert ok_mn is False
    assert diag_mn["force_exceeded"] is True
    assert diag_mn["max_force_n"] >= 1.2e6

    # 3. Kilonewton-metre ankle torque (1.5 kNm) -> FAIL
    ok_knm, diag_knm = verify_force_and_torque_limits(
        contact_forces=np.array([0.0, 0.0, 1500.0]),
        joint_torques=np.array([1500.0, 40.0, 20.0]),
        max_contact_force_n=5000.0,
        max_ankle_torque_nm=300.0,
    )
    assert ok_knm is False
    assert diag_knm["torque_exceeded"] is True
    assert diag_knm["max_torque_nm"] >= 1500.0


def test_separate_force_and_torque_residual_budgets(
    standard_qualifier: ContactModeQualifier,
) -> None:
    """Maintains independent residual budgets for linear force (N) and moment (Nm)."""
    qualifier = standard_qualifier

    # Dynamic equilibrium check with separate N and Nm tolerances
    force_residual_n = 0.5  # 0.5 N < 1.0 N budget
    torque_residual_nm = 0.04  # 0.04 Nm < 0.1 Nm budget

    is_valid, diag = qualifier.evaluate_residual_budgets(
        force_residual_n=force_residual_n,
        torque_residual_nm=torque_residual_nm,
        force_budget_n=1.0,
        torque_budget_nm=0.1,
    )
    assert is_valid is True
    assert diag["force_passed"] is True
    assert diag["torque_passed"] is True

    # Exceeding torque budget only
    is_valid_bad_torque, diag_bad_torque = qualifier.evaluate_residual_budgets(
        force_residual_n=0.5,
        torque_residual_nm=0.25,  # 0.25 Nm > 0.1 Nm budget
        force_budget_n=1.0,
        torque_budget_nm=0.1,
    )
    assert is_valid_bad_torque is False
    assert diag_bad_torque["force_passed"] is True
    assert diag_bad_torque["torque_passed"] is False


def test_compliant_vs_allocated_constitutive_check(
    standard_qualifier: ContactModeQualifier,
) -> None:
    """Compares allocated contact forces against constitutive Hunt-Crossley model."""
    qualifier = standard_qualifier

    # Sphere penetrated 2 mm into ground
    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    positions["left_heel"][2] = 0.028  # radius 0.03 -> 2 mm penetration
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    # Inferred/allocated force
    allocated_forces = {
        name: np.array([0.0, 0.0, 200.0]) if name == "left_heel" else np.zeros(3)
        for name in qualifier.sphere_names
    }

    # Evaluate constitutive check
    comp_report = qualifier.compare_allocated_to_constitutive(
        sphere_positions=positions,
        sphere_velocities=velocities,
        allocated_forces=allocated_forces,
    )
    assert "left_heel" in comp_report
    assert comp_report["left_heel"]["constitutive_normal_n"] > 0.0
    assert "discrepancy_n" in comp_report["left_heel"]


def test_contact_mode_ambiguity_and_alternative_schedules(
    standard_qualifier: ContactModeQualifier,
) -> None:
    """Generates alternative plausible schedules when contact states are ambiguous."""
    qualifier = standard_qualifier

    # Ambiguous state right on the edge of the hysteresis band
    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    # Set left toe clearance right at the threshold
    positions["left_toe"][2] += 0.007  # within [0.002, 0.012]

    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    report = qualifier.evaluate_support_mode(positions, velocities)
    assert report.has_ambiguity is True
    assert len(report.alternative_modes) > 0


def test_sensitivity_report(standard_qualifier: ContactModeQualifier) -> None:
    """Quantifies sensitivity of support metrics to variations in mass, offsets, geometry, and friction."""
    qualifier = standard_qualifier

    positions = {name: pos.copy() for name, pos in qualifier.nominal_positions.items()}
    velocities = {name: np.zeros(3) for name in qualifier.sphere_names}

    sensitivity = qualifier.compute_sensitivity_report(
        nominal_positions=positions,
        nominal_velocities=velocities,
        mass_variation_pct=10.0,
        geometry_variation_m=0.005,
        friction_variation=0.2,
    )

    assert isinstance(sensitivity, SensitivityReport)
    assert "mass_sensitivity" in sensitivity.metrics
    assert "geometry_sensitivity" in sensitivity.metrics
    assert "friction_sensitivity" in sensitivity.metrics
