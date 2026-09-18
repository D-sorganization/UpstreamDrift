"""Unit and acceptance tests for contact modes and Pinocchio force feasibility (PF-04, #10434).

Verifies:
1. Static weight balance: normal force equals body weight, COP inside polygon, zero root residual.
2. Lift-off and landing with hysteresis: clean transitions without chattering or force spikes.
3. Heel pivot: toe contact maintained, heel off ground, COP under forefoot, ankle torque bounded.
4. Moving / slipping foot: tangential force bounded by Coulomb cone, slip flagged.
5. Double support: load sharing across feet, COP contained in composite polygon.
6. Flight: feet off ground, ground forces strictly zero, arbitrary supported forces rejected.
7. Separate N and Nm residual budgets: independent gating prevents masking.
8. Rejection of unexplained MN loads or kNm ankle torques.
9. Constitutive compliant contact comparison: checks inverse dynamics forces against Hunt-Crossley.
10. Parameter sensitivity report: computes sensitivities to mass, offsets, geometry, and friction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_force_feasibility import (
    ContactFeasibilityConfig,
    FeasibilityAuditResult,
    ForceFeasibilityError,
    audit_contact_force_feasibility,
    compute_contact_sensitivity,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.contact_modes import (
    ContactHysteresisSettings,
    ContactModeSequence,
    ContactState,
    FootSupportMode,
    GlobalSupportMode,
    evaluate_support_geometry,
    infer_contact_modes,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _sample_ground() -> GroundPlane:
    return GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


def _default_contact_params() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=50000.0,
        dissipation_s_m=1.0,
        static_friction=0.9,
        dynamic_friction=0.8,
        viscous_friction=0.0,
        transition_velocity_m_s=0.05,
    )


def _foot_positions_address() -> dict[str, np.ndarray]:
    """Standard 4-sphere foot layout at neutral address stance."""
    return {
        "heel_l": np.array([-0.1, 0.15, 0.035]),
        "forefoot_l": np.array([0.15, 0.15, 0.03]),
        "heel_r": np.array([-0.1, -0.15, 0.035]),
        "forefoot_r": np.array([0.15, -0.15, 0.03]),
    }


def _foot_radii() -> dict[str, float]:
    return {
        "heel_l": 0.035,
        "forefoot_l": 0.03,
        "heel_r": 0.035,
        "forefoot_r": 0.03,
    }


# ---------------------------------------------------------------------------
# Acceptance Tests
# ---------------------------------------------------------------------------


def test_static_weight_balance() -> None:
    """Acceptance 1: Total vertical force equals body weight, COP inside polygon, zero root residual."""
    ground = _sample_ground()
    mass_kg = 75.0
    bw_n = mass_kg * 9.81

    # Equal distribution across 4 spheres: bw_n / 4 each
    f_per_sphere = bw_n / 4.0
    f_ground = np.array(
        [
            0.0,
            0.0,
            f_per_sphere,  # heel_l
            0.0,
            0.0,
            f_per_sphere,  # forefoot_l
            0.0,
            0.0,
            f_per_sphere,  # heel_r
            0.0,
            0.0,
            f_per_sphere,  # forefoot_r
        ]
    )
    tau_actuated = np.zeros(35)
    delta_tau_root = np.zeros(6)  # exact equilibrium, zero root slack

    positions = _foot_positions_address()
    audit = audit_contact_force_feasibility(
        tau_actuated=tau_actuated,
        f_ground=f_ground,
        delta_tau_root=delta_tau_root,
        body_mass_kg=mass_kg,
        ankle_indices=(10, 11, 12, 13),
        contact_positions_m=positions,
        ground=ground,
    )

    assert audit.is_feasible, (
        f"Expected feasible static balance: {audit.failure_reasons}"
    )
    assert audit.root_force_residual_n == pytest.approx(0.0)
    assert audit.root_torque_residual_nm == pytest.approx(0.0)
    assert audit.cop_contained
    assert audit.friction_cone_satisfied
    assert audit.max_grf_bw_ratio == pytest.approx(0.25, rel=1e-2)


def test_liftoff_landing_hysteresis() -> None:
    """Acceptance 2: Trajectory transitioning through boundary has hysteresis and no chattering."""
    ground = _sample_ground()
    radii = _foot_radii()

    # Trajectory of vertical motion oscillating around threshold [0.004, 0.012]
    # Engage: 0.005, Release: 0.015
    n_frames = 20
    times = np.linspace(0.0, 1.0, n_frames)

    # Foot starts engaged, lifts off slowly, then descends back
    # Bottom heights: 0.001 -> 0.010 -> 0.020 -> 0.008 -> 0.002
    bottom_heights = np.concatenate(
        [
            np.linspace(0.001, 0.008, 5),  # in gap, stays engaged
            np.linspace(0.012, 0.025, 5),  # releases above 0.015
            np.linspace(
                0.022, 0.008, 5
            ),  # in gap descending, stays open until <= 0.005
            np.linspace(0.004, 0.001, 5),  # re-engages below 0.005
        ]
    )

    pos_dict = {}
    vel_dict = {}
    for name in radii:
        r = radii[name]
        zs = bottom_heights + r
        pos = np.zeros((n_frames, 3))
        pos[:, 2] = zs
        pos_dict[name] = pos

        # Rates via finite difference
        vel = np.zeros((n_frames, 3))
        vel[1:, 2] = np.diff(zs) / (times[1] - times[0])
        vel_dict[name] = vel

    settings = ContactHysteresisSettings(
        engage_height_m=0.005,
        release_height_m=0.015,
        engage_velocity_m_s=-0.10,
        release_velocity_m_s=0.10,
    )
    seq = infer_contact_modes(
        pos_dict, vel_dict, ground, times, radii, settings=settings
    )

    modes = [f.global_mode for f in seq.frames]
    # Frame 0-4: stays DOUBLE_SUPPORT (hysteresis prevents early release at 0.008)
    assert modes[0] == GlobalSupportMode.DOUBLE_SUPPORT
    assert modes[4] == GlobalSupportMode.DOUBLE_SUPPORT

    # Frame 8-11: FLIGHT (released above 0.015)
    assert modes[8] == GlobalSupportMode.FLIGHT

    # Frame 12-14: stays FLIGHT in descending gap (hysteresis prevents early engage at 0.008)
    assert modes[13] == GlobalSupportMode.FLIGHT

    # Frame 16+: re-engaged to DOUBLE_SUPPORT (below 0.005)
    assert modes[18] == GlobalSupportMode.DOUBLE_SUPPORT


def test_heel_pivot_mode() -> None:
    """Acceptance 3: Heel elevated while forefoot in contact classified as heel pivot with bounded ankle torque."""
    ground = _sample_ground()
    radii = _foot_radii()
    times = [0.0]

    # Right foot: heel elevated (h_bottom = 0.04 > 0.015), forefoot planted (h_bottom = 0.001 < 0.005)
    # Left foot: planted
    pos_dict = {
        "heel_l": np.array([[0.0, 0.15, radii["heel_l"] + 0.001]]),
        "forefoot_l": np.array([[0.2, 0.15, radii["forefoot_l"] + 0.001]]),
        "heel_r": np.array([[0.0, -0.15, radii["heel_r"] + 0.04]]),
        "forefoot_r": np.array([[0.2, -0.15, radii["forefoot_r"] + 0.001]]),
    }
    vel_dict = {k: np.zeros((1, 3)) for k in radii}

    seq = infer_contact_modes(pos_dict, vel_dict, ground, times, radii)
    assert seq.frames[0].global_mode == GlobalSupportMode.RIGHT_HEEL_PIVOT
    assert seq.frames[0].right_foot.support_mode == FootSupportMode.TOE_ONLY

    # Audit forces during heel pivot: ankle torque must remain physiological (<= 350 Nm)
    tau_actuated = np.zeros(35)
    tau_actuated[10] = 120.0  # Ankle plantarflexion 120 Nm
    f_ground = np.array(
        [
            0.0,
            0.0,
            300.0,  # heel_l
            0.0,
            0.0,
            250.0,  # forefoot_l
            0.0,
            0.0,
            0.0,  # heel_r (elevated, 0 force)
            0.0,
            0.0,
            185.0,  # forefoot_r (pivot support)
        ]
    )
    delta_tau_root = np.zeros(6)

    positions = {k: pos_dict[k][0] for k in pos_dict}
    audit = audit_contact_force_feasibility(
        tau_actuated=tau_actuated,
        f_ground=f_ground,
        delta_tau_root=delta_tau_root,
        body_mass_kg=75.0,
        ankle_indices=[10],
        contact_positions_m=positions,
        ground=ground,
    )
    assert audit.is_feasible
    assert audit.max_ankle_torque_nm == pytest.approx(120.0)


def test_moving_slipping_foot() -> None:
    """Acceptance 4: Tangential foot velocity exceeding slip threshold flags slipping state and Coulomb check."""
    ground = _sample_ground()
    radii = _foot_radii()
    times = [0.0]

    # Feet planted but right foot sliding tangentially at 0.1 m/s (> slip_velocity 0.02)
    pos_dict = {name: np.array([[0.0, 0.0, radii[name] + 0.001]]) for name in radii}
    vel_dict = {
        "heel_l": np.zeros((1, 3)),
        "forefoot_l": np.zeros((1, 3)),
        "heel_r": np.array([[0.1, 0.0, 0.0]]),
        "forefoot_r": np.array([[0.1, 0.0, 0.0]]),
    }

    seq = infer_contact_modes(pos_dict, vel_dict, ground, times, radii)
    assert seq.frames[0].right_foot.support_mode == FootSupportMode.SLIPPING

    # Evaluate support geometry and friction cone
    positions = {k: pos_dict[k][0] for k in pos_dict}
    # Apply friction force exceeding mu * fn (mu=0.8, fn=200N -> limit=160N, apply ft=180N)
    forces = {
        "heel_l": np.array([0.0, 0.0, 200.0]),
        "forefoot_l": np.array([0.0, 0.0, 200.0]),
        "heel_r": np.array([180.0, 0.0, 200.0]),  # 180 > 0.8 * 200 = 160
        "forefoot_r": np.array([120.0, 0.0, 200.0]),  # 120 <= 160
    }
    geo = evaluate_support_geometry(positions, forces, ground, mu_friction=0.8)
    assert "heel_r" in geo.friction_cone_violations
    assert geo.friction_cone_violations["heel_r"] == pytest.approx(20.0)


def test_double_support_load_sharing() -> None:
    """Acceptance 5: Load sharing across left/right feet with COP contained in composite hull."""
    ground = _sample_ground()
    positions = _foot_positions_address()

    # 65% weight on lead (left) foot, 35% on trail (right) foot
    total_f = 750.0
    forces = {
        "heel_l": np.array([0.0, 0.0, 0.35 * total_f]),
        "forefoot_l": np.array([0.0, 0.0, 0.30 * total_f]),
        "heel_r": np.array([0.0, 0.0, 0.20 * total_f]),
        "forefoot_r": np.array([0.0, 0.0, 0.15 * total_f]),
    }
    geo = evaluate_support_geometry(positions, forces, ground)
    assert geo.inside_support_polygon
    assert geo.total_normal_force_n == pytest.approx(total_f)
    assert geo.centre_of_pressure_m is not None
    # COP Y coordinate should be shifted towards positive Y (left foot is at +0.15)
    assert geo.centre_of_pressure_m[1] > 0.0


def test_flight_rejects_arbitrary_support() -> None:
    """Acceptance 6: Feet elevated in flight reject arbitrary ground force allocation."""
    ground = _sample_ground()
    positions = {
        k: v + np.array([0.0, 0.0, 0.1])  # 10 cm in the air
        for k, v in _foot_positions_address().items()
    }

    # Solver attempts to allocate 500 N of ground force while feet are in flight
    f_ground = np.full(12, 500.0 / 4.0)
    tau_actuated = np.zeros(35)
    delta_tau_root = np.zeros(6)

    # Constitutive model predicts 0 force because penetration is 0
    params = _default_contact_params()
    constitutive_forces = {
        name: sphere_ground_contact(
            positions[name], np.zeros(3), _foot_radii()[name], ground, params
        )
        for name in positions
    }

    audit = audit_contact_force_feasibility(
        tau_actuated=tau_actuated,
        f_ground=f_ground,
        delta_tau_root=delta_tau_root,
        body_mass_kg=75.0,
        contact_positions_m=positions,
        ground=ground,
        constitutive_forces=constitutive_forces,
    )

    assert not audit.is_feasible
    assert any("in flight" in reason for reason in audit.failure_reasons)


def test_separate_n_and_nm_residual_budgets() -> None:
    """Acceptance 7: Force (N) and torque (Nm) residuals are budgeted independently."""
    mass_kg = 75.0
    f_ground = np.full(12, mass_kg * 9.81 / 4.0)
    tau_actuated = np.zeros(35)

    config = ContactFeasibilityConfig(
        force_residual_budget_n=5.0,
        torque_residual_budget_nm=1.0,
    )

    # Case A: Force violation (10 N > 5 N), torque clean (0.2 Nm <= 1.0 Nm)
    root_case_a = np.array([10.0, 0.0, 0.0, 0.1, 0.1, 0.0])
    audit_a = audit_contact_force_feasibility(
        tau_actuated=tau_actuated,
        f_ground=f_ground,
        delta_tau_root=root_case_a,
        body_mass_kg=mass_kg,
        config=config,
    )
    assert not audit_a.is_feasible
    assert any("force residual" in r for r in audit_a.failure_reasons)
    assert not any("torque residual" in r for r in audit_a.failure_reasons)

    # Case B: Torque violation (2.5 Nm > 1.0 Nm), force clean (1.0 N <= 5.0 N)
    root_case_b = np.array([0.5, 0.5, 0.0, 2.5, 0.0, 0.0])
    audit_b = audit_contact_force_feasibility(
        tau_actuated=tau_actuated,
        f_ground=f_ground,
        delta_tau_root=root_case_b,
        body_mass_kg=mass_kg,
        config=config,
    )
    assert not audit_b.is_feasible
    assert any("torque residual" in r for r in audit_b.failure_reasons)
    assert not any("force residual" in r for r in audit_b.failure_reasons)


def test_rejection_of_unexplained_mn_or_knm_loads() -> None:
    """Acceptance 8: Meganewton loads and kilonewton-metre ankle torques fail immediately."""
    mass_kg = 75.0
    delta_tau_root = np.zeros(6)

    # Case A: Meganewton force spike (1.5 MN)
    f_mn = np.zeros(12)
    f_mn[2] = 1.5e6  # 1.5 Meganewtons
    audit_mn = audit_contact_force_feasibility(
        tau_actuated=np.zeros(35),
        f_ground=f_mn,
        delta_tau_root=delta_tau_root,
        body_mass_kg=mass_kg,
    )
    assert not audit_mn.is_feasible
    assert any("meganewton load" in r for r in audit_mn.failure_reasons)

    # Case B: kilonewton-metre ankle torque (1200 Nm)
    tau_knm = np.zeros(35)
    tau_knm[12] = 1200.0  # 1.2 kNm
    f_clean = np.full(12, mass_kg * 9.81 / 4.0)
    audit_knm = audit_contact_force_feasibility(
        tau_actuated=tau_knm,
        f_ground=f_clean,
        delta_tau_root=delta_tau_root,
        body_mass_kg=mass_kg,
        ankle_indices=[12],
    )
    assert not audit_knm.is_feasible
    assert any("kilonewton-metre ankle torque" in r for r in audit_knm.failure_reasons)

    # Verify raise_on_failure exception
    with pytest.raises(ForceFeasibilityError):
        audit_contact_force_feasibility(
            tau_actuated=tau_knm,
            f_ground=f_clean,
            delta_tau_root=delta_tau_root,
            body_mass_kg=mass_kg,
            ankle_indices=[12],
            raise_on_failure=True,
        )


def test_constitutive_compliant_force_comparison() -> None:
    """Acceptance 9: Compliant physics checks allocated forces against Hunt-Crossley constitutive forces."""
    ground = _sample_ground()
    params = _default_contact_params()
    positions = _foot_positions_address()
    radii = _foot_radii()

    # Penetrate foot by 2 mm into ground
    positions_penetrated = {
        name: np.array([pos[0], pos[1], radii[name] - 0.002])
        for name, pos in positions.items()
    }
    velocities = {name: np.zeros(3) for name in positions}

    constitutive = {
        name: sphere_ground_contact(
            positions_penetrated[name],
            velocities[name],
            radii[name],
            ground,
            params,
        )
        for name in positions
    }

    # Match constitutive forces: stiffness 50000 * 0.002 = 100 N per sphere
    f_matched = np.array([0.0, 0.0, 100.0] * 4)
    audit_matched = audit_contact_force_feasibility(
        tau_actuated=np.zeros(35),
        f_ground=f_matched,
        delta_tau_root=np.zeros(6),
        body_mass_kg=40.0,  # 4 * 100 N = 400 N ~ 40 kg
        contact_positions_m=positions_penetrated,
        ground=ground,
        constitutive_forces=constitutive,
    )
    assert audit_matched.is_feasible
    assert audit_matched.constitutive_residual_n < 1.0

    # Divergent forces: allocate 300 N when constitutive is 100 N (200 N discrepancy > 50 N tol)
    f_divergent = np.array([0.0, 0.0, 300.0] * 4)
    audit_divergent = audit_contact_force_feasibility(
        tau_actuated=np.zeros(35),
        f_ground=f_divergent,
        delta_tau_root=np.zeros(6),
        body_mass_kg=40.0,
        contact_positions_m=positions_penetrated,
        ground=ground,
        constitutive_forces=constitutive,
    )
    assert not audit_divergent.is_feasible
    assert any(
        "diverge from constitutive" in r for r in audit_divergent.failure_reasons
    )


def test_contact_sensitivity_report() -> None:
    """Acceptance 10: Sensitivity report evaluates variations across mass, offset, height, and friction."""

    def mock_allocator_solve(
        mass_kg: float,
        ground_height_m: float,
        mu_friction: float,
        marker_offset_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Realistic linear scaling: vertical GRF balances mass * g,
        # ground penetration changes with height, friction changes slip capacity
        total_fz = (
            mass_kg * 9.81 + (ground_height_m * 1000.0) + (marker_offset_m * 500.0)
        )
        f_ground = np.array([0.0, 0.0, total_fz / 4.0] * 4)
        return np.zeros(35), f_ground, np.zeros(6)

    report = compute_contact_sensitivity(
        mock_allocator_solve,
        base_mass_kg=75.0,
        base_ground_height_m=0.0,
        base_friction=0.8,
    )

    assert report.base_total_force_n == pytest.approx(75.0 * 9.81, rel=1e-3)
    assert report.mass_sensitivity_n_per_kg == pytest.approx(9.81, rel=1e-2)
    assert report.ground_height_sensitivity_n_per_m == pytest.approx(1000.0, rel=1e-2)
    assert report.marker_offset_sensitivity_n_per_m == pytest.approx(500.0, rel=1e-2)
    assert report.friction_sensitivity_n_per_unit == pytest.approx(0.0)
