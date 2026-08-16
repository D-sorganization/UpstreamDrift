from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
)
from scripts.research.proximal_distal_energy.articulated_ground_forward import (
    GroundForwardConfig,
    GroundIntegrationCase,
    integrate_articulated_ground,
    solve_conditional_base_equilibrium,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
)
from scripts.research.proximal_distal_energy.articulated_shaft_forward import (
    ShaftForwardConfig,
    ShaftIntegrationCase,
    integrate_articulated_shaft,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)


def _case(activation: str = "coupled"):
    model = build_subject_scaled_model(default_synthetic_profiles()[0])[0]
    q = np.zeros(model.nq)
    qd = np.linspace(-0.01, 0.01, model.nq)
    case = GroundIntegrationCase(
        q=q,
        qd=qd,
        grip_span_m=0.18,
        hand_contact_local_x_m=0.055,
        time_step_s=0.000125,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.02,
        initial_base_displacement=(
            (0.0, 0.0, 0.0) if activation == "fixed" else (-0.001, 0.001, 0.002)
        ),
        initial_base_velocity=(
            (0.0, 0.0, 0.0) if activation == "fixed" else (-0.01, 0.01, 0.02)
        ),
        engine="mujoco",
        grip=DistributedGripConfig(station_count_per_hand=3),
        shaft=ArticulatedShaftConfig(),
        ground=ArticulatedGroundConfig(activation=activation),  # type: ignore[arg-type]
    )
    config = GroundForwardConfig(duration_s=0.001, time_steps_s=(0.00025, 0.000125))
    return model, case, config


def test_ground_forward_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="decreasing positive divisors"):
        GroundForwardConfig(duration_s=0.001, time_steps_s=(0.000125, 0.00025))
    model, case, config = _case("translation")
    with pytest.raises(ValueError, match="inactive base coordinates"):
        integrate_articulated_ground(
            model,
            replace(case, initial_base_displacement=(0.0, 0.0, 0.01)),
            config,
        )


def test_fixed_ground_delegates_exactly_to_shaft_forward() -> None:
    model, case, config = _case("fixed")
    ground_trace = integrate_articulated_ground(model, case, config)
    shaft_trace = integrate_articulated_shaft(
        model,
        ShaftIntegrationCase(
            q=case.q,
            qd=case.qd,
            grip_span_m=case.grip_span_m,
            hand_contact_local_x_m=case.hand_contact_local_x_m,
            time_step_s=case.time_step_s,
            initial_club_displacement_m=case.initial_club_displacement_m,
            initial_club_velocity_m_s=case.initial_club_velocity_m_s,
            engine=case.engine,
            grip=case.grip,
            shaft=case.shaft,
        ),
        ShaftForwardConfig(
            duration_s=config.duration_s,
            time_steps_s=config.time_steps_s,
            normalized_energy_residual_tolerance=(
                config.normalized_energy_residual_tolerance
            ),
        ),
    )
    for key in ("q", "qd", "elastic_coordinates", "total_energy_j"):
        np.testing.assert_array_equal(ground_trace[key], shaft_trace[key])
    assert ground_trace["base_coordinates"].shape[1] == 0


def test_coupled_ground_advances_with_closed_power_and_finite_energy() -> None:
    model, case, config = _case("coupled")
    trace = integrate_articulated_ground(model, case, config)
    assert np.max(np.linalg.norm(trace["ground_force_n"], axis=1)) > 0.0
    assert np.max(np.abs(trace["ground_intrinsic_free_moment_nm"])) > 0.0
    assert np.max(trace["ground_power_residual_w"]) < 1.0e-10
    assert np.max(trace["virtual_power_residual_w"]) < 1.0e-10
    assert np.all(np.isfinite(trace["work_energy_residual_j"]))
    assert np.max(np.linalg.norm(trace["base_translation_m"], axis=1)) < 0.05
    assert np.max(np.abs(trace["base_pitch_rad"])) < np.deg2rad(10.0)


def test_translation_and_free_moment_killswitches_are_distinct() -> None:
    model, coupled, config = _case("coupled")
    translation = replace(
        coupled,
        ground=ArticulatedGroundConfig(activation="translation"),
        initial_base_displacement=(-0.001, 0.001, 0.0),
        initial_base_velocity=(-0.01, 0.01, 0.0),
    )
    moment = replace(
        coupled,
        ground=ArticulatedGroundConfig(activation="free_moment"),
        initial_base_displacement=(0.0, 0.0, 0.002),
        initial_base_velocity=(0.0, 0.0, 0.02),
    )
    translation_trace = integrate_articulated_ground(model, translation, config)
    moment_trace = integrate_articulated_ground(model, moment, config)
    assert np.max(np.linalg.norm(translation_trace["ground_force_n"], axis=1)) > 0.0
    assert np.all(translation_trace["ground_intrinsic_free_moment_nm"] == 0.0)
    assert np.all(moment_trace["ground_force_n"] == 0.0)
    assert np.max(np.abs(moment_trace["ground_intrinsic_free_moment_nm"])) > 0.0


def test_conditional_base_equilibrium_closes_ground_grip_and_gravity() -> None:
    model, case, _ = _case("coupled")
    equilibrium = solve_conditional_base_equilibrium(
        model,
        case.q,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        grip_config=case.grip,
        shaft_config=case.shaft,
        ground_config=case.ground,
    )
    assert equilibrium.residual_norm < 1.0e-6
    assert equilibrium.iteration_count <= 40
    assert (equilibrium.active_station_count == 0) == (
        equilibrium.maximum_station_force_n == 0.0
    )
    assert np.linalg.norm(equilibrium.base_coordinates[:2]) < 0.05
    assert abs(equilibrium.base_coordinates[2]) < np.deg2rad(10.0)
