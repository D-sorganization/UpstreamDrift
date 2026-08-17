from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _closed_case(activation: str = "coupled"):
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    grip = DistributedGripConfig(station_count_per_hand=3)
    case = ShaftIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=0.0005,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.05,
        engine="mujoco",
        grip=grip,
        shaft=ArticulatedShaftConfig(activation=activation),  # type: ignore[arg-type]
    )
    return model, case


def test_shaft_forward_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="time_steps_s"):
        ShaftForwardConfig(time_steps_s=(0.0005, 0.001))


def test_rigid_shaft_branch_reduces_exactly_to_distributed_grip_trace() -> None:
    model, case = _closed_case("rigid")
    forward = ShaftForwardConfig(duration_s=0.005, time_steps_s=(0.0005, 0.00025))
    shaft_trace = integrate_articulated_shaft(model, case, forward)
    reference_case = DistributedIntegrationCase(
        q=case.q,
        qd=case.qd,
        grip_span_m=case.grip_span_m,
        hand_contact_local_x_m=case.hand_contact_local_x_m,
        time_step_s=case.time_step_s,
        initial_club_displacement_m=case.initial_club_displacement_m,
        initial_club_velocity_m_s=case.initial_club_velocity_m_s,
        engine=case.engine,
        grip=case.grip,
    )
    reference = integrate_distributed_grip(
        model,
        reference_case,
        DistributedForwardConfig(
            duration_s=forward.duration_s,
            time_steps_s=forward.time_steps_s,
        ),
    )

    np.testing.assert_array_equal(shaft_trace["q"], reference["q"])
    np.testing.assert_array_equal(shaft_trace["qd"], reference["qd"])
    np.testing.assert_array_equal(
        shaft_trace["force_couple_vector_nm"], reference["force_couple_vector_nm"]
    )
    assert shaft_trace["elastic_coordinates"].shape[1] == 0


def test_coupled_shaft_activates_bending_and_torsion_with_closed_ledgers() -> None:
    model, case = _closed_case()
    trace = integrate_articulated_shaft(
        model,
        case,
        ShaftForwardConfig(duration_s=0.01, time_steps_s=(0.0005, 0.00025)),
    )

    assert trace["elastic_coordinates"].shape[1] == 3
    assert np.max(np.abs(trace["elastic_coordinates"][:, :2])) > 1.0e-10
    assert np.max(np.abs(trace["elastic_coordinates"][:, 2])) > 1.0e-12
    assert np.max(trace["shaft_strain_energy_j"]) > 0.0
    assert np.max(np.abs(trace["shaft_power_residual_w"])) < 1.0e-10
    assert np.max(trace["small_deflection_ratio"]) < case.shaft.small_deflection_limit
    assert np.max(np.abs(trace["twist_angle_rad"])) < case.shaft.twist_limit_rad
    assert np.all(np.isfinite(trace["work_energy_residual_j"]))


def test_bending_and_torsion_killswitches_are_distinct() -> None:
    model, bending_case = _closed_case("bending")
    _, torsion_case = _closed_case("torsion")
    config = ShaftForwardConfig(duration_s=0.004, time_steps_s=(0.0005, 0.00025))
    bending = integrate_articulated_shaft(model, bending_case, config)
    torsion = integrate_articulated_shaft(model, torsion_case, config)

    assert bending["elastic_coordinates"].shape[1] == 2
    assert torsion["elastic_coordinates"].shape[1] == 1
    assert np.max(np.abs(bending["tip_bending_m"])) > 0.0
    assert np.max(np.abs(torsion["twist_angle_rad"])) > 0.0
