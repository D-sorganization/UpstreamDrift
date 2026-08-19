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
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

pytestmark = pytest.mark.scientific


def test_center_of_pressure_reversal_changes_only_reported_transport_moment() -> None:
    model = build_subject_scaled_model(default_synthetic_profiles()[0])[0]
    ground = ArticulatedGroundConfig()
    case = GroundIntegrationCase(
        q=np.zeros(model.nq),
        qd=np.linspace(-0.01, 0.01, model.nq),
        grip_span_m=0.18,
        hand_contact_local_x_m=0.055,
        time_step_s=0.000125,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.02,
        initial_base_displacement=(-0.001, 0.001, 0.002),
        initial_base_velocity=(-0.01, 0.01, 0.02),
        engine="mujoco",
        grip=DistributedGripConfig(station_count_per_hand=3),
        shaft=ArticulatedShaftConfig(),
        ground=ground,
    )
    config = GroundForwardConfig(duration_s=0.001, time_steps_s=(0.00025, 0.000125))
    baseline = integrate_articulated_ground(model, case, config)
    reversed_reference = integrate_articulated_ground(
        model,
        replace(
            case,
            ground=replace(
                ground,
                center_of_pressure_xz_m=tuple(
                    -value for value in ground.center_of_pressure_xz_m
                ),
            ),
        ),
        config,
    )

    for name in (
        "q",
        "qd",
        "elastic_coordinates",
        "base_coordinates",
        "ground_force_n",
        "ground_intrinsic_free_moment_nm",
        "total_energy_j",
    ):
        np.testing.assert_array_equal(baseline[name], reversed_reference[name])
    assert np.any(
        baseline["ground_transported_moment_nm"]
        != reversed_reference["ground_transported_moment_nm"]
    )
