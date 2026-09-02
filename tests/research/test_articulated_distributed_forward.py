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
from scripts.research.proximal_distal_energy.articulated_slack_contact import (
    AttachmentLawConfig,
    AttachmentLawKind,
)
from scripts.research.proximal_distal_energy.articulated_slack_forward import (
    ArticulatedSlackForwardConfig,
    SlackIntegrationCase,
    integrate_articulated_slack,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _case_inputs() -> tuple[object, dict[str, object], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span_m


def test_distributed_forward_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        DistributedForwardConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="time_steps_s"):
        DistributedForwardConfig(time_steps_s=(0.0005, 0.001))


def test_one_fiber_forward_reduces_to_point_tension_trace() -> None:
    model, metadata, q, grip_span_m = _case_inputs()
    hand_x = float(metadata["hand_contact_local_x_m"])
    distributed_config = DistributedForwardConfig(
        duration_s=0.002,
        time_steps_s=(0.001, 0.0005),
    )
    distributed = integrate_distributed_grip(
        model,
        DistributedIntegrationCase(
            q=q,
            qd=np.zeros(model.nq),
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_x,
            time_step_s=0.0005,
            initial_club_displacement_m=0.001,
            initial_club_velocity_m_s=0.05,
            engine="mujoco",
            grip=DistributedGripConfig(
                station_count_per_hand=1,
                station_width_m=0.0,
            ),
        ),
        distributed_config,
    )
    point = integrate_articulated_slack(
        model,
        SlackIntegrationCase(
            q=q,
            qd=np.zeros(model.nq),
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_x,
            time_step_s=0.0005,
            initial_club_displacement_m=0.001,
            initial_club_velocity_m_s=0.05,
            engine="mujoco",
            law=AttachmentLawConfig(
                kind=AttachmentLawKind.TENSION_ONLY,
                stiffness=1800.0,
                damping=18.0,
            ),
        ),
        ArticulatedSlackForwardConfig(
            duration_s=0.002,
            time_steps_s=(0.001, 0.0005),
        ),
    )

    assert np.allclose(distributed["q"], point["q"], atol=1.0e-12)
    assert np.allclose(
        distributed["maximum_station_force_n"],
        point["maximum_contact_force_n"],
        atol=1.0e-12,
    )


def test_five_fiber_trace_is_finite_passive_and_power_closed() -> None:
    model, metadata, q, grip_span_m = _case_inputs()
    config = DistributedForwardConfig(
        duration_s=0.002,
        time_steps_s=(0.001, 0.0005),
    )
    trace = integrate_distributed_grip(
        model,
        DistributedIntegrationCase(
            q=q,
            qd=np.zeros(model.nq),
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            time_step_s=0.0005,
            initial_club_displacement_m=0.001,
            initial_club_velocity_m_s=0.05,
            engine="mujoco",
            grip=DistributedGripConfig(station_count_per_hand=5),
        ),
        config,
    )

    assert np.all(np.isfinite(trace["q"]))
    assert np.max(trace["dissipation_power_w"]) <= 0.0
    assert np.max(np.abs(trace["virtual_power_residual_w"])) <= 1.0e-10
    assert np.all(trace["station_load_concentration"] >= 0.0)
    assert np.all(trace["station_load_concentration"] <= 1.0)
    assert trace["station_signed_gap_m"].shape == trace["station_active"].shape
    assert np.array_equal(trace["station_signed_gap_m"] > 0.0, trace["station_active"])
