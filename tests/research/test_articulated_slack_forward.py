from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

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
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_slack_forward_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        ArticulatedSlackForwardConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="time_steps_s"):
        ArticulatedSlackForwardConfig(time_steps_s=(0.0005, 0.001))


def test_open_dead_zone_trace_is_finite_and_passive() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    config = ArticulatedSlackForwardConfig(
        duration_s=0.002,
        time_steps_s=(0.001, 0.0005),
    )
    case = SlackIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=0.0005,
        initial_club_displacement_m=5.0e-4,
        initial_club_velocity_m_s=0.0,
        engine="mujoco",
        law=AttachmentLawConfig(
            kind=AttachmentLawKind.DEAD_ZONE_TENSION,
            slack_distance_m=1.0e-3,
        ),
    )
    trace = integrate_articulated_slack(model, case, config)

    assert np.all(np.isfinite(trace["q"]))
    assert trace["active_interface_count"][0] == 0
    assert np.max(trace["dissipation_power_w"]) <= 0.0
    assert np.max(np.abs(trace["virtual_power_residual_w"])) <= 1.0e-10


def test_committed_slack_atlas_is_complete_and_bounded() -> None:
    record = json.loads(
        (DATA_DIR / "articulated_slack_atlas.json").read_text(encoding="utf-8")
    )

    assert record["schema_version"] == "articulated-slack-atlas/v1"
    assert record["design"]["trajectory_count"] == 1944
    assert record["results"]["opening_cell_count"] > 0
    assert record["results"]["reattachment_cell_count"] > 0
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["claim_boundary"]["human_transfer"] == "untested"


def test_natural_states_and_event_probes_remain_distinct() -> None:
    with np.load(DATA_DIR / "articulated_slack_atlas.npz") as arrays:
        names = arrays["condition_names"].astype(str)
        probes = np.asarray(["event_probe" in name for name in names])
        natural = ~probes

        assert np.count_nonzero(arrays["opening_observed"][:, natural]) == 0
        assert np.count_nonzero(arrays["reattachment_observed"][:, natural]) == 0
        assert np.count_nonzero(arrays["opening_observed"][:, probes]) > 0
        assert np.count_nonzero(arrays["reattachment_observed"][:, probes]) > 0
        assert np.all(arrays["active_set_parity"])
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key
