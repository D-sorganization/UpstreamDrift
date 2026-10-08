"""Finish-feasibility extraction from a real MuJoCo plant (Balance-1, #11668)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    finish_feasibility_report,
)
from src.shared.python.motion_matching.pipeline.receipt_dynamics import (
    FinishFeasibilityReceipt,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/full_body_spec_v1.json"
)


@pytest.fixture(scope="module")
def static_case():
    """A planted pose held still: reference and record share one posture."""
    fs = pytest.importorskip(
        "src.shared.python.motion_matching.full_body_forward_dynamics"
    )
    model = pytest.importorskip(
        "src.engines.physics_engines.mujoco.python.full_body_model"
    )
    sim = fs.FullBodySimulator(model.NativeMujocoFullBodyModel(SPEC.read_bytes()))
    q = fs.preload_feet(sim, np.zeros(sim.nv))
    frames = 11
    times = np.linspace(0.5, 1.5, frames)
    q_track = np.tile(q, (frames, 1))
    record = fs.SimulationRecord(
        time_s=np.linspace(1.0, 1.1, frames),
        q=q_track.copy(),
        v=np.zeros((frames, sim.nv)),
        tau=np.zeros((frames, sim.nv)),
        normal_force_n=np.zeros(frames),
        weight_fraction=np.ones(frames),
        centre_of_pressure_m=np.zeros((frames, 3)),
        inside_support_polygon=np.ones(frames, dtype=bool),
        lowest_sphere_height_m=np.zeros(frames),
    )
    ground = sim.adapter.ground_plane
    zmp = fs.reference_zmp(sim, times, q_track, ground)
    return sim, times, q_track, record, zmp, ground


def test_report_of_a_still_pose_has_no_slide_or_yaw_error(static_case) -> None:
    sim, times, q_track, record, zmp, ground = static_case
    report = finish_feasibility_report(
        sim,
        times_track=times,
        q_track=q_track,
        record=record,
        zmp=zmp,
        ground=ground,
        q_ik=q_track,
    )
    for side in ("reference", "simulation"):
        block = report[side]
        assert block["foot_slide_mm_max"] == pytest.approx(0.0, abs=1e-6)
        assert block["foot_yaw_pivot_deg_max"] == pytest.approx(0.0, abs=1e-6)
        assert block["pelvis_yaw_error_deg_max"] == pytest.approx(0.0, abs=1e-9)
        assert block["vertical_force_bw_min"] == pytest.approx(1.0, abs=1e-6)
    assert report["friction_limit_mu"] == pytest.approx(
        sim.adapter.contact_parameters.dynamic_friction
    )


def test_report_validates_against_the_receipt_schema(static_case) -> None:
    sim, times, q_track, record, zmp, ground = static_case
    report = finish_feasibility_report(
        sim, times_track=times, q_track=q_track, record=record, zmp=zmp, ground=ground
    )
    parsed = FinishFeasibilityReceipt.model_validate(report)
    assert parsed.reference.pelvis_yaw_error_deg_max is None  # no IK trajectory given
    assert parsed.simulation.pelvis_yaw_error_deg_max is not None


def test_report_rejects_tilted_ground(static_case) -> None:
    from src.shared.python.motion_matching.contact_law import GroundPlane

    sim, times, q_track, record, zmp, _ = static_case
    with pytest.raises(ValueError, match="z-up"):
        finish_feasibility_report(
            sim,
            times_track=times,
            q_track=q_track,
            record=record,
            zmp=zmp,
            ground=GroundPlane(normal=(0.0, 0.6, 0.8), height_m=0.0),
        )
