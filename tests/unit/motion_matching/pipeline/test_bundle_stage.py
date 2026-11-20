"""Lane adapter of the marker-offset bundle adjustment (#15296)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import load_coordinate_swing
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.pipeline.bundle_adjustment import solve_bundle
from src.shared.python.motion_matching.pipeline.bundle_stage import (
    OFFSET_SIGMAS_M,
    StageSettings,
    adjusted_attachments,
    build_problem,
    offset_sigma,
    select_frames,
    stage_report,
)

pytestmark = [pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
FIXTURES = ROOT / "tests/fixtures/club_face"
FRAMES = 5
EVERY = StageSettings(frame_stride=1)
CLUB = "Marker_2:2:1"
#: Closure weight that shuts the weld far below its 0.5 mm residual scale.
WELD = 1e8


@pytest.fixture(scope="module")
def model() -> SimpleNamespace:
    """The committed driver model, its attachments and a lane of five frames
    whose capture is the model's own markers along the fixture swing."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )

    spec_path = EVIDENCE / "anthro_driver" / "full_body_spec_hipcal_scaled.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    receipt = json.loads((EVIDENCE / "anthro_driver" / "receipt.json").read_text())
    attachments = {
        label: (entry["body"], tuple(entry["offset_m"]))
        for label, entry in receipt["ik"]["attachments_m"].items()
    }
    source = MujocoOverlaySource(spec_path.read_bytes())
    kin = FullBodyMarkerKinematics(source, attachments)
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        spec["coordinate_order"],
    )
    height = float(spec["contact"]["ground"].get("height_m") or 0.0)
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=height)
    seen = np.ones(len(attachments), bool)
    poses = np.array(
        [  # the fixture swing is another model's: close this model's grip weld
            kin.solve_pose(
                kin.marker_positions(q), seen, q, ground=ground, closure_weight=WELD
            ).q
            for q in swing.q[np.linspace(0, len(swing.q) - 1, FRAMES).astype(int)]
        ]
    )
    lane = SimpleNamespace(
        labels=tuple(attachments),
        points=np.array([kin.marker_positions(pose) for pose in poses]),
        valid=np.ones((FRAMES, len(attachments)), bool),
        ground=ground,
        stance=[()] * FRAMES,
        bounds={},
        prior_weights={},
        plant=SimpleNamespace(
            create_ik=lambda probes: FullBodyMarkerKinematics(source, probes)
        ),
    )
    return SimpleNamespace(lane=lane, kin=kin, attachments=attachments, poses=poses)


@pytest.mark.requires_mujoco
def test_fk_and_twists_reproduce_the_lane_markers_at_the_reference_pose(model) -> None:
    prob, labels, frames = build_problem(
        model.lane, model.attachments, model.poses, EVERY
    )
    assert labels == model.lane.labels and frames == tuple(range(FRAMES))
    pose = model.poses[0]
    native = model.kin.marker_linearization(pose)
    bodies = prob.fk(pose)[prob.marker_bodies]
    arms = np.einsum("mij,mj->mi", bodies[:, :3, :3], prob.offset_seeds)
    assert np.abs(arms + bodies[:, :3, 3] - native.positions).max() < 1e-9
    twists = prob.body_jacobian(pose)[prob.marker_bodies]
    slopes = twists[:, :3] + np.cross(twists[:, 3:], arms[:, :, None], axis=1)
    assert np.abs(slopes - native.jacobian).max() < 1e-9


@pytest.mark.requires_mujoco
def test_constraint_rows_and_the_stated_sole_point(model) -> None:
    prob, _, _ = build_problem(model.lane, model.attachments, model.poses, EVERY)
    rows = prob.constraint(0, model.poses[0])
    assert rows.shape == prob.constraint(3, model.poses[3]).shape
    assert np.isfinite(rows).all() and len(rows) > 6
    body = model.attachments[CLUB][0]
    sole = (body, (0.0, 0.0, 0.05))
    stated, _, _ = build_problem(
        model.lane, model.attachments, model.poses, EVERY, sole
    )
    frame = stated.fk(model.poses[0])[
        sorted({b for b, _ in model.attachments.values()}).index(body)
    ]
    height = (frame[:3, 3] + frame[:3, :3] @ sole[1])[2] - model.lane.ground.height_m
    assert stated.constraint(0, model.poses[0])[-1] == pytest.approx(height / 0.005)
    assert stated.constraint(3, model.poses[3])[-1] == 0.0
    assert len(stated.constraint(0, model.poses[0])) == len(rows) + 1


@pytest.mark.requires_mujoco
@pytest.mark.slow
def test_adjustment_moves_a_misplaced_seed_back_and_reports_it(model) -> None:
    seeds: dict[str, Any] = dict(model.attachments)
    body, offset = seeds["LElbowOut"]
    seeds["LElbowOut"] = (body, (offset[0] + 0.010, offset[1], offset[2] - 0.008))
    settings = StageSettings(frame_stride=1, held_out=("RKneeOut",))
    prob, labels, frames = build_problem(model.lane, seeds, model.poses, settings)
    assert "RKneeOut" not in labels
    result = solve_bundle(prob, settings.solver)
    fitted = adjusted_attachments(seeds, labels, result)
    # five frames against a 12 mm prior cannot undo it all: over half comes back
    assert np.abs(np.subtract(fitted["LElbowOut"][1], offset)).max() < 0.005
    assert fitted["RKneeOut"] == seeds["RKneeOut"]
    held = labels.index(CLUB)
    assert np.array_equal(result.offsets[held], prob.offset_seeds[held])
    report = stage_report(result, labels, frames, settings)
    assert report["converged"] and report["rms_mm"] < 1.0
    assert CLUB not in report["free_markers"]
    assert not {CLUB} & set(report["weakly_identified_markers"])
    assert json.dumps(report)


def test_frame_selection_and_the_sigma_table() -> None:
    settings = StageSettings(anchor_frames=(391, 476))
    frames = select_frames(654, settings)
    assert frames[:3] == (0, 20, 40) and {391, 476} <= set(frames)
    assert len(frames) == 35 and list(frames) == sorted(set(frames))
    with pytest.raises(ValueError, match="anchor frame"):
        select_frames(100, StageSettings(anchor_frames=(100,)))
    with pytest.raises(ValueError, match="frame_stride"):
        select_frames(100, StageSettings(frame_stride=0))
    # head markers: full weight, 40 mm (the experiment's tenth weight let the prior win)
    assert offset_sigma("HeadFront") == (0.040, 0.040, 0.040)
    assert offset_sigma("LWristTop") == OFFSET_SIGMAS_M["WristTop"]
    assert offset_sigma("Marker_2:2:1") == (0.0, 0.0, 0.0)  # club markers are held
