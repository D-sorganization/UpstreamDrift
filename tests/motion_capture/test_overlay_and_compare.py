"""Projection tracks, held-out views and the variant comparison (#9795, #9796)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.compare_variants import (
    angle_rms_deg,
    compare_variants,
    joint_rms_mm,
    markdown,
)
from src.motion_capture.reconstruct.__main__ import lab_rig
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.fit2d import (
    ImageSpaceSource,
    fit_session_model_2d,
)
from src.motion_capture.reconstruct.model.golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from src.motion_capture.reconstruct.model.session import fit_session_model
from src.motion_capture.reconstruct.overlay3d import (
    PALETTE,
    camera_for_view,
    edges_from_landmark_map,
    project_track,
    skeleton_edges,
    variant_tracks,
)
from src.motion_capture.reconstruct.pipeline import (
    MatchSpec,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.motion_capture.variants import variant_dir
from tests.motion_capture.reconstruct.test_pipeline import _session

pytestmark = pytest.mark.unit


def test_project_track_marks_invisible_points() -> None:
    camera = lab_rig()[0]
    pts = np.zeros((2, 3, 3))
    pts[0, 0] = [0.0, 1.0, 0.0]  # the rig's target: visible
    pts[0, 1] = (
        camera.position_m + camera.rotation_world_from_camera[:, 2] * -1
    )  # behind
    pts[0, 2] = [0.0, 1.0, 0.0] + camera.rotation_world_from_camera[
        :, 0
    ] * 50  # off image
    px, visible = project_track(pts, camera)
    assert px.shape == (2, 3, 2) and visible.shape == (2, 3)
    assert visible[0].tolist() == [True, False, False]
    assert not visible[1].any(), "all-zero rows are unobservable"
    with pytest.raises(Exception, match="T, K, 3"):
        project_track(np.zeros((2, 3)), camera)


def test_edges_follow_the_skeleton_and_the_landmark_map() -> None:
    edges = skeleton_edges()
    names = list(JOINT_NAMES)
    assert (names.index("neck"), names.index("mid_hip")) in edges
    landmark_names = ("pelvis", "hub", "left_shoulder", "hands")
    edges2 = edges_from_landmark_map(
        landmark_names,
        {
            "pelvis": "mid_hip",
            "hub": "neck",
            "left_shoulder": "left_shoulder",
            "hands": ["left_wrist", "right_wrist"],
        },
    )
    assert (1, 0) in edges2 and (2, 1) in edges2
    assert all(3 not in e for e in edges2), "tuple sources have no single parent"


def test_rms_helpers() -> None:
    a = np.zeros((4, len(JOINT_NAMES), 3))
    b = a.copy()
    a[:, :, 0] = 1.0
    b[:, :, 0] = 1.01
    b[1, 3] = 0.0  # unobservable in b
    out = joint_rms_mm(a, b)
    assert out["overall"] == pytest.approx(10.0)
    assert out["per_joint"]["mid_hip"] == pytest.approx(10.0)
    assert out["frames"] == 4
    qa = {
        "model": "m",
        "dof_names": ["pelvis.tx", "pelvis.ty", "pelvis.tz", "a.rx"],
        "q": [[0, 0, 0, 3.1]],
    }
    qb = {"model": "m", "dof_names": qa["dof_names"], "q": [[0, 0, 0, -3.1]]}
    ang = angle_rms_deg(qa, qb)
    assert ang is not None and ang["per_dof"]["a.rx"] == pytest.approx(
        np.degrees(0.0832), abs=0.01
    )
    assert angle_rms_deg(qa, {**qb, "model": "other"}) is None


@pytest.mark.timeout(600)
def test_variant_tracks_and_comparison_on_the_lab_session(tmp_path: Path) -> None:
    session, cameras = _session(tmp_path)
    reconstruct_session(
        session, start_cameras=start_cameras_from(cameras), scale_anchor=("neck", 0.5)
    )
    reconstruct_session(
        session,
        start_cameras=start_cameras_from(cameras),
        scale_anchor=("neck", 0.5),
        match=MatchSpec(views=("face_on", "down_line"), variant="pair_fd"),
    )
    fit_session_model(
        session, GOLFER_SPEC, GOLFER_LANDMARK_MAP, options=FitOptions(max_iterations=8)
    )
    fit_session_model(
        variant_dir(session, "pair_fd"),
        GOLFER_SPEC,
        GOLFER_LANDMARK_MAP,
        options=FitOptions(max_iterations=8),
        session_root=session,
    )
    fit_session_model_2d(
        session,
        GOLFER_SPEC,
        GOLFER_LANDMARK_MAP,
        ImageSpaceSource(("overhead",), "", "observations", "cam_over"),
        options=FitOptions(max_iterations=6),
    )
    # A held-out view of the pair variant still projects (cameras from the pair's own file).
    tracks = variant_tracks(session, "pair_fd", "overhead", PALETTE[1])
    kinds = {t.kind for t in tracks}
    assert kinds == {"joints", "model"}
    joints = next(t for t in tracks if t.kind == "joints")
    assert joints.held_out and "(held out)" in joints.label
    assert joints.visible.any() and joints.edges
    # The image-space variant borrows cameras from the default variant.
    assert camera_for_view(session, "cam_over", "face_on").camera_id == "face_on"
    model_only = variant_tracks(session, "cam_over", "face_on", PALETTE[2])
    assert [t.kind for t in model_only] == ["model"]
    with pytest.raises(ValueError, match="no camera"):
        camera_for_view(session, "pair_fd", "nope")

    payload = compare_variants(session)
    names = [row["variant"] for row in payload["variants"]]
    assert names[0] == "" and set(names) == {"", "pair_fd", "cam_over"}
    pair = next(r for r in payload["variants"] if r["variant"] == "pair_fd")
    assert pair["reprojection"]["overhead"]["held_out"] is True
    assert pair["reprojection"]["face_on"]["held_out"] is False
    assert pair["reprojection"]["overhead"]["joints_rms_px"] is not None
    assert pair["joint_rms_mm"]["overall"] is not None
    assert pair["angle_rms_deg"]["overall"] is not None
    over = next(r for r in payload["variants"] if r["variant"] == "cam_over")
    assert over["joint_rms_mm"] is None and over["model_rms_px"] is not None
    text = markdown(payload)
    assert "(held out)" in text and "| (default) |" in text
    written = json.loads((session / "variants" / "comparison.json").read_text("utf-8"))
    assert written["schema_version"] == "variant-comparison/1.0.0"
    assert written["provenance"]["parameters"]["reference"] == ""
    with pytest.raises(Exception, match="reference"):
        compare_variants(session, reference="missing")
