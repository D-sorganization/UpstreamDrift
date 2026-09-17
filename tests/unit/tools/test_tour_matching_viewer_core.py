"""Tests for Tour Matching Viewer core functionality (Step 3 of Visuals Handoff).

Tests pure data operations:
1. `load_replay(path)`:
   - Accepts native `returned-replay.npz` layout (time_s, native_state, markers_m, target_m, valid).
   - Accepts OpenSim IK `.mot` layout.
   - Rejects mismatched frame counts, non-monotone time, and invalid file layouts.
2. `body_poses_from_state(spec, q)`:
   - Returns a 4x4 pose per body from the spec joint tree.
   - Validates against MuJoCo forward kinematics (xpos and xmat) to < 1e-9 tolerance across 20 random states.
3. `viewer_frame(spec, replay, i)`:
   - Returns world capsule segments, target markers, and model markers for frame `i`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.tour_matching_viewer.core import (
    ReplayData,
    body_poses_from_state,
    load_replay,
    viewer_frame,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_load_replay_npz_happy_path(tmp_path: Path) -> None:
    """Validate loading a standard native replay .npz."""
    n_frames = 10
    n_coords = 27
    n_markers = 34

    time_s = np.linspace(0.0, 0.1, n_frames)
    native_state = np.zeros((n_frames, 2 * n_coords), dtype=float)
    markers_m = np.zeros((n_frames, n_markers, 3), dtype=float)
    target_m = np.zeros((n_frames, n_markers, 3), dtype=float)
    valid = np.ones((n_frames, n_markers), dtype=bool)

    npz_path = tmp_path / "test_replay.npz"
    np.savez_compressed(
        npz_path,
        time_s=time_s,
        native_state=native_state,
        markers_m=markers_m,
        target_m=target_m,
        valid=valid,
    )

    replay = load_replay(npz_path)
    assert isinstance(replay, ReplayData)
    assert replay.frame_count == n_frames
    assert np.allclose(replay.time_s, time_s)
    assert replay.coordinates.shape == (n_frames, n_coords)
    assert replay.model_markers_m is not None
    assert replay.model_markers_m.shape == (n_frames, n_markers, 3)
    assert replay.target_markers_m is not None
    assert replay.target_markers_m.shape == (n_frames, n_markers, 3)
    assert replay.valid_mask is not None
    assert replay.valid_mask.shape == (n_frames, n_markers)


def test_load_replay_mot_happy_path(tmp_path: Path) -> None:
    """Validate loading an OpenSim IK .mot file."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    coord_names = spec["coordinate_order"]
    n_frames = 5
    time_s = np.linspace(0.0, 0.04, n_frames)

    # Construct OpenSim .mot text
    lines = [
        "Coordinates",
        "version=1",
        f"nRows={n_frames}",
        f"nColumns={len(coord_names) + 1}",
        "inDegrees=no",
        "endheader",
        "\t".join(["time", *coord_names]),
    ]
    for i, t in enumerate(time_s):
        vals = [f"{t:.4f}", *[f"{0.01 * (i + 1):.4f}" for _ in coord_names]]
        lines.append("\t".join(vals))

    mot_path = tmp_path / "test_ik.mot"
    mot_path.write_text("\n".join(lines), encoding="utf-8")

    replay = load_replay(mot_path, spec=spec)
    assert replay.frame_count == n_frames
    assert np.allclose(replay.time_s, time_s)
    assert replay.coordinates.shape == (n_frames, len(coord_names))
    assert replay.model_markers_m is None
    assert replay.target_markers_m is None


def test_load_replay_rejects_non_monotone_time(tmp_path: Path) -> None:
    """Non-monotone time must be rejected with ValueError."""
    time_s = np.array([0.0, 0.05, 0.02, 0.1])
    npz_path = tmp_path / "bad_time.npz"
    np.savez_compressed(
        npz_path,
        time_s=time_s,
        native_state=np.zeros((4, 10)),
        markers_m=np.zeros((4, 5, 3)),
        target_m=np.zeros((4, 5, 3)),
        valid=np.ones((4, 5), dtype=bool),
    )
    with pytest.raises(ValueError, match="monotone"):
        load_replay(npz_path)


def test_load_replay_rejects_mismatched_frame_counts(tmp_path: Path) -> None:
    """Arrays with differing frame counts must be rejected with ValueError."""
    time_s = np.array([0.0, 0.01, 0.02])
    npz_path = tmp_path / "mismatched.npz"
    np.savez_compressed(
        npz_path,
        time_s=time_s,
        native_state=np.zeros((4, 10)),
        markers_m=np.zeros((3, 5, 3)),
        target_m=np.zeros((3, 5, 3)),
        valid=np.ones((3, 5), dtype=bool),
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_replay(npz_path)


def test_body_poses_from_state_parity_with_mujoco() -> None:
    """Verify body_poses_from_state against MuJoCo forward kinematics (< 1e-9 tolerance)."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    xml, _ = exporter.export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(10062)
    for _ in range(20):
        q_rand = rng.normal(size=model.nq) * 0.2
        data.qpos[:] = q_rand
        mujoco.mj_forward(model, data)

        poses = body_poses_from_state(spec, q_rand)

        for b in range(model.nbody):
            bname = model.body(b).name
            expected_pos = data.xpos[b]
            expected_rot = data.xmat[b].reshape(3, 3)

            actual_pos = poses[bname][:3, 3]
            actual_rot = poses[bname][:3, :3]

            np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-9)
            np.testing.assert_allclose(actual_rot, expected_rot, atol=1e-9)


def test_viewer_frame_computes_world_segments_and_markers(tmp_path: Path) -> None:
    """Test viewer_frame produces valid WorldSegments and marker arrays."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    n_frames = 3
    n_coords = len(spec["coordinate_order"])
    n_markers = 34

    time_s = np.linspace(0.0, 0.02, n_frames)
    coords = np.zeros((n_frames, n_coords), dtype=float)
    target_m = np.ones((n_frames, n_markers, 3), dtype=float) * 1.5
    model_markers_m = np.ones((n_frames, n_markers, 3), dtype=float) * 1.4
    valid = np.ones((n_frames, n_markers), dtype=bool)

    npz_path = tmp_path / "replay.npz"
    np.savez_compressed(
        npz_path,
        time_s=time_s,
        native_state=np.column_stack([coords, np.zeros_like(coords)]),
        markers_m=model_markers_m,
        target_m=target_m,
        valid=valid,
    )
    replay = load_replay(npz_path)

    frame_data = viewer_frame(spec, replay, 0)
    assert len(frame_data.segments) > 20
    assert frame_data.target_markers is not None
    assert frame_data.target_markers.shape == (n_markers, 3)
    assert frame_data.model_markers is not None
    assert frame_data.model_markers.shape == (n_markers, 3)
    assert frame_data.valid_mask is not None
    assert frame_data.valid_mask.shape == (n_markers,)
    assert frame_data.rms_error >= 0.0


def test_native_simscape_replay_fk_matches_archived_markers() -> None:
    """Native model playback must preserve all saved MATLAB marker locations."""
    from scipy.io import loadmat
    from src.engines.physics_engines.mujoco.python.native_mjcf import transform

    evidence = ROOT / "docs/development/simscape_tour_matching/native_evidence"
    spec = json.loads((evidence / "native_geometry_spec_9967.json").read_text())
    run = evidence / "two_window_fit_9967_102"
    candidate = json.loads((run / "returned-candidate.json").read_text())
    replay = loadmat(run / "qualified_candidate_replay.mat")
    offsets = np.asarray(candidate["marker_offsets_m"])
    body_offsets = {
        j["child"]: np.linalg.inv(transform(j["child_to_follower"]))
        for j in spec["joints"]
    }
    frames = {f["name"]: f for f in spec["frames"]}
    for frame in (0, 150, 250, 306):
        poses = body_poses_from_state(
            spec, replay["q"][frame], candidate["coordinate_names"]
        )
        calculated = []
        for body, offset in zip(candidate["marker_bodies"], offsets, strict=True):
            attached = frames[body]
            owner = attached["body"]
            physical = (
                poses[owner] @ body_offsets[owner] @ transform(attached["placement"])
            )
            calculated.append(physical[:3, :3] @ offset + physical[:3, 3])
        # Saved coordinates and markers are separate sampled MATLAB outputs.
        # Qualify visual reconstruction at 0.1 mm, not machine-precision FK parity.
        errors = np.linalg.norm(
            np.asarray(calculated) - replay["prediction"][frame], axis=1
        )
        assert np.max(errors) < 1e-4


def test_visual_capsules_start_at_joint_follower_origins() -> None:
    """Physical-body visual endpoints must not be applied in joint frames twice."""
    spec = json.loads(SPEC_PATH.read_text())
    names = tuple(spec["coordinate_order"])
    q = np.linspace(-0.1, 0.1, len(names))
    replay = ReplayData(np.array([0.0]), q[None, :], coordinate_names=names)
    followers = body_poses_from_state(spec, q, names)
    frame = viewer_frame(spec, replay, 0)
    for segment in frame.segments:
        np.testing.assert_allclose(
            segment.start_m, followers[segment.body][:3, 3], atol=1e-12
        )


def test_cylinder_faces_geometry() -> None:
    from src.tools.tour_matching_viewer.core import cylinder_faces

    # Zero length segment returns empty faces
    assert cylinder_faces([0, 0, 0], [0, 0, 0], 0.05) == []
    # Zero or negative radius returns empty faces
    assert cylinder_faces([0, 0, 0], [0, 0, 1], 0.0) == []
    assert cylinder_faces([0, 0, 0], [0, 0, 1], -0.05) == []

    # Valid cylinder produces n_theta quads
    faces = cylinder_faces([0, 0, 0], [0, 0, 1], 0.05, n_theta=12)
    assert len(faces) == 12
    for quad in faces:
        assert quad.shape == (4, 3)
        # Check Z coordinates: 2 vertices at z=0, 2 vertices at z=1
        z_vals = sorted(quad[:, 2])
        np.testing.assert_allclose(z_vals[:2], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(z_vals[2:], [1.0, 1.0], atol=1e-9)
        # Check radial distance in xy is ~0.05
        radii = np.linalg.norm(quad[:, :2], axis=1)
        np.testing.assert_allclose(radii, 0.05, atol=1e-9)


def test_marker_error_vectors_and_magnitudes() -> None:
    from src.tools.tour_matching_viewer.core import marker_error_vectors

    targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    models = np.array([[0.0, 0.0, 0.01], [1.0, 0.02, 0.0], [2.0, 0.0, 0.0]])
    valid = np.array([True, True, False])

    lines = marker_error_vectors(targets, models, valid)
    assert len(lines) == 2
    np.testing.assert_allclose(lines[0][0], targets[0])
    np.testing.assert_allclose(lines[0][1], models[0])
    np.testing.assert_allclose(lines[1][0], targets[1])
    np.testing.assert_allclose(lines[1][1], models[1])

    # Viewer frame computes per-marker errors in mm
    spec = json.loads(SPEC_PATH.read_text())
    names = tuple(spec["coordinate_order"])
    q = np.zeros(len(names))
    replay = ReplayData(
        time_s=np.array([0.0]),
        coordinates=q[None, :],
        model_markers_m=models[None, :],
        target_markers_m=targets[None, :],
        valid_mask=valid[None, :],
        coordinate_names=names,
    )
    vframe = viewer_frame(spec, replay, 0)
    assert vframe.marker_errors_mm is not None
    assert len(vframe.marker_errors_mm) == 3
    np.testing.assert_allclose(vframe.marker_errors_mm[0], 10.0, atol=1e-6)  # 10 mm
    np.testing.assert_allclose(vframe.marker_errors_mm[1], 20.0, atol=1e-6)  # 20 mm
    np.testing.assert_allclose(vframe.marker_errors_mm[2], 0.0, atol=1e-6)


def test_wrench_arrow_vectors_and_viewer_frame_arrows() -> None:
    from src.tools.tour_matching_viewer.core import (
        viewer_frame,
        wrench_arrow_vectors,
    )

    pt = np.array([0.1, 0.2, 0.3])
    vec = np.array([10.0, -20.0, 30.0])
    start, end = wrench_arrow_vectors(pt, vec, scale=0.01)
    np.testing.assert_allclose(start, pt)
    np.testing.assert_allclose(end, pt + 0.01 * vec)

    spec = json.loads(SPEC_PATH.read_text())
    names = tuple(spec["coordinate_order"])
    q = np.zeros(len(names))
    wrench = np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    replay = ReplayData(
        time_s=np.array([0.0]),
        coordinates=q[None, :],
        coordinate_names=names,
        reaction_wrenches_N_Nm=wrench[None, :],
        wrench_points_m=pt[None, :],
    )
    vframe = viewer_frame(spec, replay, 0)
    assert vframe.force_arrow is not None
    assert vframe.moment_arrow is not None
    np.testing.assert_allclose(vframe.force_arrow[0], pt)
    np.testing.assert_allclose(vframe.force_arrow[1], pt + 0.005 * wrench[:3])
    np.testing.assert_allclose(vframe.moment_arrow[0], pt)
    np.testing.assert_allclose(vframe.moment_arrow[1], pt + 0.01 * wrench[3:])


def test_export_provenance_table_csv_and_json(tmp_path: Path) -> None:
    from src.tools.tour_matching_viewer.core import export_provenance_table

    time_s = np.array([0.0, 0.05, 0.10])
    coords = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    names = ("joint_a", "joint_b")
    wrenches = np.array(
        [
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            [15.0, 25.0, 35.0, 1.5, 2.5, 3.5],
            [20.0, 30.0, 40.0, 2.0, 3.0, 4.0],
        ]
    )
    replay = ReplayData(
        time_s=time_s,
        coordinates=coords,
        coordinate_names=names,
        reaction_wrenches_N_Nm=wrenches,
    )

    # Export CSV
    csv_out = tmp_path / "provenance.csv"
    res_csv = export_provenance_table(
        replay, csv_out, candidate_hash="cand123", engine_name="pinocchio"
    )
    assert res_csv.exists()
    csv_lines = csv_out.read_text(encoding="utf-8").splitlines()
    header = csv_lines[0].split(",")
    assert "frame" in header
    assert "time_s" in header
    assert "candidate_hash" in header
    assert "engine" in header
    assert "q_joint_a" in header
    assert "q_joint_b" in header
    assert "Fx_N" in header
    assert "Mz_Nm" in header
    assert len(csv_lines) == 4  # header + 3 rows

    # Export JSON
    json_out = tmp_path / "provenance.json"
    res_json = export_provenance_table(
        replay, json_out, candidate_hash="cand123", engine_name="pinocchio"
    )
    assert res_json.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["candidate_hash"] == "cand123"
    assert payload["engine"] == "pinocchio"
    assert payload["n_frames"] == 3
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["Fx_N"] == 10.0
    assert payload["rows"][0]["q_joint_a"] == 0.1
