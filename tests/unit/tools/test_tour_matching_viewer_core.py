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
    assert replay.model_markers_m.shape == (n_frames, n_markers, 3)
    assert replay.target_markers_m.shape == (n_frames, n_markers, 3)
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
    assert frame_data.target_markers.shape == (n_markers, 3)
    assert frame_data.model_markers.shape == (n_markers, 3)
    assert frame_data.valid_mask.shape == (n_markers,)
    assert frame_data.rms_error >= 0.0
