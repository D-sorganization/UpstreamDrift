"""Session-level model fit: reconstruct outputs in, model/ directory out (#9713)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import ArticulatedModel, FitOptions
from src.motion_capture.reconstruct.model.session import (
    LandmarkMap,
    fit_session_model,
    initial_state,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from tests.motion_capture.reconstruct.model.test_kinematics_fit import ARM, _true_motion

pytestmark = pytest.mark.unit

MAP = LandmarkMap(
    to_reconstruct={
        "pelvis": "mid_hip",
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
    },
    length_from_segment={"forearm": "left_wrist", "shank": "left_ankle"},
)


def test_landmark_map_contracts_and_length_substitution() -> None:
    with pytest.raises(Exception, match="reconstruct joints"):
        LandmarkMap({"pelvis": "sacrum"})
    lengths = MAP.lengths({"left_wrist": 0.27, "left_ankle": 0.41, "neck": 0.5}, ARM)
    assert lengths["forearm"] == 0.27 and lengths["shank"] == 0.41
    assert lengths["torso"] == ARM.lengths_m["torso"]


def test_fit_session_model_writes_angles_report_and_landmarks(tmp_path: Path) -> None:
    model = ArticulatedModel(ARM)
    fps, frames = 60.0, 40
    q_true = _true_motion(model, frames, fps)
    landmarks = model.landmarks(q_true)
    # Scatter the model landmarks into the 15-joint reconstruct layout.
    joints = np.full((frames, len(JOINT_NAMES), 3), np.nan)
    for name, source in MAP.to_reconstruct.items():
        joints[:, JOINT_NAMES.index(source)] = landmarks[
            :, model.landmark_names.index(name)
        ]
    recon = tmp_path / "reconstruct"
    recon.mkdir()
    np.save(recon / "joints_3d_m.npy", joints)
    (recon / "session_reconstruction.json").write_text(
        json.dumps({"fps": fps, "measured_lengths_m": {"left_wrist": 0.26}}),
        encoding="utf-8",
    )
    observed = MAP.observed(model, joints)
    assert np.isnan(observed[:, model.landmark_names.index("scapula")]).all()
    q0 = initial_state(model, observed)
    np.testing.assert_allclose(q0[:, :3], q_true[:, :3], atol=1e-12)
    fit, out_dir = fit_session_model(
        tmp_path, ARM, MAP, options=FitOptions(max_iterations=60)
    )
    assert out_dir == tmp_path / "model"
    payload = json.loads((out_dir / "joint_angles.json").read_text(encoding="utf-8"))
    assert payload["model"] == "test_arm" and len(payload["q"]) == frames
    report = json.loads((out_dir / "fit_report.json").read_text(encoding="utf-8"))
    assert report["landmarks"]["wrist"]["frames"] == frames
    assert report["landmarks"]["scapula"]["frames"] == 0  # unobserved, constrained
    assert report["rms_mm"] < 8.0
    assert np.load(out_dir / "landmarks_fit.npy").shape == landmarks.shape
    assert fit.lengths_m["forearm"] == 0.26
    with pytest.raises(Exception, match="no reconstructed joints"):
        fit_session_model(tmp_path / "nope", ARM, MAP)


def test_load_joints_turns_unobservable_zero_rows_into_nan(tmp_path: Path) -> None:
    """The reconstruction's all-zero rows are 'no two cameras saw it' (#9802)."""
    from src.motion_capture.reconstruct.model.session import load_joints

    joints = np.ones((3, 2, 3))
    joints[1, 0] = 0.0
    path = tmp_path / "joints_3d_m.npy"
    np.save(path, joints)
    loaded = load_joints(path)
    assert np.isnan(loaded[1, 0]).all() and np.isfinite(loaded[1, 1]).all()
    assert np.isfinite(loaded[0]).all()
