"""Excluded setup frames must not become calibration or speed evidence (#9860)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct import RenderOptions, SyntheticScene
from src.motion_capture.reconstruct.analytics import summarize_swing
from src.motion_capture.reconstruct.initialize import Initialization
from src.motion_capture.reconstruct.pipeline import _initial_cameras
from tests.motion_capture.reconstruct.test_initialize import _rig

pytestmark = pytest.mark.unit


def test_initial_camera_subject_frame_starts_with_selected_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cams = _rig()
    views, _ = SyntheticScene(cams, n_frames=40).render(
        RenderOptions(noise_px=0, occlusion_rate=0, outlier_rate=0)
    )
    directory = tmp_path / "observations"
    directory.mkdir()
    for view, payload in views.items():
        payload["frames"] = payload["frames"][20:]
        (directory / f"{view}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "src.motion_capture.reconstruct.pipeline.initialize_cameras",
        lambda *a, **k: Initialization(tuple(cams), ()),
    )
    result = _initial_cameras(
        tmp_path,
        [c.camera_id for c in cams],
        [(c.camera_id, c.matrix, c.image_size_px) for c in cams],
        ("neck", 0.5),
    )
    assert len(result) == len(cams)
    assert all(np.isfinite(c.position_m).all() for c in result)


def test_excluded_prefix_and_suffix_do_not_change_swing_metrics() -> None:
    _, truth = SyntheticScene(_rig(), n_frames=60).render(
        RenderOptions(noise_px=0, occlusion_rate=0, outlier_rate=0)
    )
    joints = np.asarray(truth.joints_3d_m)
    baseline, baseline_series = summarize_swing(joints, 60)
    padded = np.pad(joints, ((25, 30), (0, 0), (0, 0)))
    result, series = summarize_swing(padded, 60)
    assert result.peak_hand_speed_mps == pytest.approx(baseline.peak_hand_speed_mps)
    assert result.max_shoulder_turn_deg == pytest.approx(baseline.max_shoulder_turn_deg)
    assert result.peak_hand_speed_frame == baseline.peak_hand_speed_frame + 25
    assert result.events.top_frame == baseline.events.top_frame + 25
    assert result.frames == 115
    assert np.allclose(series.hand_speed_mps[25:85], baseline_series.hand_speed_mps)
    assert not series.hand_speed_mps[:25].any()
    assert np.isnan(series.shoulder_turn_deg[:25]).all()
