"""Settings and smoothing of the grip-driven club motion (#15684)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from src.shared.python.club_fitting.grip_motion import (
    ClubPoseTrack,
    FittingSettings,
    GripMotion,
    grip_point_in_club,
    load_settings,
    repair_outliers,
)
from src.shared.python.motion_matching import club_models as cm

pytestmark = pytest.mark.unit

SAMPLE_DT = 0.002


def test_outlier_repair_fixes_a_single_bad_sample() -> None:
    t = np.arange(40) * SAMPLE_DT
    sig = np.stack([t, t**2, 0 * t, 1 + 0 * t, 0 * t, 0 * t, 0 * t], axis=1)
    sig = sig + 1e-6 * np.random.default_rng(0).standard_normal(sig.shape)
    bad = sig.copy()
    bad[20, 0] += 0.5
    fixed, times = repair_outliers(t, bad, 4, 6.0)
    assert t[20] in times
    assert abs(fixed[20, 0] - sig[20, 0]) < 1e-6


def test_settings_load_and_validate(tmp_path) -> None:
    cfg = load_settings()
    assert cfg.record_every >= 1 and cfg.lowpass_cutoff_hz is not None
    for bad in (
        {"integration_dt_s": 0.0},
        {"record_every": 0},
        {"lowpass_cutoff_hz": -1.0},
        {"filter_fit_samples": 2},
        {"lead_in_s": float("nan")},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(cfg, **bad)
    data = dataclasses.asdict(cfg) | {"surprise": 1}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load_settings(path)
    assert isinstance(cfg, FittingSettings)


def test_track_contracts() -> None:
    t = np.arange(10) * SAMPLE_DT
    grip, rot = np.zeros((10, 3)), np.repeat(np.eye(3)[None], 10, axis=0)
    ClubPoseTrack(t, grip, rot)
    with pytest.raises(ValueError):
        ClubPoseTrack(t[::-1], grip, rot)
    with pytest.raises(ValueError):
        ClubPoseTrack(t, grip[:5], rot)


def test_grip_motion_follows_a_straight_line() -> None:
    cfg = dataclasses.replace(load_settings(), lowpass_cutoff_hz=None)
    t = np.arange(0.0, 0.4, SAMPLE_DT)
    grip = np.stack([2.0 * t, 0 * t, 1 + 0 * t], axis=1)
    rot = np.repeat(np.eye(3)[None], len(t), axis=0)
    motion = GripMotion(
        ClubPoseTrack(t, grip, rot), grip_point_in_club(cm.DRIVER), 0.0, 0.39, cfg
    )
    _, origin, velocity, omega, accel, _ = motion.state(0.2)
    assert velocity == pytest.approx([2.0, 0.0, 0.0], abs=1e-6)
    assert omega == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert accel == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert origin[0] == pytest.approx(0.4, abs=1e-6)
    with pytest.raises(ValueError, match="at least"):
        GripMotion(
            ClubPoseTrack(t, grip, rot), grip_point_in_club(cm.DRIVER), 0.0, 0.01, cfg
        )
