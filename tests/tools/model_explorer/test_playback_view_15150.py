"""The explorer opens on the behind-the-golfer view, midway ball-to-feet (#15150)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.golf_view_presets import get_view_preset
from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.tools.model_explorer import playback_view as pv
from src.tools.native_viewer_export import behind_camera as bc

pytestmark = pytest.mark.unit

LOOK = (1.0, 0.0, 0.9)
ANCHOR = (0.4, -0.3, 0.0)


def test_settings_file_documents_the_default_view() -> None:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert raw["default_view"] == "down_the_line"
    assert "default_view" in raw["sources"]
    assert pv.load_view_settings().default_view == "down_the_line"


def test_default_view_must_be_a_known_preset(tmp_path) -> None:
    bad = tmp_path / "view.json"
    bad.write_text(json.dumps({"default_view": "selfie", "sources": {}}))
    with pytest.raises(ValueError, match="selfie"):
        pv.load_view_settings(bad)


def test_default_view_is_offered_by_the_view_combo() -> None:
    assert pv.load_view_settings().default_view in VIEW_ORDER


def test_shift_moves_a_behind_camera_onto_the_anchor_line() -> None:
    moved = np.asarray(bc.shift_to_anchor("down_the_line", LOOK, ANCHOR))
    right = get_view_preset("down_the_line").image_right()
    # the anchor now sits at the picture centre (no sideways offset) ...
    assert np.dot(np.asarray(ANCHOR) - moved, right) == pytest.approx(0.0, abs=1e-12)
    # ... and only the sideways component moved: height and depth are kept
    assert np.cross(moved - np.asarray(LOOK), right) == pytest.approx(0.0, abs=1e-12)


def test_shift_leaves_other_views_and_a_missing_anchor_alone() -> None:
    assert bc.shift_to_anchor("face_on", LOOK, ANCHOR) == LOOK
    assert bc.shift_to_anchor("down_the_line", LOOK, None) == LOOK


def test_playback_lookat_uses_the_anchor_for_behind_views_only() -> None:
    behind = pv.playback_lookat("down_the_line", ANCHOR)
    assert behind == bc.shift_to_anchor("down_the_line", pv.DEFAULT_LOOKAT_M, ANCHOR)
    assert pv.playback_lookat("face_on", ANCHOR) == pv.DEFAULT_LOOKAT_M
    assert pv.playback_lookat("down_the_line", None) == pv.DEFAULT_LOOKAT_M


def test_playback_lookat_rejects_unknown_views() -> None:
    with pytest.raises(ValueError, match="selfie"):
        pv.playback_lookat("selfie", ANCHOR)


REPO = Path(__file__).resolve().parents[3]


def _fixture_pose(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.tools.model_explorer.pose_results import load_pose_frame_source

    run = tmp_path / "fit"
    shutil.copytree(REPO / "tests/fixtures/results_playback/mujoco_fit", run)
    spec = REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    (run / "full_body_spec_hipcal_scaled.json").write_bytes(spec.read_bytes())
    return load_pose_frame_source(run / "dynamics_record.npz")


def test_anchor_of_a_run_sits_the_set_fraction_from_ball_to_feet(tmp_path) -> None:
    from src.tools.model_explorer.playback_ball_overlay import swing_for_source
    from src.tools.native_viewer_export.ball import address_ball
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    pose = _fixture_pose(tmp_path)
    anchor = pv.behind_anchor_for(pose)
    assert anchor is not None and np.all(np.isfinite(anchor))
    swing = swing_for_source(pose)
    settings = bc.load_behind_settings()
    coords = dict(zip(pose.coordinate_order, map(float, pose.q[0]), strict=True))
    feet = bc.feet_midpoint(
        ClubPoseModel(pose.spec_bytes), coords, settings.foot_bodies
    )
    ball = np.asarray(address_ball(swing, None).position_m)
    expected = ball + settings.fraction * (np.asarray(feet) - ball)
    assert anchor == pytest.approx(tuple(expected), abs=1e-9)


def test_anchor_is_none_when_the_run_has_no_time_base(tmp_path) -> None:
    from dataclasses import replace

    pose = _fixture_pose(tmp_path)
    frozen = replace(pose, time_s=np.zeros_like(pose.time_s))
    assert pv.behind_anchor_for(frozen) is None


def test_layout_settings_are_documented_and_positive() -> None:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    layout = pv.load_layout_settings()
    for key in (
        "render_stretch",
        "plot_stretch",
        "hide_editor_docks",
        "slider_min_width_px",
    ):
        assert key in raw["sources"], key
    assert layout.render_stretch > layout.plot_stretch > 0
    assert isinstance(layout.hide_editor_docks, bool)


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"render_stretch": 0}, "render_stretch"),
        ({"plot_stretch": -1}, "plot_stretch"),
        ({"render_stretch": 2.5}, "render_stretch"),
        ({"hide_editor_docks": "yes"}, "hide_editor_docks"),
        ({"slider_min_width_px": 0}, "slider_min_width_px"),
    ],
)
def test_layout_settings_reject_bad_values(tmp_path, patch, match) -> None:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["layout"].update(patch)
    bad = tmp_path / "view.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        pv.load_layout_settings(bad)


def _patched(tmp_path, block: str, patch) -> Path:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    raw[block] = patch(raw[block]) if callable(patch) else patch
    out = tmp_path / "view.json"
    out.write_text(json.dumps(raw))
    return out


def test_camera_distances_are_documented_and_known_views() -> None:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    distances = pv.load_camera_distances()
    assert "camera_distance_m" in raw["sources"]
    assert set(distances) <= set(VIEW_ORDER)
    assert {"down_the_line", "face_on"} <= set(distances)
    preset = get_view_preset("down_the_line").default_distance_m
    assert 0 < distances["down_the_line"] < preset  # closer: the golfer fills more


@pytest.mark.parametrize(
    ("distances", "match"),
    [
        ({"selfie": 2.0}, "selfie"),
        ({"face_on": 0}, "face_on"),
        ({"face_on": -1.5}, "face_on"),
        ({"face_on": float("nan")}, "face_on"),
        ({"face_on": "near"}, "face_on"),
        ({"face_on": True}, "face_on"),
    ],
)
def test_camera_distances_reject_bad_values(tmp_path, distances, match) -> None:
    path = _patched(tmp_path, "camera_distance_m", distances)
    with pytest.raises(ValueError, match=match):
        pv.load_camera_distances(path)


def test_framing_settings_are_documented() -> None:
    raw = json.loads(pv.SETTINGS_PATH.read_text(encoding="utf-8"))
    cfg = pv.load_framing_settings()
    for key in ("framing", "camera_distance_m"):
        assert key in raw["sources"], key
    assert 0 < cfg.min_aspect < cfg.max_aspect
    assert cfg.render_height_px >= 16 and cfg.resize_debounce_ms > 0


def test_render_size_follows_the_aspect_and_is_clamped() -> None:
    cfg = pv.load_framing_settings()
    wide = cfg.render_size(10.0)
    narrow = cfg.render_size(0.01)
    assert wide == (round(cfg.render_height_px * cfg.max_aspect), cfg.render_height_px)
    assert narrow == (
        round(cfg.render_height_px * cfg.min_aspect),
        cfg.render_height_px,
    )
    mid = cfg.render_size(2.0)
    assert mid[0] / mid[1] == pytest.approx(2.0, abs=cfg.aspect_step)
    with pytest.raises(ValueError, match="aspect"):
        cfg.render_size(float("nan"))
    with pytest.raises(ValueError, match="aspect"):
        cfg.render_size(0.0)


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"render_height_px": 8}, "render_height_px"),
        ({"min_aspect": 4.0}, "min_aspect"),
        ({"max_aspect": 0}, "max_aspect"),
        ({"aspect_step": 0}, "aspect_step"),
        ({"aspect_step": float("inf")}, "aspect_step"),
        ({"resize_debounce_ms": 0}, "resize_debounce_ms"),
        ({"resize_debounce_ms": 1.5}, "resize_debounce_ms"),
    ],
)
def test_framing_settings_reject_bad_values(tmp_path, patch, match) -> None:
    path = _patched(tmp_path, "framing", lambda blk: {**blk, **patch})
    with pytest.raises(ValueError, match=match):
        pv.load_framing_settings(path)
