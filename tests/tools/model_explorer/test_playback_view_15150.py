"""The explorer opens on the behind-the-golfer view, midway ball-to-feet (#15150)."""

from __future__ import annotations

import json

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
