"""``ui/src/utils/cameraPresets.golden.json`` matches the desktop tile (#14474).

``cameraPresets.test.ts`` holds the React side to the same file.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.putting_dynamics import Mode
from src.tools.putting_green_gui._camera import CAMERA_PRESETS, camera_pose
from src.tools.putting_green_gui._scene_builder import PuttConfig, build_putt_scene

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[4] / "ui/src/utils/cameraPresets.golden.json"
GOLDEN = json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _playback():
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    from src.tools.putting_green_gui import _playback

    return _playback


def test_presets_and_labels_are_the_desktop_bar() -> None:
    playback = _playback()
    assert GOLDEN["presets"] == list(CAMERA_PRESETS)
    assert GOLDEN["view_labels"] == playback._VIEW_LABELS
    assert GOLDEN["phase_labels"] == {
        mode.value: label for mode, label in playback._PHASE_LABELS.items()
    }
    assert set(GOLDEN["phase_labels"]) == {mode.value for mode in Mode}


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=["angled"])
def test_poses_are_the_desktop_camera_poses(case: dict) -> None:
    scene = SimpleNamespace(
        green_size=tuple(case["green_size"]),
        start_xyz=np.array(case["start_xyz"]),
        cup_xyz=np.array(case["cup_xyz"]),
    )
    for preset, pose in case["poses"].items():
        core = dataclasses.asdict(camera_pose(scene, preset))
        assert core["center"] == pytest.approx(tuple(pose["center"]))
        for key in ("distance", "elevation_deg", "azimuth_deg"):
            assert core[key] == pytest.approx(pose[key]), (preset, key)


@pytest.mark.parametrize("tile", GOLDEN["tiles"], ids=lambda t: str(t["distance_m"]))
def test_tiles_are_the_desktop_putt_tiles(tile: dict) -> None:
    """The web frames each putt on the desktop's tile around it."""
    scene = build_putt_scene(PuttConfig(cup_distance_m=tile["distance_m"]))
    assert scene.green_size == pytest.approx(tuple(tile["green_size"]))
    assert scene.start_xyz[:2] == pytest.approx(tile["start_xy"])
    assert scene.cup_xyz[:2] == pytest.approx(tile["cup_xy"])
