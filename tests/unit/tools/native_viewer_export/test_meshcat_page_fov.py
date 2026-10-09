"""MeshCat pages use the shared viewer field of view (NV-9, #11697)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.shared.python.golf_view_presets import VIEWER_FOV_Y_RAD
from src.tools.native_viewer_export.backends._meshcat_page import MeshcatPage

pytestmark = pytest.mark.unit


class _FakePage:
    def __init__(self, has_camera: bool = True) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._has_camera = has_camera

    def evaluate(self, script: str, arg: Any = None) -> bool:
        self.calls.append((script, arg))
        return self._has_camera


def test_default_fov_is_the_shared_viewer_fov() -> None:
    page = MeshcatPage("http://localhost", 64, 48)
    assert page.fov_y_rad == VIEWER_FOV_Y_RAD


def test_apply_fov_sets_the_three_js_camera_in_degrees() -> None:
    page = MeshcatPage("http://localhost", 64, 48)
    fake = _FakePage()
    page._page = fake

    page.apply_fov()

    ((script, arg),) = fake.calls
    assert "camera.fov" in script and "updateProjectionMatrix" in script
    assert arg == pytest.approx(math.degrees(VIEWER_FOV_Y_RAD))


def test_apply_fov_fails_loudly_without_a_viewer_camera() -> None:
    page = MeshcatPage("http://localhost", 64, 48)
    page._page = _FakePage(has_camera=False)

    with pytest.raises(RuntimeError, match="camera"):
        page.apply_fov()


def test_fov_must_lie_in_the_open_interval() -> None:
    with pytest.raises(ValueError, match="fov_y_rad"):
        MeshcatPage("http://localhost", 64, 48, fov_y_rad=math.pi)
