"""MeshCat pages use the shared viewer field of view (NV-9, #11697)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.shared.python.golf_view_presets import VIEWER_FOV_Y_RAD
from src.tools.native_viewer_export.backends._meshcat_page import (
    PAGE_LOAD_TIMEOUT_MS,
    MeshcatPage,
)

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


class _FakeNavPage:
    def __init__(self) -> None:
        self.gotos: list[tuple[str, float]] = []

    def goto(self, url: str, timeout: float) -> None:
        self.gotos.append((url, timeout))


def test_navigation_uses_the_page_load_timeout() -> None:
    """A loaded host (load 25+) needs longer than Playwright's implicit 30 s."""
    page = MeshcatPage("http://localhost:7000", 64, 48)
    assert page.load_timeout_ms == PAGE_LOAD_TIMEOUT_MS >= 120_000
    fake = _FakeNavPage()
    page._page = fake

    page.navigate()

    assert fake.gotos == [("http://localhost:7000", PAGE_LOAD_TIMEOUT_MS)]


def test_page_load_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="load_timeout_ms"):
        MeshcatPage("http://localhost", 64, 48, load_timeout_ms=0)


def test_fov_must_lie_in_the_open_interval() -> None:
    with pytest.raises(ValueError, match="fov_y_rad"):
        MeshcatPage("http://localhost", 64, 48, fov_y_rad=math.pi)
