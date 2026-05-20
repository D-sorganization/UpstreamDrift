"""Tests for MuJoCoOffscreenRenderer temp-file handling.

Issue #2502: load_urdf_file() must use a unique temp filename rather than
the fixed '_temp_fixed_model.urdf', which races in parallel runs and fails
in read-only source directories.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest


def _make_qt_stubs() -> dict[str, ModuleType]:
    """Build minimal PyQt6 stubs so mujoco_viewer can be imported without Qt."""
    qt_mocks: dict[str, ModuleType] = {}
    for mod in [
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtOpenGLWidgets",
        "defusedxml",
        "defusedxml.ElementTree",
    ]:
        qt_mocks[mod] = MagicMock()
    # pyqtSignal must return a MagicMock when called (class-body decoration)
    qt_mocks["PyQt6.QtCore"].pyqtSignal = MagicMock(return_value=MagicMock())
    return qt_mocks


@pytest.fixture(scope="module")
def offscreen_renderer_class() -> Generator[type, None, None]:
    """Import MuJoCoOffscreenRenderer with PyQt6 stubbed out."""
    with patch.dict(sys.modules, _make_qt_stubs()):
        from src.tools.model_explorer.mujoco_viewer import MuJoCoOffscreenRenderer

        yield MuJoCoOffscreenRenderer


class TestIssue2502TempFileHandling:
    """MuJoCoOffscreenRenderer.load_urdf_file must use a unique temp filename."""

    def _mock_mujoco(self) -> MagicMock:
        mock = MagicMock()
        model = MagicMock()
        model.vis.global_.offwidth = 640
        model.vis.global_.offheight = 480
        mock.MjModel.from_xml_path.return_value = model
        mock.MjData.return_value = MagicMock()
        mock.Renderer.return_value = MagicMock()
        mock.MjvCamera.return_value = MagicMock()
        mock.MjvOption.return_value = MagicMock()
        return mock

    def test_no_fixed_temp_file_remains_after_success(
        self, offscreen_renderer_class, tmp_path
    ) -> None:
        """_temp_fixed_model.urdf must not exist in the source dir after load."""
        urdf_file = tmp_path / "model.urdf"
        urdf_file.write_text("<robot name='t'><link name='base'/></robot>")

        renderer = offscreen_renderer_class()
        with patch.dict(sys.modules, _make_qt_stubs()):
            import src.tools.model_explorer.mujoco_viewer as mv

            with (
                patch.object(mv, "mujoco", self._mock_mujoco()),
                patch.object(mv, "MUJOCO_AVAILABLE", True),
            ):
                renderer.load_urdf_file(str(urdf_file))

        assert not (tmp_path / "_temp_fixed_model.urdf").exists()

    def test_temp_filename_is_not_fixed_string(
        self, offscreen_renderer_class, tmp_path
    ) -> None:
        """The temp file passed to MjModel.from_xml_path must not be the fixed name."""
        urdf_file = tmp_path / "model.urdf"
        urdf_file.write_text("<robot name='t'><link name='base'/></robot>")

        renderer = offscreen_renderer_class()
        captured_names: list[str] = []

        def capture(path_str: str) -> NoReturn:
            captured_names.append(Path(path_str).name)
            raise RuntimeError("abort after capture")

        mock_mujoco = self._mock_mujoco()
        mock_mujoco.MjModel.from_xml_path.side_effect = capture

        with patch.dict(sys.modules, _make_qt_stubs()):
            import src.tools.model_explorer.mujoco_viewer as mv

            with (
                patch.object(mv, "mujoco", mock_mujoco),
                patch.object(mv, "MUJOCO_AVAILABLE", True),
                contextlib.suppress(RuntimeError, OSError),
            ):
                renderer.load_urdf_file(str(urdf_file))

        assert len(captured_names) == 1
        assert (
            captured_names[0] != "_temp_fixed_model.urdf"
        ), f"Expected a unique temp name; got {captured_names[0]!r}"
