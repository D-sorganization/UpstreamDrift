"""Tests for security fix in mujoco_viewer.py (issue #3302).

Validates that _launch_external_viewer uses secure_popen instead of raw
subprocess.Popen to prevent command injection vulnerabilities.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMuJoCoViewerSecurity:
    """Security contract tests for MuJoCoViewerWidget._launch_external_viewer."""

    @pytest.fixture
    def viewer_module(self):
        """Import viewer module with Qt stubs to avoid PyQt6 dependency."""
        qt_mocks = {}
        for mod_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
        ):
            mod = MagicMock()
            mod.__spec__ = MagicMock()
            qt_mocks[mod_name] = mod

        with patch.dict(sys.modules, qt_mocks):
            import src.tools.model_explorer.mujoco_viewer as mv

            yield mv

    @pytest.fixture
    def viewer_widget(self, viewer_module):
        """Return a MuJoCoViewerWidget instance with mocked parent."""
        with patch.object(
            viewer_module.MuJoCoViewerWidget, "__init__", lambda self, parent=None: None
        ):
            widget = viewer_module.MuJoCoViewerWidget.__new__(
                viewer_module.MuJoCoViewerWidget
            )
            widget._urdf_content = "<robot></robot>"
            widget._renderer = None
            return widget

    @patch("src.tools.model_explorer.mujoco_viewer.secure_popen")
    def test_launch_external_viewer_uses_secure_popen(
        self, mock_secure_popen, viewer_widget
    ) -> None:
        """_launch_external_viewer must delegate to secure_popen, not raw Popen."""
        mock_process = MagicMock()
        mock_secure_popen.return_value = mock_process

        viewer_widget._launch_external_viewer()

        mock_secure_popen.assert_called_once()
        args, kwargs = mock_secure_popen.call_args
        cmd = args[0]
        assert cmd[0] == sys.executable
        assert cmd[1] == "-c"
        assert "mujoco" in cmd[2]
        assert "mujoco.viewer.launch" in cmd[2]

    @patch("src.tools.model_explorer.mujoco_viewer.secure_popen")
    def test_launch_external_viewer_no_urdf_skips_popen(
        self, mock_secure_popen, viewer_widget
    ) -> None:
        """When no URDF content is set, secure_popen must not be called."""
        viewer_widget._urdf_content = ""

        viewer_widget._launch_external_viewer()

        mock_secure_popen.assert_not_called()

    @patch("src.tools.model_explorer.mujoco_viewer.secure_popen")
    def test_launch_external_viewer_command_structure(
        self, mock_secure_popen, viewer_widget
    ) -> None:
        """Verify the command passed to secure_popen is a list with expected structure."""
        viewer_widget._launch_external_viewer()

        assert mock_secure_popen.called
        args, _kwargs = mock_secure_popen.call_args
        cmd = args[0]
        assert isinstance(cmd, list)
        assert len(cmd) == 3
        assert cmd[0] == sys.executable
        assert cmd[1] == "-c"
        # The -c argument should contain the inline Python script
        assert "from_xml_path" in cmd[2]
        assert "mujoco.viewer.launch" in cmd[2]

    def test_no_raw_subprocess_popen_import(self) -> None:
        """The module must not import subprocess.Popen directly.

        Bandit B603/B604 flags raw subprocess.Popen calls. By using
        secure_popen from the shared security module, we avoid these
        warnings and gain executable validation.
        """
        import ast

        source = (
            Path(__file__).parents[4]
            / "src"
            / "tools"
            / "model_explorer"
            / "mujoco_viewer.py"
        ).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "subprocess", (
                        "mujoco_viewer.py must not import subprocess directly; "
                        "use secure_popen from src.shared.python.security.secure_subprocess"
                    )
            if isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess", (
                    "mujoco_viewer.py must not import from subprocess; "
                    "use secure_popen from src.shared.python.security.secure_subprocess"
                )