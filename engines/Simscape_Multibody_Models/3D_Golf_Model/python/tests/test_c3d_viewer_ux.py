import os
import sys
import typing
from unittest.mock import MagicMock, call, patch

import pytest

# Gracefully skip if PyQt6 is not installed (e.g. in CI environments)
try:
    import matplotlib  # noqa: F401
    import matplotlib.artist  # noqa: F401
    import matplotlib.figure  # noqa: F401

    # Import dependencies to prevent reloading issues when patching sys.modules
    import numpy  # noqa: F401
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
except ImportError:
    pytest.skip(
        "Required packages (PyQt6, matplotlib, numpy) not installed",
        allow_module_level=True,
    )


# Ensure QApplication exists
@pytest.fixture(scope="session")
def qapp() -> typing.Generator[QApplication, None, None]:
    """Fixture that ensures a QApplication exists for the test session."""
    instance = QApplication.instance()
    if instance is None:
        app = QApplication(sys.argv)
    else:
        app = typing.cast(QApplication, instance)
    yield app


def test_c3d_viewer_open_file_ux(qapp: QApplication) -> None:
    """
    Test that opening a file triggers the expected UX behaviors
    (wait cursor, status bar update).
    """
    # Mock ezc3d in sys.modules throughout the test
    with patch.dict(sys.modules, {"ezc3d": MagicMock()}):
        from apps.c3d_viewer import C3DViewerMainWindow

        window = C3DViewerMainWindow()

        # We want to verify status bar messages.
        # QMainWindow.statusBar() returns the QStatusBar widget.
        real_status_bar = window.statusBar()
        assert real_status_bar is not None

        # Use patch.object to spy on showMessage
        with patch.object(real_status_bar, "showMessage") as mock_show_message:
            # Mock the file dialog
            test_path = "/path/to/test.c3d"
            with patch(
                "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(test_path, "C3D files (*.c3d)"),
            ):
                # Mock ezc3d
                # Since ezc3d is mocked in sys.modules, patch("ezc3d.c3d") should work if we target the mock
                # But cleaner is to mock the return value of ezc3d.c3d if we knew the structure.
                # Since we don't know if patch("ezc3d.c3d") works when ezc3d is a MagicMock in sys.modules
                # (it should work if the mock has attributes), let's try.
                # Actually, patch string imports the module. sys.modules has it.
                with patch("ezc3d.c3d") as mock_c3d:
                    # Minimal mock data
                    mock_data = MagicMock()
                    mock_data.__getitem__.side_effect = lambda k: {
                        "data": {
                            "points": MagicMock(shape=(4, 1, 1)),
                            "analogs": MagicMock(shape=(1, 1, 1)),
                        },
                        "parameters": {
                            "POINT": {
                                "LABELS": {"value": []},
                                "UNITS": {"value": [""]},
                                "RATE": {"value": [1.0]},
                            },
                            "ANALOG": {
                                "LABELS": {"value": []},
                                "RATE": {"value": [1.0]},
                                "UNITS": {"value": []},
                            },
                            "TRIAL": {},
                        },
                    }.get(k, {})
                    mock_c3d.return_value = mock_data

                    with patch(
                        "PyQt6.QtWidgets.QApplication.setOverrideCursor"
                    ) as mock_set_cursor:
                        with patch(
                            "PyQt6.QtWidgets.QApplication.restoreOverrideCursor"
                        ) as mock_restore_cursor:
                            window.open_c3d_file()

                            # Verify basic execution
                            assert mock_c3d.called

                            # Verify Cursor UX
                            mock_set_cursor.assert_called_once_with(
                                Qt.CursorShape.WaitCursor
                            )
                            mock_restore_cursor.assert_called_once()

                            # Verify Status Bar UX
                            # calls: "Loading test.c3d...", "Loaded test.c3d successfully."
                            assert mock_show_message.call_count == 2

                            filename = os.path.basename(test_path)
                            expected_calls = [
                                call(f"Loading {filename}..."),
                                call(f"Loaded {filename} successfully."),
                            ]
                            mock_show_message.assert_has_calls(expected_calls)
