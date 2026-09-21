"""Tests for Tour Matching Viewer native viewer button (MV-05 #10481)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.tools.tour_matching_viewer.core import ReplayData

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _build_dummy_replay() -> ReplayData:
    time_s = np.linspace(0.0, 1.0, 5)
    q = np.zeros((5, 10))
    return ReplayData(
        time_s=time_s,
        coordinates=q,
        coordinate_names=tuple(f"q_{i}" for i in range(10)),
    )


def test_native_viewer_button_present(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    assert hasattr(widget, "_open_native_btn")
    assert widget._open_native_btn.text() == "Open Native…"
    widget.cleanup()


def test_native_viewer_button_no_replay_shows_info(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QMessageBox
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    with patch.object(QMessageBox, "information") as mock_info:
        widget._open_native_btn.click()
        mock_info.assert_called_once()
        assert "No candidate" in mock_info.call_args[0][2]
    widget.cleanup()


def test_native_viewer_button_launches_selected_backend(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QInputDialog, QMessageBox
    from src.shared.python.motion_matching.native_viewers import ViewerLaunchResult
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    widget._replay = _build_dummy_replay()

    with (
        patch.object(QInputDialog, "getItem", return_value=("meshcat", True)),
        patch(
            "src.shared.python.motion_matching.native_viewers.open_in_native_viewer",
            return_value=ViewerLaunchResult(
                success=True, backend="meshcat", url="http://127.0.0.1:7000/static/"
            ),
        ) as mock_open,
        patch.object(QMessageBox, "information"),
    ):
        widget._open_native_btn.click()
        mock_open.assert_called_once()
        args, kwargs = mock_open.call_args
        assert args[1] == "meshcat"
    widget.cleanup()


def test_native_viewer_button_shows_warning_on_unavailable(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QInputDialog, QMessageBox
    from src.shared.python.motion_matching.native_viewers import ViewerUnavailableError
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    widget._replay = _build_dummy_replay()

    with (
        patch.object(QInputDialog, "getItem", return_value=("gepetto", True)),
        patch(
            "src.shared.python.motion_matching.native_viewers.open_in_native_viewer",
            side_effect=ViewerUnavailableError("Gepetto not installed"),
        ),
        patch.object(QMessageBox, "warning") as mock_warn,
    ):
        widget._open_native_btn.click()
        mock_warn.assert_called_once()
        assert "Gepetto not installed" in mock_warn.call_args[0][2]
    widget.cleanup()
