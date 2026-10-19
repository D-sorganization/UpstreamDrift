"""Tests for Model Explorer Results Playback UI (#12996).

Verifies that loading results:
1. Exposes a playback bar with play/pause, time readout, and slider range [0, frame_count - 1].
2. Scrubbing moves the playback cursor and updates viewport marker positions to source.markers[index].
3. Detects the real engine from the archive (not the literal "engine" default).
4. Enables video export with the detected engine.
5. Preserves postcondition 0 <= current_frame < frame_count.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="Model Explorer needs PyQt6")

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from src.tools.model_explorer.gui import MainWidget  # noqa: E402
from src.tools.model_explorer.results_session import ResultsSession  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.ui]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["explorer"])


@pytest.fixture
def explorer(qapp: QApplication) -> Generator[MainWidget, None, None]:
    widget = MainWidget()
    widget.resize(1000, 700)
    yield widget
    widget.cleanup()
    widget.deleteLater()


@pytest.fixture(autouse=True)
def boxes() -> Generator[list[str], None, None]:
    shown: list[str] = []

    def _record(*args: Any, **_kwargs: Any) -> int:
        shown.append(" ".join(str(a) for a in args[1:]))
        return int(QMessageBox.StandardButton.Ok)

    with (
        patch.object(QMessageBox, "warning", _record),
        patch.object(QMessageBox, "critical", _record),
        patch.object(QMessageBox, "information", _record),
    ):
        yield shown


@pytest.fixture
def small_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "drake_swing_results.npz"
    time_s = np.array([0.0, 0.033, 0.066, 0.099, 0.132], dtype=float)
    model_markers = np.zeros((5, 4, 3), dtype=float)
    for i in range(5):
        model_markers[i, :, :] = float(i + 1) * 0.1
    target_markers = model_markers * 0.95

    np.savez(
        archive,
        time_s=time_s,
        model_markers_m=model_markers,
        target_markers_m=target_markers,
        engine=np.array("drake"),
    )
    return archive


def test_load_results_shows_playback_bar_and_scrubs_viewport(
    explorer: MainWidget, small_archive: Path
) -> None:
    """Failing test first: after load_results, playback bar is exposed and scrubs viewport."""
    assert explorer.load_results(small_archive) is True

    # 1. Playback bar exposed with slider range [0, frame_count - 1]
    assert hasattr(explorer, "playback_bar")
    bar = explorer.playback_bar
    assert bar is not None
    assert bar.isEnabled()
    assert (bar.slider.minimum(), bar.slider.maximum()) == (0, 4)
    assert bar.slider.value() == 0

    # 2. Viewport exposes marker_positions matching source.markers[0]
    session = explorer._results
    assert session is not None
    source = session._source
    assert hasattr(source, "markers")
    np.testing.assert_allclose(explorer.viewport.marker_positions, source.markers[0])

    # 3. Moving slider calls scrub_results and updates viewport marker positions
    bar.slider.setValue(2)
    assert explorer.current_result_frame == 2
    np.testing.assert_allclose(explorer.viewport.marker_positions, source.markers[2])

    # 4. Time readout driven by ResultsSession.current_time_s
    assert "0.066" in bar.time_label.text()
    assert "3/5" in bar.time_label.text()

    # 5. Play / pause controls and timer advance
    bar.play_button.click()
    assert bar.play_button.text() == "Pause"
    bar._advance()
    assert explorer.current_result_frame == 3
    np.testing.assert_allclose(explorer.viewport.marker_positions, source.markers[3])
    bar.play_button.click()
    assert bar.play_button.text() == "Play"


def test_load_results_detects_real_engine_from_archive(
    explorer: MainWidget, small_archive: Path
) -> None:
    """Archive engine is detected and passed to ResultsSession."""
    assert explorer.load_results(small_archive) is True
    session = explorer._results
    assert session is not None
    assert session.engine == "drake"
    assert "Drake" in session.title


def test_smoke_load_scrub_and_export_video(
    explorer: MainWidget, small_archive: Path, tmp_path: Path
) -> None:
    """Smoke test: load, scrub, and export video on a small archive."""
    assert explorer.load_results(small_archive) is True
    explorer.scrub_results(1)
    assert explorer.current_result_frame == 1

    out_mp4 = tmp_path / "smoke_exported.mp4"
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(out_mp4), "MP4 Video (*.mp4)")
    ):
        out = explorer.export_results_video()

    assert out == out_mp4
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0


def test_postcondition_bounds_enforced(
    explorer: MainWidget, small_archive: Path
) -> None:
    """Postcondition: 0 <= current_frame < frame_count is enforced."""
    assert explorer.load_results(small_archive) is True
    frames = explorer.results_frame_count
    assert frames == 5

    with pytest.raises(ValueError, match="outside"):
        explorer.scrub_results(frames)

    with pytest.raises(ValueError, match="outside"):
        explorer.scrub_results(-1)

    assert 0 <= explorer.current_result_frame < frames
