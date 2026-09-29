"""Headless tests for the Video Analyzer ``MainWidget`` (issue #8883).

Before this PR, clicking the Video Analyzer tile opened a window whose
only content was a static ``QLabel("Video Analyzer (GUI placeholder)")``
— it never touched the tested ``SwingAnalyzer`` math in ``analyzer.py``.
These tests assert the fix: choosing a video and running analysis must
call ``SwingAnalyzer.analyze_video`` and render its result, and a failure
(bad file, missing MediaPipe) must produce an honest status/report
message instead of a crash or a silently blank screen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    """Fresh ``MainWidget``, cleaned up on teardown."""
    from src.tools.video_analyzer.gui import MainWidget

    main_widget = MainWidget()
    yield main_widget
    main_widget.cleanup()
    main_widget.deleteLater()


def _fake_video(tmp_path: Path) -> Path:
    """A file that exists on disk; the GUI layer never decodes it directly."""
    video = tmp_path / "swing.mp4"
    video.write_bytes(b"not a real container, just a path for the GUI layer")
    return video


def test_key_widgets_exist(widget) -> None:  # noqa: ANN001
    """The window exposes real controls, not just a static label."""
    assert widget.choose_button is not None
    assert widget.analyze_button is not None
    assert widget.report_text is not None
    assert widget.status_label is not None
    assert widget.path_label is not None


def test_analyze_disabled_until_video_chosen(widget) -> None:  # noqa: ANN001
    """The Analyze action is not available with nothing selected."""
    assert widget.analyze_button.isEnabled() is False


def test_set_video_path_enables_analyze(widget, tmp_path) -> None:  # noqa: ANN001
    video = _fake_video(tmp_path)
    widget.set_video_path(video)
    assert widget.analyze_button.isEnabled() is True
    assert str(video) in widget.path_label.text()


def test_set_video_path_rejects_missing_file(widget, tmp_path) -> None:  # noqa: ANN001
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(ValueError, match="not found"):
        widget.set_video_path(missing)
    assert widget.analyze_button.isEnabled() is False


def test_run_analysis_invokes_swing_analyzer(  # noqa: ANN001
    widget, tmp_path, monkeypatch
) -> None:
    """The synchronous core must call the real, tested analysis method."""
    import src.tools.video_analyzer.gui as gui_module
    from src.tools.video_analyzer.types import PostureMetrics

    video = _fake_video(tmp_path)
    widget.set_video_path(video)

    calls: list[Path] = []

    def _fake_analyze_video(self, path):  # noqa: ANN001, ANN202
        calls.append(Path(path))
        return PostureMetrics(head_stability=87.5)

    monkeypatch.setattr(gui_module.SwingAnalyzer, "analyze_video", _fake_analyze_video)

    metrics = widget.run_analysis()

    assert calls == [video]
    assert metrics is not None
    assert metrics.head_stability == pytest.approx(87.5)
    assert "87.5" in widget.report_text.toPlainText()
    assert "complete" in widget.status_label.text().lower()


def test_run_analysis_reports_honest_error_on_failure(  # noqa: ANN001
    widget, tmp_path, monkeypatch
) -> None:
    """A failure must be surfaced in the UI, never crash or blank out."""
    import src.tools.video_analyzer.gui as gui_module

    video = _fake_video(tmp_path)
    widget.set_video_path(video)

    def _boom(self, path):  # noqa: ANN001, ANN202
        raise RuntimeError("MediaPipe (>=0.10, Tasks API) is not installed")

    monkeypatch.setattr(gui_module.SwingAnalyzer, "analyze_video", _boom)

    result = widget.run_analysis()

    assert result is None
    assert "not installed" in widget.status_label.text()
    assert "not installed" in widget.report_text.toPlainText()


def test_run_analysis_without_video_prompts_selection(widget) -> None:  # noqa: ANN001
    """Clicking Analyze with nothing chosen never calls the analyzer."""
    result = widget.run_analysis()
    assert result is None
    assert "choose" in widget.status_label.text().lower()
