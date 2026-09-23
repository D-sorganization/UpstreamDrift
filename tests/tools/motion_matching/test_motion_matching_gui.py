"""Tests for the Motion Matching launcher tile GUI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6", reason="motion matching tile needs PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.motion_matching.gui import (  # noqa: E402
    WINDOW_TITLE,
    MotionMatchingWidget,
    get_dockable_ui,
)
from src.tools.motion_matching import pipeline  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the test session."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def widget() -> MotionMatchingWidget:
    return MotionMatchingWidget()


def test_widget_construction_and_tabs(widget: MotionMatchingWidget) -> None:
    assert widget.tabs.count() == 4
    tab_titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert "Matching" in tab_titles
    assert "Downswing experiment" in tab_titles
    assert "MJX" in tab_titles
    assert "Club-Only" in tab_titles


def test_stages_group_and_mutual_exclusion(widget: MotionMatchingWidget) -> None:
    assert widget.stages_group is not None
    assert widget.free_wrists.isChecked() is False
    assert widget.bound_wrists.isChecked() is False

    # Check free_wrists, then bound_wrists
    widget.free_wrists.setChecked(True)
    assert widget.free_wrists.isChecked() is True
    assert widget.bound_wrists.isChecked() is False

    widget.bound_wrists.setChecked(True)
    assert widget.bound_wrists.isChecked() is True
    assert widget.free_wrists.isChecked() is False

    widget.fit_closure.setChecked(True)
    widget.zmp_filter.setChecked(True)
    widget.shooting_fit.setValue(4)
    widget.shooting_gain.setValue(0.65)

    req = widget.request()
    assert req.bound_wrists is True
    assert req.free_wrists is False
    assert req.fit_closure is True
    assert req.zmp_filter is True
    assert req.shooting_fit == 4
    assert req.shooting_gain == 0.65


def test_matching_tab_backend_selection(widget: MotionMatchingWidget) -> None:
    # Default is mujoco and physical
    req = widget.request()
    assert req.backend == "mujoco"
    assert req.step_mode == "physical"

    # Select pink and projection
    widget.backend.setCurrentText("pink")
    widget.step_mode.setCurrentText("projection")

    req2 = widget.request()
    assert req2.backend == "pink"
    assert req2.step_mode == "projection"


def test_downswing_experiment_request_binding(
    widget: MotionMatchingWidget, tmp_path: Path
) -> None:
    widget.exp_run_dir.setText(str(tmp_path))
    widget.exp_name.setText("custom_test")
    widget.exp_cutoff.setValue(15.0)
    widget.exp_omega.setValue(115.0)
    widget.exp_feedforward.setValue(0.85)
    widget.exp_balance.setChecked(False)

    exp_req = widget.experiment_request()
    assert exp_req.run == Path(tmp_path)
    assert exp_req.name == "custom_test"
    assert exp_req.cutoff_hz == 15.0
    assert exp_req.omega == 115.0
    assert exp_req.feedforward == 0.85
    assert exp_req.balance is False

    cmd = widget.experiment_command()
    assert "--name" in cmd and "custom_test" in cmd
    assert "--no-balance" in cmd


def test_mjx_commands_binding(widget: MotionMatchingWidget, tmp_path: Path) -> None:
    widget.mjx_run_dir.setText(str(tmp_path))
    opt_file = tmp_path / "opt.npz"
    widget.mjx_ref_path.setText(str(opt_file))

    export_cmd = widget.mjx_export_command()
    assert export_cmd == pipeline.export_mjx_command(tmp_path)

    val_cmd = widget.mjx_validate_command()
    assert val_cmd == pipeline.validate_reference_command(tmp_path, opt_file)


def test_get_dockable_ui() -> None:
    window = get_dockable_ui()
    assert window.windowTitle() == WINDOW_TITLE
    assert isinstance(window.centralWidget(), MotionMatchingWidget)


def test_engine_selection_from_plant_registry(widget: MotionMatchingWidget) -> None:
    """MS-82: Engine selection combo lists engines from plant registry."""
    items = [widget.backend.itemText(i) for i in range(widget.backend.count())]
    # mujoco must be in the list, along with other registered plant backends
    assert "mujoco" in items
    assert len(items) >= 2


def test_results_pane_shows_movies_and_metrics(
    widget: MotionMatchingWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MS-82: Results pane shows QMovie playback, 5 metrics, badge, and buttons."""
    # Create fake receipt and playback GIFs
    receipt = {
        "capture": "driver",
        "club": {"name": "driver"},
        "backend": "mujoco",
        "address": {
            "calibrated": {
                "marker_rms_m": 0.008,
                "centre_of_mass": {"inside_support_polygon": True},
            }
        },
        "ik": {
            "marker_rms_m": 0.022,
            "constrained_ik": {"is_qualified": True, "all_frames_converged": True},
        },
        "dynamics": {
            "root_tracking_rms_m": 0.028,
            "inside_support_polygon_fraction": 0.92,
            "backswing_to_1s": {"root_error_max_m": 0.005},
        },
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "ik_playback.gif").write_bytes(b"GIF89a")
    (tmp_path / "tracking_playback.gif").write_bytes(b"GIF89a")

    # Point request to tmp_path using monkeypatch so it is cleanly reverted
    monkeypatch.setattr(
        pipeline.MatchRequest, "output_dir", property(lambda self: tmp_path)
    )
    req = pipeline.MatchRequest(capture="driver", club="driver")
    widget._request = req

    widget.tabs.setCurrentIndex(0)
    widget._on_finished(0)

    # Check metrics
    metrics = widget.metrics_values()
    assert metrics["full_capture_ik_rms_mm"] == 22.0
    assert metrics["address_marker_rms_mm"] == 8.0
    assert metrics["backswing_root_error_max_mm"] == 5.0
    assert metrics["whole_run_root_rms_mm"] == 28.0
    assert metrics["inside_support_polygon_fraction"] == 0.92

    # Check acceptance badge
    assert widget.acceptance_badge.text() in ("PASSED", "QUALIFIED")

    # Check QMovie playback loaded
    assert widget.ik_movie is not None
    assert widget.tracking_movie is not None

    # Check navigation action buttons exist
    assert widget.open_browser_btn is not None
    assert widget.open_viewer_btn is not None
    assert widget.open_browser_btn.text() == "Open in Results Browser"
    assert widget.open_viewer_btn.text() == "Open in Viewer"

    # Navigation button triggers
    widget.open_browser_btn.click()
    assert widget._browser_window is not None
    widget._browser_window.close()

    widget.open_viewer_btn.click()
    assert widget._viewer_window is not None
    widget._viewer_window.close()

    # Cleanup
    widget.cleanup()
    assert widget.ik_movie is None
    assert widget.tracking_movie is None


def test_step_failure_updates_ui(widget: MotionMatchingWidget) -> None:
    """Non-zero exit code updates results label with failure message."""
    widget.tabs.setCurrentIndex(0)
    widget._on_finished(2)
    assert "Step failed with exit code 2" in widget.results.text()


def test_extract_five_metrics_and_acceptance_logic() -> None:
    """Verify verdict transitions for rejected and unclassified receipts."""
    # Qualified but unconverged -> REJECTED
    summary_unconverged = {
        "is_qualified": True,
        "all_frames_converged": False,
        "full_capture_ik_rms_mm": 12.0,
    }
    metrics, verdict = pipeline.extract_five_metrics_and_acceptance(summary_unconverged)
    assert verdict == "REJECTED"
    assert metrics["full_capture_ik_rms_mm"] == 12.0

    # Unqualified -> REJECTED
    summary_rejected = {
        "is_qualified": False,
        "all_frames_converged": True,
    }
    _, verdict2 = pipeline.extract_five_metrics_and_acceptance(summary_rejected)
    assert verdict2 == "REJECTED"

    # None -> UNCLASSIFIED
    summary_empty = {}
    _, verdict3 = pipeline.extract_five_metrics_and_acceptance(summary_empty)
    assert verdict3 == "UNCLASSIFIED"


def test_neural_assisted_controls_initialization(widget: MotionMatchingWidget) -> None:
    """Verify Neural-Assisted Motion Matching group and controls (NM-11, #10626)."""
    assert widget.neural_group is not None
    assert widget.neural_mode.count() == 3
    assert widget.neural_mode.currentText() == "Classical Only"
    assert widget.allow_classical_fallback.isChecked() is True
    assert widget.neural_model_selector.count() >= 3
    model_ids = [
        widget.neural_model_selector.itemText(i)
        for i in range(widget.neural_model_selector.count())
    ]
    assert "driven_double_pendulum" in model_ids


def test_update_neural_metrics_display(widget: MotionMatchingWidget) -> None:
    """Verify neural badge, empirical confidence, and time breakdown formatting (NM-11, #10626)."""
    # 1. Preview mode
    widget.update_neural_metrics(
        status="NEURAL_ACCEPTED",
        is_preview=True,
        confidence=0.885,
        t_neural_s=0.015,
        t_polish_s=0.045,
        t_total_s=0.060,
    )
    assert "PREVIEW" in widget.neural_status_badge.text()
    assert "0.885" in widget.metric_neural_confidence.text()
    assert "15.0ms" in widget.metric_time_breakdown.text()

    # 2. Verified mode
    widget.update_neural_metrics(
        status="VERIFIED",
        is_preview=False,
        confidence=0.942,
        t_neural_s=0.020,
        t_polish_s=0.080,
        t_total_s=0.100,
    )
    assert widget.neural_status_badge.text() == "VERIFIED"
    assert "0.942" in widget.metric_neural_confidence.text()
    assert "20.0ms" in widget.metric_time_breakdown.text()

    # 3. Classical fallback
    widget.update_neural_metrics(
        status="CLASSICAL_FALLBACK",
        is_preview=False,
        confidence=0.450,
        t_neural_s=0.020,
        t_polish_s=0.0,
        t_total_s=0.150,
    )
    assert widget.neural_status_badge.text() == "CLASSICAL FALLBACK"
    assert "0.450" in widget.metric_neural_confidence.text()
