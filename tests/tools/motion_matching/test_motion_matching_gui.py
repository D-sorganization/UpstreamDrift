"""Tests for the Motion Matching launcher tile GUI."""

from __future__ import annotations

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
    assert widget.tabs.count() == 3
    tab_titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert "Matching" in tab_titles
    assert "Downswing experiment" in tab_titles
    assert "MJX" in tab_titles


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
