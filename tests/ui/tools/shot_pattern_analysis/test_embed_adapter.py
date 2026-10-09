"""Headless lifecycle checks for the optional PyQt tool host."""

from __future__ import annotations

import os
import json
from pathlib import Path
import sys
import time

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


def test_adapter_creates_and_cleans_up_widget_idempotently(qapp) -> None:  # noqa: ANN001
    try:
        import PyQt6.QtCore  # noqa: F401
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")

    from src.tools.shot_pattern_analysis._embed_adapter import (
        ShotPatternAnalysisEmbedAdapter,
    )

    adapter = ShotPatternAnalysisEmbedAdapter()
    widget = adapter.create_main_widget(None)
    assert adapter.create_main_widget(None) is widget
    adapter.cleanup()
    adapter.cleanup()


def test_cleanup_cancels_an_active_analysis_process(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    marker = tmp_path / "export-started"
    code = (
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('started'); time.sleep(30)"
    )
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "MPLBACKEND": "Agg",
    }
    started = widget.action_bar.start(
        "Test active export",
        lambda ctx: widget._run_command(
            [sys.executable, "-c", code], Path.cwd(), env, ctx
        ),
        on_finished=lambda _result: None,
        on_failed=lambda _message: None,
    )
    assert started
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert marker.exists(), "managed subprocess did not start"
    assert widget.cleanup(), "the cancelled process worker outlived cleanup"
    widget.deleteLater()
    qapp.processEvents()


def test_empty_output_folder_does_not_start_a_worker(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    widget.output_edit.clear()
    widget.run_analysis()
    assert not widget.action_bar.is_running, widget.status_label.text()
    assert "Choose an output folder" in widget.status_label.text()
    assert widget.cleanup()


def test_delivery_controls_expose_explicit_face_to_loft_coupling(qapp):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    assert widget.delivery_mode.currentData() == "fixed_loft"
    assert not widget.lie.isEnabled()
    assert not widget.shaft_lean.isEnabled()
    assert "Delivered loft is held fixed" in widget.delivery_assumption.text()
    widget.delivery_mode.setCurrentIndex(1)
    assert widget.delivery_mode.currentData() == "shaft_rotation"
    assert widget.lie.isEnabled()
    assert widget.shaft_lean.isEnabled()
    assert "Shaft Rotation Coupling" in widget.description_label.text()
    assert "rotates about the assumed shaft axis" in widget.delivery_assumption.text()
    assert widget.cleanup()


def test_club_preset_switches_to_the_illustrative_iron_inputs(qapp):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    widget.club_preset.setCurrentIndex(widget.club_preset.findData("seven_iron"))
    assert widget.club_speed.value() == pytest.approx(36.0)
    assert widget.loft.value() == pytest.approx(24.0)
    assert widget.attack_angle.value() == pytest.approx(-4.0)
    assert widget.lie.value() == pytest.approx(63.0)
    assert widget.clubhead_mass.value() == pytest.approx(0.272)
    assert "not measured golfer means" in widget.preset_assumption.text()
    widget.club_speed.setValue(37.5)
    assert widget.club_preset.currentData() == "seven_iron"
    assert "7-Iron (Modified)" in widget.preset_assumption.text()
    assert widget.club_speed.value() == pytest.approx(37.5)
    assert widget.cleanup()


def test_custom_club_controls_reach_shaft_rotation_cli(monkeypatch, qapp, tmp_path):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget, _ProcessOutput

    widget = MainWidget()
    widget.output_edit.setText(str(tmp_path))
    widget.club_preset.setCurrentIndex(widget.club_preset.findData("custom"))
    widget.club_speed.setValue(33.5)
    widget.loft.setValue(31.2)
    widget.attack_angle.setValue(-3.5)
    widget.clubhead_mass.setValue(0.315)
    widget.delivery_mode.setCurrentIndex(
        widget.delivery_mode.findData("shaft_rotation")
    )
    widget.lie.setValue(64.5)
    widget.shaft_lean.setValue(12.0)
    observed: dict[str, list[str]] = {}

    class Context:
        @staticmethod
        def raise_if_cancelled() -> None:
            return None

        @staticmethod
        def report(_fraction, _message) -> None:  # noqa: ANN001
            return None

    def run_command(command, _cwd, _env, _context):  # noqa: ANN001
        observed["command"] = command
        (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
        return _ProcessOutput(0, "")

    def start(_label, worker, **_callbacks):  # noqa: ANN001
        worker(Context())
        return True

    monkeypatch.setattr(MainWidget, "_run_command", staticmethod(run_command))
    monkeypatch.setattr(widget.action_bar, "start", start)
    widget.run_analysis()
    command = observed["command"]

    assert widget.club_preset.currentData() == "custom"
    assert command[command.index("--club-preset") + 1] == "custom"
    assert command[command.index("--delivery-mode") + 1] == "shaft_rotation"
    assert command[command.index("--club-speed-mps") + 1] == "33.5"
    assert command[command.index("--loft-deg") + 1] == "31.2"
    assert command[command.index("--attack-angle-deg") + 1] == "-3.5"
    assert command[command.index("--clubhead-mass-kg") + 1] == "0.315"
    assert command[command.index("--lie-deg") + 1] == "64.5"
    assert command[command.index("--shaft-lean-deg") + 1] == "12"
    assert widget.cleanup()


def test_unavailable_approach_scoring_keeps_gui_results_visible(qapp, tmp_path):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    patterns = {
        name: {
            "aimed_lateral_sd_m": 1.2,
            "aimed_target_hit_fraction": 0.5,
            "aimed_target_rmse_m": 2.1,
            "mean_carry_m": 195.3,
        }
        for name in ("Straight", "Draw", "Fade")
    }
    widget._completed(
        {
            "output_dir": tmp_path,
            "summary": {
                "patterns": patterns,
                "approach_scoring": {
                    "status": "unavailable",
                    "reason": "target carry is outside the supported baseline",
                },
                "config": {"loft_deg": 10.9},
            },
        }
    )

    assert widget.summary_table.rowCount() == 3
    assert widget.summary_table.item(0, 2).text() == "1.20 m"
    assert widget.summary_table.item(0, 6).text() == "Unavailable"
    assert "outside the supported baseline" in widget.scoring_label.text()
    assert widget.cleanup()


def test_gui_prefers_club_specific_course_scoring(qapp, tmp_path):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    patterns = {
        name: {
            "aimed_lateral_sd_m": 1.2,
            "aimed_target_hit_fraction": 0.5,
            "aimed_target_rmse_m": 2.1,
            "mean_carry_m": 195.3,
        }
        for name in ("Straight", "Draw", "Fade")
    }
    widget._completed(
        {
            "output_dir": tmp_path,
            "summary": {
                "patterns": patterns,
                "config": {"loft_deg": 24.0},
                "approach_scoring": {"status": "unavailable", "reason": "legacy"},
                "course_scoring": {
                    "status": "available",
                    "scenario": "Historical PGA approach to a centered circular green",
                    "patterns": {
                        name: {"mean_strokes_gained": 0.25}
                        for name in ("Straight", "Draw", "Fade")
                    },
                    "paired_benefit_vs_straight": {
                        "Draw": {"estimate": 0.1, "lower_95": 0.0, "upper_95": 0.2},
                        "Fade": {"estimate": -0.1, "lower_95": -0.2, "upper_95": 0.0},
                    },
                },
            },
        }
    )

    assert widget.summary_table.item(0, 6).text() == "+0.250"
    assert "centered circular green" in widget.scoring_label.text()
    assert "legacy" not in widget.scoring_label.text()
    assert widget.cleanup()


@pytest.mark.slow
def test_gui_runs_a_small_analysis_and_renders_the_report(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        pytest.skip("native Rust flight kernel is not available")
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    widget.resize(1200, 900)
    widget.output_edit.setText(str(tmp_path))
    widget.shot_count.setValue(2)
    widget.face_sd.setValue(2.0)
    widget.curve_scale.setValue(2.0)
    widget.delivery_mode.setCurrentIndex(1)
    widget.club_preset.setCurrentIndex(widget.club_preset.findData("seven_iron"))
    widget.shaft_lean.setValue(10.0)
    assert "Shaft Rotation Coupling" in widget.description_label.text()
    widget.run_analysis()
    deadline = time.monotonic() + 90
    while widget.action_bar.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.03)
    qapp.processEvents()
    assert not widget.action_bar.is_running, widget.status_label.text()
    assert widget.summary_table.rowCount() == 3
    assert all((tmp_path / name).is_file() for name in widget.figure_labels)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["face_sd_deg"] == 2.0
    assert summary["config"]["curve_scale"] == 2.0
    assert summary["config"]["delivery_mode"] == "shaft_rotation"
    assert summary["config"]["lie_deg"] == 63.0
    assert summary["config"]["shaft_lean_deg"] == 10.0
    assert summary["config"]["club_speed_mps"] == 36.0
    assert summary["config"]["loft_deg"] == 24.0
    assert summary["config"]["attack_angle_deg"] == -4.0
    assert summary["config"]["clubhead_mass_kg"] == 0.272
    # Numeric adjustments retain the selected club context for scoring.
    assert summary["config"]["club_id"] == "seven_iron"
    assert "+3°/+6°" in widget.description_label.text()
    assert widget.summary_table.columnCount() == 7
    assert widget.summary_table.item(0, 1).text() == "24.0°"
    assert widget.summary_table.item(0, 6) is not None
    assert "Not a Player-Score Prediction" in widget.scoring_label.text()
    assert widget.grab().save(str(tmp_path / "widget.png"))
    assert widget.cleanup()
