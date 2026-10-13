"""Headless contracts for the legacy and matched study presenters."""

from __future__ import annotations

import inspect
import os
import sys

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
    return QApplication.instance() or QApplication([])


def test_gui_methods_fit_architecture_budget(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget
    from src.tools.shot_pattern_analysis.matched_gui import (
        MatchedStudyWidget,
        format_matched_course_scoring,
    )

    for method in (
        MainWidget._build_ui,
        MatchedStudyWidget._build_ui,
        format_matched_course_scoring,
    ):
        assert len(inspect.getsourcelines(method)[0]) <= 100
    assert len(inspect.signature(MainWidget._run_cli).parameters) <= 8


def test_widgets_retain_controls_and_initial_semantics(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    try:
        assert [widget.mode_tabs.tabText(i) for i in range(2)] == [
            "Independent Pattern Comparison",
            "Matched Delivery Study",
        ]
        assert (widget.shot_count.value(), widget.seed.value()) == (10000, 20261008)
        assert (
            widget.delivery_mode.currentData(),
            widget.club_preset.currentData(),
        ) == (
            "fixed_loft",
            "driver",
        )
        assert list(widget.figure_labels) == [
            "overhead_flight.png",
            "dispersion.png",
            "dispersion_equal_range.png",
        ]
        matched = widget.matched_study
        assert (matched.shot_count.value(), matched.seed.value()) == (10000, 20261009)
        assert (matched.match_tolerance.value(), matched.target_radius.value()) == (
            0.05,
            15.0,
        )
        assert matched.run_button.text() == "Run Matched Study"
        assert matched.previews.tabText(1) == "Raw Flight Trajectories"
        assert len(matched.noise_inputs) == 7
    finally:
        widget.cleanup()


def test_course_scoring_text_is_exact_for_conditional_driver(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.matched_gui import (
        format_matched_course_scoring,
    )

    scoring = {
        "status": "conditional_only",
        "club_id": "driver",
        "scenario": "Historical PGA tee shot on a flat hole",
        "hole_distance_m": 400.0,
        "fairway_half_width_m": 15.0,
        "paired_n": 8,
        "total_n": 10,
        "patterns": {
            "Straight": {"mean_strokes_gained": -0.2},
            "Draw": {"mean_strokes_gained": 0.1},
            "Fade": {"mean_strokes_gained": None},
        },
        "paired_benefit_vs_straight": {
            "Draw": {"estimate": 0.3, "lower_95": 0.1, "upper_95": 0.5},
            "Fade": {"estimate": None, "lower_95": None, "upper_95": None},
        },
    }
    assert format_matched_course_scoring(scoring) == (
        "Scenario: Historical PGA tee shot on a flat hole (400 m hole, 30 m fairway corridor).\n"
        "Common supported pairs: 8/10 (conditional on common supported indices only; full population unavailable due to unsupported terminal lies).\n"
        "Mean Strokes Gained: Straight: -0.200; Draw: +0.100; Fade: unavailable.\n"
        "Paired Difference vs. Straight (95% Monte Carlo Interval): Draw: +0.300 [+0.100, +0.500] strokes; Fade: unavailable.\n"
        "Caveats: Model-conditional historical PGA Tour benchmark (approximate baseline fit; not player-specific). "
        "Flat hypothetical geometry; carry landing is final lie without roll, hazards, or obstruction. "
        "Intervals describe Monte Carlo sampling only."
    )


def test_legacy_controls_emit_exact_cli_arguments(qapp, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget, _ProcessOutput

    widget = MainWidget()
    widget.output_edit.setText(str(tmp_path))
    widget.shot_count.setValue(7)
    widget.seed.setValue(23)
    widget.face_sd.setValue(1.5)
    widget.curve_scale.setValue(1.2)
    widget.delivery_mode.setCurrentIndex(
        widget.delivery_mode.findData("shaft_rotation")
    )
    widget.club_preset.setCurrentIndex(widget.club_preset.findData("custom"))
    widget.lie.setValue(64.5)
    widget.shaft_lean.setValue(12.0)
    widget.club_speed.setValue(33.5)
    widget.loft.setValue(31.2)
    widget.attack_angle.setValue(-3.5)
    widget.clubhead_mass.setValue(0.315)
    captured = []

    class Context:
        def raise_if_cancelled(self) -> None:
            return None

    def run_command(command, _root, _env, _ctx):  # noqa: ANN001
        captured.extend(command)
        (tmp_path / "summary.json").write_text("{}")
        return _ProcessOutput(0, "")

    def start(_label, worker, **_callbacks):  # noqa: ANN001
        worker(Context())
        return True

    monkeypatch.setattr(MainWidget, "_run_command", staticmethod(run_command))
    monkeypatch.setattr(widget.action_bar, "start", start)
    try:
        widget.run_analysis()
        assert captured == [
            sys.executable,
            "-m",
            "src.tools.shot_pattern_analysis",
            "--output",
            str(tmp_path),
            "--shots",
            "7",
            "--seed",
            "23",
            "--face-sd",
            "1.5",
            "--curve-scale",
            "1.2",
            "--delivery-mode",
            "shaft_rotation",
            "--lie-deg",
            "64.5",
            "--shaft-lean-deg",
            "12",
            "--club-speed-mps",
            "33.5",
            "--loft-deg",
            "31.2",
            "--attack-angle-deg",
            "-3.5",
            "--clubhead-mass-kg",
            "0.315",
            "--club-preset",
            "custom",
        ]
    finally:
        widget.cleanup()
