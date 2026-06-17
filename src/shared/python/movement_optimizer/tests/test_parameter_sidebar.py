# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the ParameterSidebar façade and state methods."""

from __future__ import annotations

import pytest

from movement_optimizer.gui.parameter_sidebar import ParameterSidebar
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory.result import ProgressReport


@pytest.fixture
def sidebar(qapp) -> ParameterSidebar:
    return ParameterSidebar()


def test_action_handlers_connect_and_emit(sidebar) -> None:
    fired: list[str] = []
    names = [
        "optimize_current",
        "optimize_both",
        "cancel_requested",
        "export_requested",
        "reset_requested",
        "save_solution_requested",
        "load_solution_requested",
        "export_video_requested",
        "export_plots_requested",
        "export_excel_requested",
        "add_comparison_requested",
        "compare_trials_requested",
        "clear_comparison_requested",
    ]
    sidebar.connect_action_handlers({name: (lambda n=name: fired.append(n)) for name in names})
    for name in names:
        getattr(sidebar, name).emit()
    assert set(fired) == set(names)


def test_is_3d_mode_reflects_combo(sidebar) -> None:
    sidebar.model_combo.setCurrentIndex(0)
    assert not sidebar.is_3d_mode()
    sidebar.model_combo.setCurrentIndex(1)
    assert sidebar.is_3d_mode()


def test_optimizing_idle_and_body_model(sidebar) -> None:
    sidebar.show_optimizing()
    assert not sidebar.opt_btn.isEnabled()
    assert sidebar.cancel_btn.isEnabled()
    sidebar.show_idle()
    assert sidebar.opt_btn.isEnabled()
    body = sidebar.get_body_model()
    assert isinstance(body, BodyModel)


def test_reset_defaults_restores_known_values(sidebar) -> None:
    sidebar.mass_slider.set_value(99.0)
    sidebar.reset_defaults()
    assert sidebar.mass_slider.value() == pytest.approx(75.0, abs=1.0)


def test_optimization_params_and_multipliers(sidebar) -> None:
    bar, dur, smooth = sidebar.get_optimization_params()
    assert bar > 0 and dur > 0 and smooth >= 0
    multipliers = sidebar.get_segment_multipliers()
    assert set(multipliers) == {"lower_leg", "upper_leg", "torso"}


def test_result_progress_and_cancel_labels(sidebar) -> None:
    sidebar.set_progress_done("1.5s", 100)
    assert sidebar.progress.value() == 100
    sidebar.set_cancelled()
    assert "Cancelled" in sidebar.prog_label.text()
    sidebar.set_stall_message("slow")
    assert not sidebar.stall_label.isHidden()
    sidebar.clear_stall_message()
    assert sidebar.stall_label.isHidden()
    sidebar.set_result_label("done")
    assert sidebar.result_label.text() == "done"


def test_post_run_buttons_and_comparison_payloads(sidebar) -> None:
    sidebar.enable_post_run_buttons()
    assert sidebar.export_btn.isEnabled()
    assert sidebar.add_compare_btn.isEnabled()

    params = sidebar.get_body_params_dict()
    assert set(params) == {"body_mass", "height", "seg_multipliers"}
    trial, bar = sidebar.get_comparison_trial_data()
    assert isinstance(trial, dict) and bar > 0
    bar2, body_params = sidebar.get_comparison_context()
    assert bar2 == bar and body_params == trial


@pytest.mark.parametrize("available", [True, False])
def test_availability_toggles(sidebar, available: bool) -> None:
    sidebar.set_comparison_available(available)
    assert sidebar.compare_btn.isEnabled() == available
    sidebar.set_clear_comparison_available(available)
    assert sidebar.clear_compare_btn.isEnabled() == available
    sidebar.set_cancellation_available(available)
    assert sidebar.cancel_btn.isEnabled() == available


def test_set_cancelling_disables_run_buttons(sidebar) -> None:
    sidebar.set_cancelling()
    assert not sidebar.opt_btn.isEnabled()
    assert not sidebar.both_btn.isEnabled()
    assert "Cancel" in sidebar.cancel_btn.text()


def _report(**overrides) -> ProgressReport:
    base = {
        "iteration": 300,
        "cost": 12.5,
        "best_cost": 10.0,
        "improvement_pct": 1.2,
        "elapsed_s": 5.0,
        "cost_history": [100.0, 50.0, 10.0],
    }
    base.update(overrides)
    return ProgressReport(**base)


def test_update_progress_covers_normal_stalled_and_long(sidebar) -> None:
    sidebar.update_progress(_report())  # normal "Converging"
    assert "Converging" in sidebar.prog_label.text()
    assert sidebar.stall_label.isHidden()

    sidebar.update_progress(_report(is_stalled=True, stall_reason="no progress"))
    assert not sidebar.stall_label.isHidden()

    sidebar.update_progress(_report(elapsed_s=200.0))  # long-running warning branch
    assert not sidebar.stall_label.isHidden()

    sidebar.update_progress(_report(iteration=10))  # "Exploring" + short elapsed
    assert "Exploring" in sidebar.prog_label.text()
