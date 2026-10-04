"""Desktop research controls stay responsive during job submission."""

import threading
import time
import pytest

pytestmark = pytest.mark.unit


def test_refit_dialog_uses_shared_session_without_blocking_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from src.tools.necromatcher.refit_dialog import ResearchRefitDialog

    app = QApplication.instance() or QApplication([])
    entered, release = threading.Event(), threading.Event()
    ticks = []
    submissions = []
    record = {
        "run_id": "a" * 32,
        "source_fit_id": "source",
        "new_fit_id": "new",
        "status": "running",
        "acceptance": "partial",
        "blockers": [],
        "message": "Computing",
    }

    class Session:
        def submit(self, source, identity, options):
            submissions.append((source, identity, options))
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Submission not released")
            return record

        def view(self, run):
            return record

        def cancel(self, run):
            record.update(status="cancelled", acceptance="interrupted")
            return record

    dialog = ResearchRefitDialog(
        "source",
        {
            "frame_indices": [0, 2],
            "coordinate_order": ["hip"],
            "coordinate_units": ["rad"],
            "recorded_options": None,
        },
        Session(),
    )
    dialog.identity.setText("new")
    dialog.scales.setText("1")
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        dialog.start.click()
        assert entered.wait(2)
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            app.processEvents()
        assert ticks
        release.set()
        deadline = time.monotonic() + 2
        while dialog.run is None and time.monotonic() < deadline:
            app.processEvents()
        assert dialog.run is not None
        assert submissions[0][:2] == ("source", "new")
        assert dialog.status.text().startswith("running · partial")
        dialog.cancel.click()
        assert dialog.status.text().startswith("cancelled · interrupted")
    finally:
        release.set()
        dialog.cleanup()
        timer.stop()


def _recipe_plan():
    from dataclasses import asdict
    from src.shared.python.motion_matching.historical_fit import ImageFitConfig
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    config = ImageFitConfig(
        max_iterations=30,
        prior_weight=0.001,
        smoothness_weight=0.0001,
        constraint_options=ConstraintOptions(
            GroundPlane((0.0, 0.0, 1.0), 0.0),
            1000,
            1000,
            1000,
            0.01,
            0.1,
            0.01,
            ("heel_r",),
        ),
        interior_fractions=(0.25, 0.5, 0.75),
        coordinate_bounds=(("hip", -1.0, 1.0),),
        initialization_policy="authored_range_project_zero_slopes",
    )
    return config, {
        "frame_indices": list(range(9)),
        "coordinate_order": ["hip"],
        "coordinate_units": ["rad"],
        "baseline_config": asdict(config),
        "recorded_options": {
            "frame_indices": [2, 4, 6],
            "coordinate_scales": [1],
            "knot_count": 3,
            "config": {"prior_weight": 99},
        },
        "preserved_spline": {
            "available": True,
            "knot_count": 4,
            "source_interval": [10.0, 12.0],
            "reason": "Verified",
        },
    }


def _recipe_dialog(monkeypatch, plan):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.refit_dialog import ResearchRefitDialog

    app = QApplication.instance() or QApplication([])
    return app, ResearchRefitDialog("source", plan, object())


def test_refit_preserves_complete_baseline_recipe_when_editing_weights(monkeypatch):
    from dataclasses import replace

    config, plan = _recipe_plan()
    app, dialog = _recipe_dialog(monkeypatch, plan)
    dialog.fields["prior_weight"].setText("0.002")
    options = dialog._options()
    assert options.config == replace(config, prior_weight=0.002)
    assert options.initialization_source == "sampled_parent"
    assert "heel_r" in dialog.recipe_summary.text()
    assert "3" in dialog.recipe_summary.text()
    dialog.cleanup()


def test_refit_resume_uses_strict_saved_domain_and_full_interval(monkeypatch):
    from dataclasses import replace

    config, plan = _recipe_plan()
    app, dialog = _recipe_dialog(monkeypatch, plan)
    dialog.initialization.setCurrentIndex(1)
    assert not dialog.fields["knot_count"].isEnabled()
    assert not dialog.frames.isEnabled()
    assert dialog.frames.text() == "0, 2, 4, 6, 8"
    options = dialog._options()
    assert options.initialization_source == "preserved_spline"
    assert options.operation == "fit"
    assert options.knot_count == 4
    assert options.frame_indices == (0, 2, 4, 6, 8)
    assert options.config == replace(config, initialization_policy="strict")
    dialog.initialization.setCurrentIndex(0)
    assert dialog.fields["knot_count"].isEnabled()
    assert dialog.fields["knot_count"].text() == "3"
    assert dialog.frames.text() == "2, 4, 6"
    dialog.cleanup()


def test_refit_lossless_seed_is_explicit_strict_and_does_not_replace_recipe(
    monkeypatch,
):
    from dataclasses import replace

    config, plan = _recipe_plan()
    app, dialog = _recipe_dialog(monkeypatch, plan)
    try:
        index = dialog.initialization.findData("restricted_spline")
        assert index >= 0
        dialog.initialization.setCurrentIndex(index)
        with pytest.raises(ValueError, match="reviewed window"):
            dialog._options()
        dialog._inherited_scope = {"first_frame": 2, "end_exclusive_frame": 7}
        options = dialog._options()
        assert options.operation == "restrict_initialization"
        assert options.initialization_source == "restricted_spline"
        assert options.frame_indices == (2, 4, 6)
        assert options.knot_count == 3
        assert options.config == replace(config, initialization_policy="strict")
        assert dialog.frames.isEnabled()
    finally:
        dialog.cleanup()


def test_refit_unavailable_resume_is_disabled_and_explains_reason(monkeypatch):
    _, plan = _recipe_plan()
    plan["preserved_spline"] = {"available": False, "reason": "No saved spline"}
    app, dialog = _recipe_dialog(monkeypatch, plan)
    assert not dialog.initialization.model().item(1).isEnabled()
    assert "No saved spline" in dialog.recipe_summary.text()
    dialog.initialization.setCurrentIndex(1)
    with pytest.raises(ValueError, match="No saved spline"):
        dialog._options()
    dialog.cleanup()


def test_refit_recorded_config_fallback_retains_full_recipe(monkeypatch):
    from dataclasses import asdict

    config, plan = _recipe_plan()
    del plan["baseline_config"]
    plan["recorded_options"]["config"] = asdict(config)
    app, dialog = _recipe_dialog(monkeypatch, plan)
    assert dialog._options().config == config
    dialog.cleanup()


def test_refit_rejects_malformed_baseline_instead_of_silent_fallback(monkeypatch):
    _, plan = _recipe_plan()
    plan["baseline_config"] = {"constraint_options": {"ground": "invalid"}}
    with pytest.raises(ValueError, match="ground"):
        _recipe_dialog(monkeypatch, plan)


def test_refit_resume_retains_edited_training_sample_ids(monkeypatch):
    _, plan = _recipe_plan()
    app, dialog = _recipe_dialog(monkeypatch, plan)
    dialog.frames.setText("3, 5, 7")
    dialog.initialization.setCurrentIndex(1)
    assert dialog._options().frame_indices == (0, 3, 5, 7, 8)
    dialog.initialization.setCurrentIndex(0)
    assert dialog.frames.text() == "3, 5, 7"
    dialog.cleanup()


def test_refit_scheduled_recipe_summary_and_scalar_edit_preserve_phases(monkeypatch):
    from dataclasses import asdict, replace
    from src.shared.python.motion_matching.historical_fit import (
        ContactPinPhase,
        ContactPinSchedule,
        ScheduledConstraintOptions,
    )

    config, plan = _recipe_plan()
    schedule = ContactPinSchedule(
        "capture",
        "sha256:" + "a" * 64,
        (
            ContactPinPhase((20, 2), (22, 2), ("heel_r",), ("sha256:" + "b" * 64,)),
            ContactPinPhase((22, 2), (24, 2), (), ("sha256:" + "c" * 64,)),
        ),
    )
    config = replace(
        config,
        constraint_options=ScheduledConstraintOptions(
            config.constraint_options, schedule
        ),
    )
    plan["baseline_config"] = asdict(config)
    app, dialog = _recipe_dialog(monkeypatch, plan)
    summary = dialog.recipe_summary.text()
    assert "2 Authored Contact Phases" in summary
    assert "Review Source Interval: 10" in summary
    assert "12" in summary
    assert "Authored Contact Hypothesis" in summary
    assert "heel_r" in summary
    dialog.fields["prior_weight"].setText("0.003")
    assert dialog._options().config == replace(config, prior_weight=0.003)
    dialog.initialization.setCurrentIndex(1)
    assert dialog._options().config.constraint_options == config.constraint_options
    dialog.cleanup()


def test_editable_modes_and_resume_roundtrip_preserve_user_domain(monkeypatch):
    _, plan = _recipe_plan()
    app, dialog = _recipe_dialog(monkeypatch, plan)
    try:
        dialog.frames.setText("0, 2, 4, 6")
        dialog.fields["knot_count"].setText("4")
        dialog.initialization.setCurrentIndex(2)
        assert dialog.frames.text() == "0, 2, 4, 6"
        assert dialog.fields["knot_count"].text() == "4"
        dialog.initialization.setCurrentIndex(0)
        assert dialog.frames.text() == "0, 2, 4, 6"
        assert dialog.fields["knot_count"].text() == "4"
        dialog.initialization.setCurrentIndex(2)
        dialog.frames.setText("2, 4, 6")
        dialog.fields["knot_count"].setText("3")
        dialog.initialization.setCurrentIndex(1)
        assert dialog.frames.text() == "0, 2, 4, 6, 8"
        assert not dialog.frames.isEnabled()
        dialog.initialization.setCurrentIndex(2)
        assert dialog.frames.text() == "2, 4, 6"
        assert dialog.fields["knot_count"].text() == "3"
        assert dialog.frames.isEnabled()
        dialog.initialization.setCurrentIndex(1)
        dialog.initialization.setCurrentIndex(0)
        assert dialog.frames.text() == "2, 4, 6"
        assert dialog.fields["knot_count"].text() == "3"
    finally:
        dialog.cleanup()
