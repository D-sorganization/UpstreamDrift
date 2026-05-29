"""Headless tests for the Simulation Backends ``MainWidget``.

Runs offscreen with the Agg matplotlib backend (configured in
``conftest.py``). The widget actions are synchronous and dialog-free, so
the tests drive them directly. Each test builds at most one
``MainWidget`` and tears it down via the ``widget`` fixture to keep Qt's
C++ teardown deterministic.
"""

from __future__ import annotations

import pytest

from src.shared.python.simulation_backends import has_mujoco

pytestmark = [pytest.mark.unit]


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    """Yield a fresh ``MainWidget``, deleting it on teardown."""
    from src.tools.simulation_backends_launcher.gui import MainWidget

    main_widget = MainWidget()
    yield main_widget
    main_widget.deleteLater()


def test_key_widgets_exist(widget) -> None:  # noqa: ANN001
    """All advertised controls and output panes are present."""
    assert widget.backend_combo is not None
    assert widget.upper_mass_spin is not None
    assert widget.lower_clubhead_mass_spin is not None
    assert widget.damping_wrist_spin is not None
    assert widget.plane_incl_spin is not None
    assert widget.gravity_check is not None
    assert widget.horizon_spin is not None
    assert widget.dt_spin is not None
    assert widget.run_button is not None
    assert widget.sweep_button is not None
    assert widget.crossval_button is not None
    assert widget.export_button is not None
    assert widget.canvas is not None
    assert widget.report_text is not None
    assert widget.status_label is not None


def test_backend_combo_lists_all_backends(widget) -> None:  # noqa: ANN001
    """The combo carries the raw backend names as item data."""
    names = {
        widget.backend_combo.itemData(i) for i in range(widget.backend_combo.count())
    }
    assert {"ode", "mujoco", "mjwarp"} <= names


def test_default_backend_is_ode(widget) -> None:  # noqa: ANN001
    """ODE (always available) is the initial selection."""
    assert widget.current_backend_name() == "ode"


def test_editing_upper_mass_updates_params(widget) -> None:  # noqa: ANN001
    """Editing the upper-mass spin flows through to ``current_params``."""
    widget.upper_mass_spin.setValue(9.0)
    assert widget.current_params().upper.mass_kg == pytest.approx(9.0)


def test_gravity_checkbox_flows_to_params(widget) -> None:  # noqa: ANN001
    """The gravity checkbox toggles ``gravity_enabled`` in the model."""
    widget.gravity_check.setChecked(False)
    assert widget.current_params().gravity_enabled is False
    widget.gravity_check.setChecked(True)
    assert widget.current_params().gravity_enabled is True


def test_run_rollout_sets_last_trace(widget) -> None:  # noqa: ANN001
    """An ODE rollout stores a trace of ``horizon + 1`` samples and a status."""
    horizon = 120
    widget.horizon_spin.setValue(horizon)
    widget.run_rollout()
    assert widget._last_trace is not None  # noqa: SLF001
    assert widget._last_trace.num_steps == horizon + 1  # noqa: SLF001
    assert widget._last_trace.backend == "ode"  # noqa: SLF001
    assert widget.status_label.text()


def test_run_sweep_writes_report(widget) -> None:  # noqa: ANN001
    """The clubhead-mass sweep populates the report pane."""
    widget.horizon_spin.setValue(150)
    widget.run_sweep()
    report = widget.report_text.toPlainText()
    assert report
    assert "sweep" in report.lower()


def test_export_before_rollout_raises(widget, tmp_path) -> None:  # noqa: ANN001
    """Exporting with no rollout yet is a precondition violation."""
    with pytest.raises(ValueError, match="rollout"):
        widget.export_trace_to(str(tmp_path / "nothing.h5"))


def test_export_after_rollout_writes_file(widget, tmp_path) -> None:  # noqa: ANN001
    """Exporting after a rollout returns ``True`` and writes the HDF5 file."""
    widget.horizon_spin.setValue(60)
    widget.run_rollout()
    out = tmp_path / "trace.h5"
    assert widget.export_trace_to(str(out)) is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_run_cross_validation_does_not_raise(widget) -> None:  # noqa: ANN001
    """Cross-validation always populates the report without raising."""
    widget.run_cross_validation()
    assert widget.report_text.toPlainText()


@pytest.mark.skipif(not has_mujoco(), reason="MuJoCo not installed")
def test_cross_validation_reports_pass(widget) -> None:  # noqa: ANN001
    """With MuJoCo present, the ODE/MuJoCo cross-check passes."""
    widget.run_cross_validation()
    report = widget.report_text.toPlainText()
    assert "mass_matrix" in report
    assert "trajectory" in report
    assert "PASS" in report
    assert "FAIL" not in report
