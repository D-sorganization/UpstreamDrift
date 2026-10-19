"""Offscreen GUI tests for the Spine Viewer (SPN-15b, #12772)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit


def _load_widget_class():
    try:
        import matplotlib  # noqa: F401
        import PyQt6.QtWidgets  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    from src.tools.spine_viewer.gui import SpineViewerWidget

    return SpineViewerWidget


@pytest.fixture
def qapp():
    _load_widget_class()
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_empty_widget_shows_no_data_not_qualified(qapp) -> None:
    widget = _load_widget_class()()
    assert "no study" in widget.status_text().lower()
    assert "QUALIFIED" not in widget.status_text().replace("UNQUALIFIED", "")
    widget.cleanup()


def test_loaded_unqualified_study_is_prominent(qapp, study_factory) -> None:
    from .conftest import SOURCE

    widget = _load_widget_class()()
    widget.load_study(study_factory())
    assert "UNQUALIFIED" in widget.status_text()
    assert SOURCE in widget.distribution_text()
    assert widget.current_quantity() == "compression"
    widget.set_quantity("ap_shear")
    assert widget.current_quantity() == "ap_shear"
    assert widget.current_figure() is not None
    widget.cleanup()


def test_bad_receipt_shows_error_and_clears_data(qapp, study_factory) -> None:
    widget = _load_widget_class()()
    widget.load_study(study_factory())
    widget.load_study(widget_bad := study_factory().parent / "missing.json")
    assert widget_bad
    assert "ERROR" in widget.status_text()
    assert widget.current_figure() is None
    widget.cleanup()


def test_cleanup_is_idempotent(qapp, study_factory) -> None:
    widget = _load_widget_class()()
    widget.load_study(study_factory())
    widget.cleanup()
    widget.cleanup()


def test_run_study_panel_controls_initialized(qapp) -> None:
    widget = _load_widget_class()()
    model_combo = widget.model_combo()
    assert model_combo.count() > 0
    # Must list fixture
    items = [model_combo.itemText(i) for i in range(model_combo.count())]
    assert any("fixture" in item for item in items)

    # Fixture status shown honestly
    assert "fixture" in widget.model_status_text().lower()
    assert "qualified" not in widget.model_status_text().lower().replace(
        "unqualified", ""
    )

    # Coupling laws
    law_combo = widget.coupling_law_combo()
    law_items = [law_combo.itemText(i) for i in range(law_combo.count())]
    assert any("bruno2015_regional" in item for item in law_items)

    assert widget.run_study_button() is not None
    assert widget.motion_path_edit() is not None
    widget.cleanup()


def test_model_selection_updates_status_honestly(qapp) -> None:
    widget = _load_widget_class()()
    model_combo = widget.model_combo()
    # Find an asset item
    idx = -1
    for i in range(model_combo.count()):
        data = model_combo.itemData(i)
        if data and data != "fixture":
            idx = i
            break
    assert idx >= 0
    model_combo.setCurrentIndex(idx)
    status_text = widget.model_status_text().lower()
    assert (
        "missing" in status_text
        or "unpinned" in status_text
        or "qualified" in status_text
    )

    # Switch back to fixture
    fixture_idx = -1
    for i in range(model_combo.count()):
        if model_combo.itemData(i) == "fixture":
            fixture_idx = i
            break
    assert fixture_idx >= 0
    model_combo.setCurrentIndex(fixture_idx)
    assert "fixture" in widget.model_status_text().lower()
    assert "qualified" not in widget.model_status_text().lower().replace(
        "unqualified", ""
    )
    widget.cleanup()


def test_start_study_with_missing_inputs_shows_error(qapp) -> None:
    widget = _load_widget_class()()
    widget.motion_path_edit().setText("non_existent_file.sto")
    widget.start_study()
    assert "error" in widget.run_study_status_text().lower()
    widget.cleanup()


def test_study_worker_success_loads_receipt(qapp, tmp_path, monkeypatch) -> None:
    from .conftest import write_study

    widget = _load_widget_class()()
    out_dir = tmp_path / "worker_out"

    motion = tmp_path / "swing.sto"
    motion.write_text("motion")
    golf_model = tmp_path / "golf.osim"
    golf_model.write_text("golf")
    model = tmp_path / "mod.osim"
    model.write_text("model")

    from src.engines.physics_engines.opensim.python.spine import study as study_mod

    def fake_run(req):
        write_study(req.output_dir)

    monkeypatch.setattr(study_mod, "run_spine_study", fake_run)

    widget.study_id_edit().setText("worker_test")
    widget.motion_path_edit().setText(str(motion))

    # Trigger run with explicit overrides to use tmp_path models
    payload = {
        "study_id": "worker_test",
        "model_id": "fixture",
        "motion_path": str(motion),
        "golf_model_path": str(golf_model),
        "model_path": str(model),
        "output_dir": str(out_dir),
    }
    widget.start_study(payload=payload, root=tmp_path)
    # Wait for completion
    finished = widget.wait_for_study(timeout_ms=5000)
    assert finished
    assert "UNQUALIFIED" in widget.status_text()
    assert widget.current_figure() is not None
    widget.cleanup()


def test_study_worker_failure_shows_error(qapp, tmp_path, monkeypatch) -> None:
    widget = _load_widget_class()()
    out_dir = tmp_path / "worker_fail"

    motion = tmp_path / "swing.sto"
    motion.write_text("motion")
    golf_model = tmp_path / "golf.osim"
    golf_model.write_text("golf")
    model = tmp_path / "mod.osim"
    model.write_text("model")

    from src.engines.physics_engines.opensim.python.spine import study as study_mod

    def failing_run(req):
        raise ValueError("simulated solver crash")

    monkeypatch.setattr(study_mod, "run_spine_study", failing_run)

    payload = {
        "study_id": "worker_fail_test",
        "model_id": "fixture",
        "motion_path": str(motion),
        "golf_model_path": str(golf_model),
        "model_path": str(model),
        "output_dir": str(out_dir),
    }
    widget.start_study(payload=payload, root=tmp_path)
    finished = widget.wait_for_study(timeout_ms=5000)
    assert finished
    assert "simulated solver crash" in widget.run_study_status_text()
    widget.cleanup()
