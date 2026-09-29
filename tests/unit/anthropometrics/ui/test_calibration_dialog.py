"""Tests for :class:`SubjectCalibrationDialog` (issue #4820)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Headless Qt platform must be set before any PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")

if "PySide6" in sys.modules:  # pragma: no cover - environment-dependent
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable",
        allow_module_level=True,
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


from PyQt6.QtWidgets import QApplication  # noqa: E402

from anthropometrics import SubjectAnthropometrics  # noqa: E402
from anthropometrics.readers.c3d_subject_info import (  # noqa: E402
    C3DSubjectMetadata,
)
from anthropometrics.ui.calibration_dialog import (  # noqa: E402
    SubjectCalibrationDialog,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def fake_meta() -> C3DSubjectMetadata:
    """A representative populated metadata record."""
    from anthropometrics import Sex

    return C3DSubjectMetadata(
        subject_id="JOHN_DOE",
        height_m=1.83,
        mass_kg=82.0,
        age_years=34.0,
        sex=Sex.MALE,
        leg_length_m=0.92,
        arm_length_m=0.78,
    )


@pytest.fixture
def patched_reader(
    monkeypatch: pytest.MonkeyPatch, fake_meta: C3DSubjectMetadata
) -> C3DSubjectMetadata:
    """Patch the dialog's C3D reader to return *fake_meta* without I/O."""

    def _fake_read(_path: Any) -> C3DSubjectMetadata:
        return fake_meta

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.read_c3d_subject_metadata",
        _fake_read,
    )
    return fake_meta


# --------------------------------------------------------------------------- #
# Construction + DbC                                                          #
# --------------------------------------------------------------------------- #
def test_dialog_constructs(qapp: QApplication) -> None:
    dialog = SubjectCalibrationDialog()
    try:
        # Both compute-dependent buttons start disabled.
        assert not dialog._save_button.isEnabled()
        assert not dialog._export_button.isEnabled()
        # Compute is enabled because default spinbox values are valid.
        assert dialog._compute_button.isEnabled()
        # Estimator combo carries the three published estimators.
        assert dialog._estimator_combo.count() == 3
        items = [
            dialog._estimator_combo.itemText(i)
            for i in range(dialog._estimator_combo.count())
        ]
        assert items == ["de_leva", "dempster", "zatsiorsky"]
        # Engine combo uses ADAPTER_REGISTRY.keys().
        from anthropometrics import ADAPTER_REGISTRY

        engines = [
            dialog._engine_combo.itemText(i)
            for i in range(dialog._engine_combo.count())
        ]
        assert sorted(engines) == sorted(ADAPTER_REGISTRY.keys())
    finally:
        dialog.deleteLater()


def test_open_for_mocap_rejects_invalid_type(qapp: QApplication) -> None:
    dialog = SubjectCalibrationDialog()
    try:
        with pytest.raises(TypeError, match="path must be"):
            dialog.open_for_mocap(42)  # type: ignore[arg-type]
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Auto-fill                                                                   #
# --------------------------------------------------------------------------- #
def test_set_mocap_path_autofills_height_mass(
    qapp: QApplication,
    tmp_path: Path,
    patched_reader: C3DSubjectMetadata,
) -> None:
    dialog = SubjectCalibrationDialog()
    try:
        mocap = tmp_path / "shot.c3d"
        mocap.write_bytes(b"\x00")  # placeholder — reader is patched
        dialog.set_mocap_path(mocap)
        assert dialog._height_spin.value() == pytest.approx(1.83, abs=1e-6)
        assert dialog._mass_spin.value() == pytest.approx(82.0, abs=1e-6)
        assert dialog._subject_id_edit.text() == "JOHN_DOE"
        assert dialog._mocap_edit.text() == str(mocap)
    finally:
        dialog.deleteLater()


def test_auto_fill_silently_ignores_reader_errors(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _broken_read(_path: Any) -> C3DSubjectMetadata:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.read_c3d_subject_metadata",
        _broken_read,
    )
    dialog = SubjectCalibrationDialog()
    try:
        dialog.set_mocap_path(tmp_path / "missing.c3d")
        # Spinbox values are unchanged (defaults).
        assert dialog._height_spin.value() == pytest.approx(1.75, abs=1e-6)
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Compute → SegmentPropertiesPanel update                                     #
# --------------------------------------------------------------------------- #
def test_compute_updates_segment_panel(
    qapp: QApplication,
    patched_reader: C3DSubjectMetadata,
    tmp_path: Path,
) -> None:
    dialog = SubjectCalibrationDialog()
    try:
        dialog.set_mocap_path(tmp_path / "shot.c3d")
        # Spinbox change exercises the compute-enabled refresh.
        dialog._mass_spin.setValue(80.0)
        assert dialog._compute_button.isEnabled()

        dialog._compute_button.click()

        record = dialog.result_record()
        assert isinstance(record, SubjectAnthropometrics)
        # The segment list was populated from the produced record.
        assert dialog._segment_list.count() == len(record.segments)
        # The first segment is selected and rendered in the panel.
        assert dialog._panel.current_segment is not None
        first_name, first_props = record.segments[0]
        assert dialog._panel.current_segment is first_props
        assert first_name in dialog._panel.title()
        # Save / Export are now enabled.
        assert dialog._save_button.isEnabled()
        assert dialog._export_button.isEnabled()
    finally:
        dialog.deleteLater()


def test_segment_selection_changes_panel(
    qapp: QApplication,
    patched_reader: C3DSubjectMetadata,
    tmp_path: Path,
) -> None:
    dialog = SubjectCalibrationDialog()
    try:
        dialog.set_mocap_path(tmp_path / "shot.c3d")
        dialog._compute_button.click()
        record = dialog.result_record()
        assert record is not None
        assert dialog._segment_list.count() >= 2

        dialog._segment_list.setCurrentRow(1)
        _name, second_props = record.segments[1]
        assert dialog._panel.current_segment is second_props
    finally:
        dialog.deleteLater()


def test_compute_disabled_when_height_invalid(qapp: QApplication) -> None:
    """The Compute button reflects spinbox validity (DbC)."""
    dialog = SubjectCalibrationDialog()
    try:
        # The QDoubleSpinBox enforces its own range — pushing to the
        # absolute lower bound is still valid by construction. Force
        # the underlying check to fail by monkeying the spinbox value.
        dialog._height_spin.setMinimum(0.0)
        dialog._height_spin.setValue(0.0)
        dialog._refresh_compute_enabled()
        assert not dialog._compute_button.isEnabled()
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Save                                                                        #
# --------------------------------------------------------------------------- #
def test_save_writes_file_to_default_subjects_dir(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    patched_reader: C3DSubjectMetadata,
    tmp_path: Path,
) -> None:
    redirect = tmp_path / "subjects"

    def _redirected_dir() -> Path:
        return redirect

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.default_subjects_dir",
        _redirected_dir,
    )
    # Suppress message box.
    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.QMessageBox.information",
        lambda *_a, **_k: None,
    )

    dialog = SubjectCalibrationDialog()
    try:
        dialog.set_mocap_path(tmp_path / "shot.c3d")
        dialog._subject_id_edit.setText("subject_xyz")
        dialog._compute_button.click()
        dialog._save_button.click()
        target = redirect / "subject_xyz.json"
        assert target.exists()
        # File must be valid JSON parseable back into the canonical type.
        from anthropometrics import load_subject

        roundtrip = load_subject(target)
        assert roundtrip.subject_id == "subject_xyz"
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Export                                                                      #
# --------------------------------------------------------------------------- #
def test_export_invokes_run_pipeline(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    patched_reader: C3DSubjectMetadata,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_pipeline(
        mocap_file: Any,
        *,
        subject_height_m: float,
        subject_mass_kg: float,
        estimator: str,
        target_engines: tuple[str, ...],
        output_dir: Any,
    ) -> SubjectAnthropometrics:
        captured["mocap_file"] = mocap_file
        captured["subject_height_m"] = subject_height_m
        captured["subject_mass_kg"] = subject_mass_kg
        captured["estimator"] = estimator
        captured["target_engines"] = tuple(target_engines)
        captured["output_dir"] = output_dir
        return _PLACEHOLDER_RECORD

    # Stub QFileDialog.getExistingDirectory to skip user interaction.
    out_dir = tmp_path / "export"
    out_dir.mkdir()

    def _fake_get_existing(*_a: Any, **_k: Any) -> str:
        return str(out_dir)

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog._pipeline.run_pipeline",
        _fake_run_pipeline,
    )
    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.QFileDialog.getExistingDirectory",
        _fake_get_existing,
    )
    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.QMessageBox.information",
        lambda *_a, **_k: None,
    )

    dialog = SubjectCalibrationDialog()
    try:
        mocap = tmp_path / "shot.c3d"
        dialog.set_mocap_path(mocap)
        dialog._estimator_combo.setCurrentText("dempster")
        dialog._engine_combo.setCurrentText("drake")
        dialog._compute_button.click()
        dialog._export_button.click()

        assert captured["mocap_file"] == mocap
        assert captured["subject_height_m"] == pytest.approx(1.83, abs=1e-6)
        assert captured["subject_mass_kg"] == pytest.approx(82.0, abs=1e-6)
        assert captured["estimator"] == "dempster"
        assert captured["target_engines"] == ("drake",)
        assert captured["output_dir"] == out_dir
    finally:
        dialog.deleteLater()


def test_export_aborts_when_user_cancels_directory_dialog(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    patched_reader: C3DSubjectMetadata,
    tmp_path: Path,
) -> None:
    """Cancelled QFileDialog returns ''. Pipeline must not run."""
    called = {"n": 0}

    def _never(*_a: Any, **_k: Any) -> SubjectAnthropometrics:
        called["n"] += 1
        return _PLACEHOLDER_RECORD

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog._pipeline.run_pipeline",
        _never,
    )
    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.QFileDialog.getExistingDirectory",
        lambda *_a, **_k: "",
    )

    dialog = SubjectCalibrationDialog()
    try:
        dialog.set_mocap_path(tmp_path / "shot.c3d")
        dialog._compute_button.click()
        dialog._export_button.click()
        assert called["n"] == 0
    finally:
        dialog.deleteLater()


def test_export_warns_when_mocap_path_missing(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    patched_reader: C3DSubjectMetadata,
) -> None:
    captured = {"warned": False}

    def _record_warn(*_a: Any, **_k: Any) -> None:
        captured["warned"] = True

    monkeypatch.setattr(
        "anthropometrics.ui.calibration_dialog.QMessageBox.warning",
        _record_warn,
    )

    dialog = SubjectCalibrationDialog()
    try:
        # Run compute with default scalars (mocap path empty).
        dialog._compute_button.click()
        dialog._export_button.click()
        assert captured["warned"] is True
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Module-level placeholder record builder                                     #
# --------------------------------------------------------------------------- #
def _placeholder_record() -> SubjectAnthropometrics:
    """Cheap stand-in :class:`SubjectAnthropometrics` for export stub."""
    from anthropometrics.estimators import DeLevaEstimator

    return DeLevaEstimator().estimate(
        subject_id="placeholder",
        height_m=1.75,
        mass_kg=70.0,
    )


_PLACEHOLDER_RECORD = _placeholder_record()
