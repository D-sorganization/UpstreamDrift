"""Headless pytest-qt tests for the calibration dialog (issue #4820)."""

from __future__ import annotations

import os

# Headless Qt platform must be set before any PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import sys
from pathlib import Path

import pytest

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication, QDialogButtonBox  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox  # noqa: E402

from src.shared.python.anthropometrics._subject_anthropometrics import (  # noqa: E402
    SubjectAnthropometrics,
)
from src.tools.starting_pose_matcher.widgets.calibration_dialog import (  # noqa: E402
    ESTIMATORS,
    SEX_VALUES,
    CalibrationDialog,
    build_subject_record,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    return app  # type: ignore[return-value]


@pytest.fixture
def dialog(qapp: QApplication, tmp_path: Path) -> CalibrationDialog:
    dlg = CalibrationDialog(subjects_dir=tmp_path, default_subject_id="test_subj")
    yield dlg
    dlg.deleteLater()


# --------------------------------------------------------------------- #
# Construction & form state                                             #
# --------------------------------------------------------------------- #
def test_dialog_constructs_with_expected_widgets(dialog: CalibrationDialog) -> None:
    assert dialog.windowTitle() == "Subject Anthropometrics Calibration"
    assert dialog.isModal()
    values = dialog.current_form_values()
    assert set(values) == {
        "subject_id",
        "height_m",
        "mass_kg",
        "age_years",
        "sex",
        "estimator",
    }


def test_default_form_values_are_in_range(dialog: CalibrationDialog) -> None:
    values = dialog.current_form_values()
    assert 0.5 <= values["height_m"] <= 2.5  # type: ignore[operator]
    assert 10.0 <= values["mass_kg"] <= 250.0  # type: ignore[operator]
    assert 1 <= values["age_years"] <= 120  # type: ignore[operator]
    assert values["sex"] in SEX_VALUES
    assert values["estimator"] in ESTIMATORS


def test_estimator_combo_lists_all_three(dialog: CalibrationDialog) -> None:
    items = [
        dialog.estimator_combo.itemText(i)
        for i in range(dialog.estimator_combo.count())
    ]
    assert set(items) == {"de_leva", "dempster", "zatsiorsky"}


def test_sex_combo_lists_all_options(dialog: CalibrationDialog) -> None:
    data = [dialog.sex_combo.itemData(i) for i in range(dialog.sex_combo.count())]
    assert set(data) == set(SEX_VALUES)


# --------------------------------------------------------------------- #
# Validation (boundary)                                                 #
# --------------------------------------------------------------------- #
def test_empty_subject_id_disables_ok(dialog: CalibrationDialog) -> None:
    dialog.id_edit.setText("   ")
    ok_btn = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert not ok_btn.isEnabled()
    assert "required" in dialog.status_label.text().lower()


def test_non_empty_subject_id_enables_ok(dialog: CalibrationDialog) -> None:
    dialog.id_edit.setText("subject_42")
    ok_btn = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn is not None
    assert ok_btn.isEnabled()


def test_build_subject_record_rejects_bad_height() -> None:
    with pytest.raises(ValueError, match="height_m"):
        build_subject_record(
            subject_id="x",
            height_m=10.0,  # absurd
            mass_kg=70.0,
            age_years=30,
            sex="M",
            estimator="de_leva",
        )


def test_build_subject_record_rejects_bad_mass() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        build_subject_record(
            subject_id="x",
            height_m=1.7,
            mass_kg=5.0,
            age_years=30,
            sex="M",
            estimator="de_leva",
        )


def test_build_subject_record_rejects_bad_age() -> None:
    with pytest.raises(ValueError, match="age_years"):
        build_subject_record(
            subject_id="x",
            height_m=1.7,
            mass_kg=70.0,
            age_years=0,
            sex="M",
            estimator="de_leva",
        )


def test_build_subject_record_rejects_bad_sex() -> None:
    with pytest.raises(ValueError, match="sex"):
        build_subject_record(
            subject_id="x",
            height_m=1.7,
            mass_kg=70.0,
            age_years=30,
            sex="other",
            estimator="de_leva",
        )


def test_build_subject_record_rejects_bad_estimator() -> None:
    with pytest.raises(ValueError, match="estimator"):
        build_subject_record(
            subject_id="x",
            height_m=1.7,
            mass_kg=70.0,
            age_years=30,
            sex="M",
            estimator="bogus",
        )


def test_build_subject_record_rejects_empty_subject_id() -> None:
    with pytest.raises(ValueError, match="subject_id"):
        build_subject_record(
            subject_id="   ",
            height_m=1.7,
            mass_kg=70.0,
            age_years=30,
            sex="M",
            estimator="de_leva",
        )


# --------------------------------------------------------------------- #
# Pure helper                                                           #
# --------------------------------------------------------------------- #
def test_build_subject_record_returns_subject_anthropometrics() -> None:
    record = build_subject_record(
        subject_id="alice",
        height_m=1.70,
        mass_kg=65.0,
        age_years=28,
        sex="F",
        estimator="de_leva",
    )
    assert isinstance(record, SubjectAnthropometrics)
    assert record.subject_id == "alice"
    assert record.height_m == pytest.approx(1.70)
    assert record.mass_kg == pytest.approx(65.0)


@pytest.mark.parametrize("estimator", list(ESTIMATORS))
def test_each_estimator_produces_valid_record(estimator: str) -> None:
    record = build_subject_record(
        subject_id=f"s_{estimator}",
        height_m=1.78,
        mass_kg=75.0,
        age_years=30,
        sex="M",
        estimator=estimator,
    )
    assert isinstance(record, SubjectAnthropometrics)
    assert record.subject_id == f"s_{estimator}"
    assert len(record.segments) > 0


# --------------------------------------------------------------------- #
# Accept flow / persistence                                             #
# --------------------------------------------------------------------- #
def test_accept_persists_subject_json(
    dialog: CalibrationDialog, tmp_path: Path
) -> None:
    captured: list[object] = []
    dialog.calibrated.connect(captured.append)

    dialog.id_edit.setText("alpha")
    dialog.height_spin.setValue(1.80)
    dialog.mass_spin.setValue(80.0)
    dialog.age_spin.setValue(35)
    dialog.sex_combo.setCurrentIndex(dialog.sex_combo.findData("M"))
    dialog.estimator_combo.setCurrentText("dempster")

    dialog._on_accept()

    assert dialog.result() == int(QDialog.DialogCode.Accepted)
    res = dialog.result_record
    assert res is not None
    assert res.saved_path == tmp_path / "alpha.json"
    assert res.saved_path.exists()
    payload = json.loads(res.saved_path.read_text())
    assert payload["subject_id"] == "alpha"
    assert len(captured) == 1


def test_accept_with_invalid_subject_id_does_not_emit(
    dialog: CalibrationDialog,
) -> None:
    captured: list[object] = []
    dialog.calibrated.connect(captured.append)
    dialog.id_edit.setText("")
    dialog._on_accept()
    # Should not have closed/accepted; status reports the error.
    assert dialog.result_record is None
    assert captured == []
    assert dialog.status_label.text() != ""


def test_persist_creates_subjects_dir_if_missing(
    qapp: QApplication, tmp_path: Path
) -> None:
    nested = tmp_path / "nested" / "deeper"
    assert not nested.exists()
    dlg = CalibrationDialog(subjects_dir=nested, default_subject_id="bob")
    try:
        dlg._on_accept()
        assert nested.exists()
        assert (nested / "bob.json").exists()
    finally:
        dlg.deleteLater()
