"""Unit and behavioral UI tests for OpenCap Session Import Action (#11409).

Tests the PyQt6 OpenCap session import action:
1. Lists the session's trials from session directory.
2. Allows user selection of a trial.
3. Loads chosen trial through load_opencap_session.
4. Passes scaled model and kinematics to the OpenSim engine target.
5. Handles invalid / missing session directories gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PyQt6 = pytest.importorskip("PyQt6")

from src.engines.physics_engines.opensim.python.opencap_import_action import (
    OpenCapImportAction,
    OpenCapImportDialog,
)
from src.engines.physics_engines.opensim.python.opensim_gui import MainWidget
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    write_kinematics,
    write_opencap_session,
    write_scaled_model,
)

pytestmark = pytest.mark.unit


def _create_test_session(
    tmp_path: Path, trials: tuple[str, ...] = ("neutral", "swing1", "swing2")
) -> Path:
    session = write_opencap_session(tmp_path, trials)
    write_scaled_model(session)
    for trial in trials:
        write_kinematics(session, trial)
    return session


class MockOpenSimEngine:
    """Mock OpenSim engine target conforming to the engine loading protocol."""

    def __init__(self) -> None:
        self.model_path: str | None = None
        self.kinematics: Any = None
        self.opencap_session: Any = None

    def load_from_path(self, path: str) -> None:
        self.model_path = path

    def set_kinematics(self, kinematics: Any) -> None:
        self.kinematics = kinematics


def test_action_inspects_and_lists_trials(tmp_path: Path) -> None:
    session_dir = _create_test_session(tmp_path, ("neutral", "swing1", "swing2"))
    action = OpenCapImportAction()

    metadata = action.inspect_session(session_dir)

    assert metadata.session_dir == session_dir
    assert metadata.trials == ["neutral", "swing1", "swing2"]
    assert metadata.model_file is not None
    assert metadata.model_file.name == "LaiUhlrich2022_scaled.osim"
    assert metadata.subject.mass_kg == pytest.approx(79.5)
    assert "swing1" in metadata.kinematics_trials
    assert "swing2" in metadata.kinematics_trials


def test_action_imports_trial_and_hands_off_to_mock_engine(tmp_path: Path) -> None:
    session_dir = _create_test_session(tmp_path, ("neutral", "swing1", "swing2"))
    mock_engine = MockOpenSimEngine()
    action = OpenCapImportAction(target_engine=mock_engine)

    session = action.import_session(session_dir, trial="swing2")
    action.hand_off_to_engine(session)

    assert session.trial == "swing2"
    assert mock_engine.model_path is not None
    assert Path(mock_engine.model_path).name == "LaiUhlrich2022_scaled.osim"
    assert mock_engine.kinematics is not None
    assert mock_engine.kinematics == session.kinematics
    assert mock_engine.opencap_session == session


def test_dialog_ui_lists_trials_and_user_selection(qapp: Any, tmp_path: Path) -> None:
    session_dir = _create_test_session(tmp_path, ("neutral", "swing1", "swing2"))
    mock_engine = MockOpenSimEngine()
    action = OpenCapImportAction(target_engine=mock_engine)

    dialog = OpenCapImportDialog(action=action)
    dialog.set_session_dir(session_dir)

    # 1. Dialog populates trials in list widget
    assert dialog.trial_list.count() == 3
    trial_names = [
        item.text()
        for i in range(dialog.trial_list.count())
        if (item := dialog.trial_list.item(i)) is not None
    ]
    assert trial_names == ["neutral", "swing1", "swing2"]

    # 2. Select specific trial
    dialog.trial_list.setCurrentRow(2)
    assert dialog.selected_trial() == "swing2"

    # 3. Accept / Import
    dialog.accept_import()

    assert dialog.imported_session is not None
    assert dialog.imported_session.trial == "swing2"
    assert mock_engine.model_path == str(dialog.imported_session.model_file)
    assert mock_engine.kinematics is not None


def test_action_hands_off_to_opensim_main_widget(
    qapp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    session_dir = _create_test_session(tmp_path, ("neutral", "swing1"))
    widget = MainWidget(model_path=None)
    action = OpenCapImportAction(target_engine=widget)

    session = action.import_session(session_dir, trial="swing1")
    action.hand_off_to_engine(session)

    assert widget.model_path == str(session.model_file)
    assert widget.kinematics is not None
    assert widget.opencap_session == session
    assert "OpenCap Session Loaded" in widget.lbl_status.text()
    assert "swing1" in widget.lbl_details.text()


def test_action_handles_missing_session_directory() -> None:
    action = OpenCapImportAction()
    with pytest.raises(NotADirectoryError):
        action.inspect_session(Path("/nonexistent/directory/path/for/opencap"))
