"""Focused tests for workspace project/session metadata."""

from __future__ import annotations

import pytest

from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace import SessionProjectStore

pytestmark = pytest.mark.unit


def test_create_list_and_load_session_round_trips(tmp_path) -> None:
    store = SessionProjectStore(tmp_path / "study")

    project = store.create_project("project-001", "Swing Study")
    subject = store.add_subject(
        "subject-001",
        "Subject One",
        metadata={"handedness": "right"},
    )
    session = store.create_session(
        "session-001",
        subject.subject_id,
        "Baseline capture",
        metadata={"operator": "lab"},
    )
    dataset = store.register_dataset(
        "dataset-001",
        session.session_id,
        "inputs/baseline.c3d",
        "c3d",
        metadata={"frames": 120},
    )

    fresh = SessionProjectStore(tmp_path / "study")
    loaded_project = fresh.load_project()
    sessions = fresh.list_sessions(subject.subject_id)

    assert project.project_id == "project-001"
    assert loaded_project.subjects["subject-001"].metadata["handedness"] == "right"
    assert sessions == [session]
    assert fresh.load_session("session-001").metadata["operator"] == "lab"
    assert fresh.list_datasets("session-001") == [dataset]


def test_session_requires_existing_subject(tmp_path) -> None:
    store = SessionProjectStore(tmp_path / "study")
    store.create_project("project-001", "Swing Study")

    with pytest.raises(KeyError, match="unknown subject_id"):
        store.create_session("session-001", "subject-missing", "Baseline")


def test_duplicate_session_is_rejected(tmp_path) -> None:
    store = SessionProjectStore(tmp_path / "study")
    store.create_project("project-001", "Swing Study")
    store.add_subject("subject-001", "Subject One")
    store.create_session("session-001", "subject-001", "Baseline")

    with pytest.raises(StateError, match="session already exists"):
        store.create_session("session-001", "subject-001", "Duplicate")
