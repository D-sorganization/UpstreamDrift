"""Durable project/session metadata for the unified workspace."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts.exceptions import StateError

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROJECT_FILE = "project.json"


@dataclass(frozen=True)
class SubjectMetadata:
    """A study participant or model subject in a workspace project."""

    subject_id: str
    display_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMetadata:
    """Input dataset metadata attached to a project session."""

    dataset_id: str
    session_id: str
    path: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionMetadata:
    """A capture, analysis, or simulation session under one subject."""

    session_id: str
    subject_id: str
    name: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectMetadata:
    """The project spine persisted to ``project.json``."""

    project_id: str
    name: str
    root: str
    created_at: str
    updated_at: str
    subjects: dict[str, SubjectMetadata] = field(default_factory=dict)
    sessions: dict[str, SessionMetadata] = field(default_factory=dict)
    datasets: dict[str, DatasetMetadata] = field(default_factory=dict)


class SessionProjectStore:
    """JSON-backed project/session/dataset metadata store.

    Postcondition: successful mutations are durable in ``project.json`` and a
    fresh store pointed at the same root can load them.
    """

    def __init__(self, root: Path | str) -> None:
        if not isinstance(root, (Path, str)):
            raise TypeError("root must be a pathlib.Path or str")
        self._root = Path(root).expanduser().resolve()
        self._path = self._root / _PROJECT_FILE

    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._root

    def create_project(self, project_id: str, name: str) -> ProjectMetadata:
        """Create and persist a new project metadata file."""
        _validate_id(project_id, "project_id")
        _validate_non_empty(name, "name")
        if self._path.exists():
            raise StateError(f"project already exists at {self._path}")
        now = _utc_now()
        project = ProjectMetadata(
            project_id=project_id,
            name=name,
            root=str(self._root),
            created_at=now,
            updated_at=now,
        )
        self._save(project)
        return project

    def load_project(self) -> ProjectMetadata:
        """Load the project metadata from disk."""
        if not self._path.exists():
            raise KeyError(f"project metadata not found: {self._path}")
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"malformed project metadata: {self._path}") from exc
        except OSError as exc:
            raise StateError(f"could not read project metadata: {self._path}") from exc
        if not isinstance(raw, dict):
            raise StateError("project metadata must be a JSON object")
        return _project_from_dict(raw)

    def add_subject(
        self,
        subject_id: str,
        display_name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SubjectMetadata:
        """Add or replace a subject record."""
        _validate_id(subject_id, "subject_id")
        _validate_non_empty(display_name, "display_name")
        subject = SubjectMetadata(
            subject_id=subject_id,
            display_name=display_name,
            metadata=dict(metadata or {}),
        )
        project = self.load_project()
        subjects = dict(project.subjects)
        subjects[subject_id] = subject
        self._save(_replace_project(project, subjects=subjects))
        return subject

    def create_session(
        self,
        session_id: str,
        subject_id: str,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMetadata:
        """Create a session for an existing subject."""
        _validate_id(session_id, "session_id")
        _validate_id(subject_id, "subject_id")
        _validate_non_empty(name, "name")
        project = self.load_project()
        if subject_id not in project.subjects:
            raise KeyError(f"unknown subject_id: {subject_id}")
        if session_id in project.sessions:
            raise StateError(f"session already exists: {session_id}")
        session = SessionMetadata(
            session_id=session_id,
            subject_id=subject_id,
            name=name,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        sessions = dict(project.sessions)
        sessions[session_id] = session
        self._save(_replace_project(project, sessions=sessions))
        return session

    def list_sessions(self, subject_id: str | None = None) -> list[SessionMetadata]:
        """Return sessions ordered by creation time, optionally by subject."""
        if subject_id is not None:
            _validate_id(subject_id, "subject_id")
        sessions = list(self.load_project().sessions.values())
        if subject_id is not None:
            sessions = [
                session for session in sessions if session.subject_id == subject_id
            ]
        return sorted(
            sessions, key=lambda session: (session.created_at, session.session_id)
        )

    def load_session(self, session_id: str) -> SessionMetadata:
        """Load one session by id."""
        _validate_id(session_id, "session_id")
        try:
            return self.load_project().sessions[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def register_dataset(
        self,
        dataset_id: str,
        session_id: str,
        path: Path | str,
        kind: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetMetadata:
        """Attach an input dataset or result directory to a session."""
        _validate_id(dataset_id, "dataset_id")
        _validate_id(session_id, "session_id")
        _validate_non_empty(kind, "kind")
        if not isinstance(path, (Path, str)):
            raise TypeError("path must be a pathlib.Path or str")
        project = self.load_project()
        if session_id not in project.sessions:
            raise KeyError(f"unknown session_id: {session_id}")
        dataset = DatasetMetadata(
            dataset_id=dataset_id,
            session_id=session_id,
            path=str(Path(path)),
            kind=kind,
            metadata=dict(metadata or {}),
        )
        datasets = dict(project.datasets)
        datasets[dataset_id] = dataset
        self._save(_replace_project(project, datasets=datasets))
        return dataset

    def list_datasets(self, session_id: str | None = None) -> list[DatasetMetadata]:
        """Return datasets ordered by id, optionally scoped to one session."""
        if session_id is not None:
            _validate_id(session_id, "session_id")
        datasets = list(self.load_project().datasets.values())
        if session_id is not None:
            datasets = [
                dataset for dataset in datasets if dataset.session_id == session_id
            ]
        return sorted(datasets, key=lambda dataset: dataset.dataset_id)

    def _save(self, project: ProjectMetadata) -> None:
        now = _utc_now()
        payload = asdict(_replace_project(project, updated_at=now))
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self._path, payload)
        except OSError as exc:
            raise StateError(f"could not write project metadata: {self._path}") from exc


def _replace_project(
    project: ProjectMetadata,
    *,
    root: str | None = None,
    updated_at: str | None = None,
    subjects: dict[str, SubjectMetadata] | None = None,
    sessions: dict[str, SessionMetadata] | None = None,
    datasets: dict[str, DatasetMetadata] | None = None,
) -> ProjectMetadata:
    return replace(
        project,
        root=project.root if root is None else root,
        updated_at=project.updated_at if updated_at is None else updated_at,
        subjects=project.subjects if subjects is None else subjects,
        sessions=project.sessions if sessions is None else sessions,
        datasets=project.datasets if datasets is None else datasets,
    )


def _project_from_dict(raw: dict[str, Any]) -> ProjectMetadata:
    subjects = {
        key: SubjectMetadata(**value)
        for key, value in dict(raw.get("subjects", {})).items()
    }
    sessions = {
        key: SessionMetadata(**value)
        for key, value in dict(raw.get("sessions", {})).items()
    }
    datasets = {
        key: DatasetMetadata(**value)
        for key, value in dict(raw.get("datasets", {})).items()
    }
    return ProjectMetadata(
        project_id=str(raw["project_id"]),
        name=str(raw["name"]),
        root=str(raw["root"]),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        subjects=subjects,
        sessions=sessions,
        datasets=datasets,
    )


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_ID_RE.pattern}")


def _validate_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _utc_now() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    iso_text = timestamp.isoformat()
    return iso_text.replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise
