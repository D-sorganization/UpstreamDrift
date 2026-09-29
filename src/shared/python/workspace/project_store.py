"""Durable project/session metadata for the unified workspace."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from src.shared.python.core.contracts.exceptions import StateError
from .artifact_handoff import (
    ArtifactKind,
    ArtifactReference,
    SUPPORTED_FRAMES,
    SUPPORTED_SCHEMAS,
    WorkspaceHandoff,
)

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
class RunMetadata:
    """A versioned run carrying parameters and typed artifact references."""

    run_id: str
    project_id: str
    session_id: str
    subject_id: str
    engine: str
    model_id: str
    club: dict[str, Any]
    units: dict[str, str]
    frame: str
    timebase: dict[str, Any]
    parameters: dict[str, Any]
    inputs: tuple[ArtifactReference, ...] = ()
    outputs: tuple[ArtifactReference, ...] = ()
    status: str = "draft"
    qualification: dict[str, Any] | None = None
    engine_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: _utc_now())

    def to_handoff(self) -> WorkspaceHandoff:
        """Export this run as a versioned WorkspaceHandoff."""
        return WorkspaceHandoff(
            handoff_id=self.run_id,
            project_id=self.project_id,
            session_id=self.session_id,
            subject_id=self.subject_id,
            engine=self.engine,
            model_id=self.model_id,
            club=self.club,
            units=self.units,
            frame=self.frame,
            timebase=self.timebase,
            parameters=self.parameters,
            inputs=self.inputs,
            outputs=self.outputs,
            status=self.status,
            qualification=self.qualification,
            engine_version=self.engine_version,
            metadata=self.metadata,
            schema_version=self.schema_version,
            created_at=self.created_at,
        )

    @classmethod
    def from_handoff(cls, handoff: WorkspaceHandoff) -> RunMetadata:
        """Construct a RunMetadata instance from a WorkspaceHandoff."""
        return cls(
            run_id=handoff.handoff_id,
            project_id=handoff.project_id,
            session_id=handoff.session_id,
            subject_id=handoff.subject_id,
            engine=handoff.engine,
            model_id=handoff.model_id,
            club=dict(handoff.club),
            units=dict(handoff.units),
            frame=handoff.frame,
            timebase=dict(handoff.timebase),
            parameters=dict(handoff.parameters),
            inputs=handoff.inputs,
            outputs=handoff.outputs,
            status=handoff.status,
            qualification=handoff.qualification,
            engine_version=handoff.engine_version,
            metadata=dict(handoff.metadata),
            schema_version=handoff.schema_version,
            created_at=handoff.created_at,
        )


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
    runs: dict[str, RunMetadata] = field(default_factory=dict)
    active_run_id: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class SessionProjectStore:
    """JSON-backed project/session/dataset/run metadata store.

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

    # ------------------------------------------------------------------------
    # Run & Artifact Handoff Management
    # ------------------------------------------------------------------------

    def register_run(
        self, handoff_or_run: WorkspaceHandoff | RunMetadata
    ) -> RunMetadata:
        """Register and persist a new run / handoff into project metadata.

        Enforces:
        - Session exists in project
        - Subject identity matches session's subject (rejects cross-session subject mismatch)
        - Frame and schemas are known and supported
        - Referenced artifacts exist on disk and match recorded cryptographic hashes
        - Canceled/failed run cannot be marked with completed qualification
        """
        if isinstance(handoff_or_run, WorkspaceHandoff):
            run = RunMetadata.from_handoff(handoff_or_run)
        elif isinstance(handoff_or_run, RunMetadata):
            run = handoff_or_run
        else:
            raise TypeError(
                f"handoff_or_run must be WorkspaceHandoff or RunMetadata, got {type(handoff_or_run).__name__}"
            )

        _validate_id(run.run_id, "run_id")
        project = self.load_project()
        if run.session_id not in project.sessions:
            raise KeyError(f"unknown session_id: {run.session_id}")
        session = project.sessions[run.session_id]

        # DbC: Cross-session subject mismatch check
        if session.subject_id != run.subject_id:
            raise ValueError(
                f"cross-session subject mismatch: session '{run.session_id}' "
                f"belongs to subject '{session.subject_id}', got '{run.subject_id}'"
            )

        # DbC: Frame and schema validation
        if run.frame not in SUPPORTED_FRAMES:
            raise ValueError(
                f"unknown or unsupported frame: {run.frame!r}; "
                f"supported frames are {sorted(SUPPORTED_FRAMES)}"
            )

        for art in (*run.inputs, *run.outputs):
            if art.schema not in SUPPORTED_SCHEMAS:
                raise ValueError(
                    f"unknown or unsupported schema: {art.schema!r}; "
                    f"supported schemas are {sorted(SUPPORTED_SCHEMAS)}"
                )

        # DbC: Artifact presence and hash verification BEFORE writing to disk
        for art in (*run.inputs, *run.outputs):
            art.verify_on_disk(self._root)

        # DbC: Invariant: canceled or failed jobs never become completed results
        if run.status in {"failed", "canceled"} and run.qualification is not None:
            if run.qualification.get("passed", False):
                raise ValueError(
                    f"{run.status} run cannot be marked with completed qualification"
                )

        runs = dict(project.runs)
        runs[run.run_id] = run
        self._save(_replace_project(project, runs=runs))
        return run

    def load_run(self, run_id: str) -> RunMetadata:
        """Load a run by run_id."""
        _validate_id(run_id, "run_id")
        project = self.load_project()
        if run_id not in project.runs:
            raise KeyError(f"unknown run_id: {run_id}")
        return project.runs[run_id]

    def list_runs(self, session_id: str | None = None) -> list[RunMetadata]:
        """Return runs ordered by creation time, optionally scoped to one session."""
        if session_id is not None:
            _validate_id(session_id, "session_id")
        runs = list(self.load_project().runs.values())
        if session_id is not None:
            runs = [run for run in runs if run.session_id == session_id]
        return sorted(runs, key=lambda r: (r.created_at, r.run_id))

    def set_active_run(self, run_id: str | None) -> None:
        """Select the active run context."""
        project = self.load_project()
        if run_id is not None:
            _validate_id(run_id, "run_id")
            if run_id not in project.runs:
                raise KeyError(f"unknown run_id: {run_id}")
        self._save(_replace_project(project, active_run_id=run_id))

    def get_active_run(self) -> RunMetadata | None:
        """Return the active run if set and present, else None."""
        project = self.load_project()
        if project.active_run_id is None:
            return None
        return project.runs.get(project.active_run_id)

    def clone_run(
        self,
        source_run_id: str,
        new_run_id: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> RunMetadata:
        """Clone run context to a new run ID without overwriting source evidence.

        Postcondition:
        - new_run_id has source inputs, parameters, and configuration
        - outputs and completion qualification are NOT copied over
        - source run evidence remains completely unchanged
        """
        _validate_id(new_run_id, "new_run_id")
        project = self.load_project()
        if source_run_id not in project.runs:
            raise KeyError(f"unknown source_run_id: {source_run_id}")
        if new_run_id in project.runs:
            raise StateError(f"run already exists: {new_run_id}")

        source = project.runs[source_run_id]
        merged_params = dict(source.parameters)
        if parameters is not None:
            merged_params.update(parameters)

        cloned = RunMetadata(
            run_id=new_run_id,
            project_id=source.project_id,
            session_id=source.session_id,
            subject_id=source.subject_id,
            engine=source.engine,
            model_id=source.model_id,
            club=dict(source.club),
            units=dict(source.units),
            frame=source.frame,
            timebase=dict(source.timebase),
            parameters=merged_params,
            inputs=source.inputs,
            outputs=(),
            status="draft",
            qualification=None,
            engine_version=source.engine_version,
            metadata=dict(source.metadata),
            schema_version=source.schema_version,
            created_at=_utc_now(),
        )

        runs = dict(project.runs)
        runs[new_run_id] = cloned
        self._save(_replace_project(project, runs=runs))
        return cloned

    def check_run_artifacts(self, run_id: str) -> list[str]:
        """Check all referenced artifacts for a run and explain missing files."""
        run = self.load_run(run_id)
        missing: list[str] = []
        for art in (*run.inputs, *run.outputs):
            resolved = art.resolve_path(self._root)
            if not resolved.exists() or not resolved.is_file():
                kind_str = (
                    art.kind.value
                    if isinstance(art.kind, ArtifactKind)
                    else str(art.kind)
                )
                missing.append(
                    f"Artifact '{art.artifact_id}' ({kind_str}) file is missing at '{resolved}'. "
                    f"No substitute was invented."
                )
        return missing

    def export_handoff(self, run_id: str) -> WorkspaceHandoff:
        """Export a run as a WorkspaceHandoff."""
        return self.load_run(run_id).to_handoff()

    def import_handoff(self, handoff: WorkspaceHandoff) -> RunMetadata:
        """Import a WorkspaceHandoff into the project store."""
        return self.register_run(handoff)

    def _save(self, project: ProjectMetadata) -> None:
        now = _utc_now()
        proj_updated = _replace_project(project, updated_at=now)
        payload = asdict(proj_updated)
        payload.pop("extra_fields", None)
        # Merge back any preserved extra fields from prior migration
        for k, v in proj_updated.extra_fields.items():
            if k not in payload:
                payload[k] = v
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self._path, payload)
        except OSError as exc:
            raise StateError(f"could not write project metadata: {self._path}") from exc


def _replace_project(
    project: ProjectMetadata,
    **changes: Any,
) -> ProjectMetadata:
    return replace(project, **changes)


def _project_from_dict(raw: dict[str, Any]) -> ProjectMetadata:
    known_keys = {
        "project_id",
        "name",
        "root",
        "created_at",
        "updated_at",
        "subjects",
        "sessions",
        "datasets",
        "runs",
        "active_run_id",
    }
    extra_fields = {k: v for k, v in raw.items() if k not in known_keys}
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
    runs = {
        key: _run_from_dict(value) for key, value in dict(raw.get("runs", {})).items()
    }
    active_run_id = raw.get("active_run_id")
    if active_run_id is not None:
        active_run_id = str(active_run_id)

    return ProjectMetadata(
        project_id=str(raw["project_id"]),
        name=str(raw["name"]),
        root=str(raw["root"]),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        subjects=subjects,
        sessions=sessions,
        datasets=datasets,
        runs=runs,
        active_run_id=active_run_id,
        extra_fields=extra_fields,
    )


def _run_from_dict(raw: dict[str, Any]) -> RunMetadata:
    inputs = tuple(
        ArtifactReference(
            artifact_id=str(art["artifact_id"]),
            path=str(art["path"]),
            hash=str(art["hash"]),
            schema=str(art["schema"]),
            kind=str(art["kind"]),
            metadata=dict(art.get("metadata", {})),
        )
        for art in raw.get("inputs", ())
    )
    outputs = tuple(
        ArtifactReference(
            artifact_id=str(art["artifact_id"]),
            path=str(art["path"]),
            hash=str(art["hash"]),
            schema=str(art["schema"]),
            kind=str(art["kind"]),
            metadata=dict(art.get("metadata", {})),
        )
        for art in raw.get("outputs", ())
    )
    return RunMetadata(
        run_id=str(raw["run_id"]),
        project_id=str(raw["project_id"]),
        session_id=str(raw["session_id"]),
        subject_id=str(raw["subject_id"]),
        engine=str(raw["engine"]),
        model_id=str(raw["model_id"]),
        club=dict(raw.get("club", {})),
        units=dict(raw.get("units", {})),
        frame=str(raw["frame"]),
        timebase=dict(raw.get("timebase", {})),
        parameters=dict(raw.get("parameters", {})),
        inputs=inputs,
        outputs=outputs,
        status=str(raw.get("status", "draft")),
        qualification=raw.get("qualification"),
        engine_version=raw.get("engine_version"),
        metadata=dict(raw.get("metadata", {})),
        schema_version=str(raw.get("schema_version", "1.0.0")),
        created_at=str(raw["created_at"]),
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
