"""Launch-monitor project/session aggregation and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import pandas as pd

from src.tools.launch_monitor_model.schema import ImportedSession, ImportManifest


class LaunchMonitorProject:
    """Named collection of imported sessions with deterministic persistence."""

    SCHEMA_VERSION = "launch-monitor-project-1.0"

    def __init__(self, name: str) -> None:
        if not name.strip():
            raise ValueError("project name must be non-empty")
        self.name = name.strip()
        self.sessions: list[ImportedSession] = []
        self.audit_log: list[dict[str, object]] = []

    def add_session(self, session: ImportedSession) -> None:
        """Add a session, rejecting duplicate session identities."""
        if any(existing.session_id == session.session_id for existing in self.sessions):
            raise ValueError(f"Session already exists: {session.session_id}")
        self.sessions.append(session)

    def remove_session(self, session_id: str) -> None:
        """Remove a session by id."""
        before = len(self.sessions)
        self.sessions = [
            item for item in self.sessions if item.session_id != session_id
        ]
        if len(self.sessions) == before:
            raise ValueError(f"Unknown session: {session_id}")

    def combined_shots(self) -> pd.DataFrame:
        """Return the union of all source and canonical columns."""
        if not self.sessions:
            return pd.DataFrame()
        return pd.concat(
            [session.shots for session in self.sessions],
            ignore_index=True,
            sort=False,
        )

    def record_actions(self, actions: tuple[dict[str, object], ...]) -> None:
        """Append treatment/filter actions to the durable project audit log."""
        self.audit_log.extend(dict(action) for action in actions)

    def save(self, destination: str | Path) -> Path:
        """Save the project as a portable JSON document."""
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "name": self.name,
            "audit_log": self.audit_log,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "name": session.name,
                    "manifest": asdict(session.manifest),
                    "metadata": session.metadata,
                    "shots_table": session.shots.to_json(
                        orient="table", date_format="iso", index=False
                    ),
                }
                for session in self.sessions
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, source: str | Path) -> LaunchMonitorProject:
        """Load and validate a saved project."""
        path = Path(source).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported launch-monitor project schema")
        project = cls(str(payload["name"]))
        project.audit_log = list(payload.get("audit_log", []))
        for item in payload.get("sessions", []):
            manifest_data = dict(item["manifest"])
            manifest_data["source_columns"] = tuple(manifest_data["source_columns"])
            manifest_data["warnings"] = tuple(manifest_data.get("warnings", ()))
            shots = pd.read_json(StringIO(item["shots_table"]), orient="table")
            project.add_session(
                ImportedSession(
                    session_id=str(item["session_id"]),
                    name=str(item["name"]),
                    shots=shots,
                    manifest=ImportManifest(**manifest_data),
                    source_path=Path(manifest_data["source_path"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return project
