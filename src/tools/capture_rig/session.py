"""What a session bundle offers the tool: media, observations, results.

Read-only view over the files the rig commands write (``recordings.json``,
``proxies.json``, ``observations/``, ``reconstruct/``). Nothing here decides
anything about the data; it reports what exists and where, so the widgets
stay ignorant of the bundle layout (Law of Demeter).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.motion_capture.rig.bundle import RecordingEntry, load_bundle
from src.motion_capture.rig.ingest import INGEST_INDEX_FILE
from src.motion_capture.rig.proxy import PROXIES_FILE, ProxiesIndex
from src.shared.python.core.contracts import require

OBSERVATIONS_DIR = "observations"
RECONSTRUCT_DIR = "reconstruct"
SWING_SUMMARY_FILE = "swing_summary.json"
SESSION_RECONSTRUCTION_FILE = "session_reconstruction.json"


@dataclass(frozen=True)
class ViewMedia:
    """One view's files; any of them may be missing."""

    view: str
    identity: str
    recording: Path | None
    proxy: Path | None
    observations: Path | None
    fps: float | None

    @property
    def playable(self) -> Path | None:
        """The proxy when it exists (browser-friendly H.264), else the recording."""
        return self.proxy or self.recording


@dataclass(frozen=True)
class SessionMedia:
    root: Path
    plan_name: str
    views: tuple[ViewMedia, ...]
    swing_summary: dict[str, Any] | None
    reconstruction: dict[str, Any] | None
    problems: tuple[str, ...]

    @property
    def ingested(self) -> bool:
        return any(v.observations is not None for v in self.views)

    def view(self, name: str) -> ViewMedia:
        for v in self.views:
            if v.view == name:
                return v
        raise KeyError(name)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _proxies(root: Path) -> dict[str, Path]:
    path = root / PROXIES_FILE
    if not path.is_file():
        return {}
    index = ProxiesIndex.model_validate_json(path.read_text(encoding="utf-8"))
    return {p.view: root / p.file for p in index.proxies if p.ok and p.file}


def _observations(root: Path) -> dict[str, Path]:
    index = _read_json(root / OBSERVATIONS_DIR / INGEST_INDEX_FILE)
    if index is None:
        return {}
    out = {}
    for row in index.get("views", []):
        if row.get("status") == "available" and row.get("file"):
            out[row["view"]] = root / OBSERVATIONS_DIR / row["file"]
    return out


def _rate(entry: RecordingEntry) -> float | None:
    return entry.achieved_fps or float(entry.requested_mode.fps)


def load_session(root: Path) -> SessionMedia:
    """Everything the tool can show for a bundle; raises ``ValueError`` if not a bundle."""
    require(root.is_dir(), "session must be a directory", str(root))
    plan, index, _ = load_bundle(root)
    proxies, observations = _proxies(root), _observations(root)
    problems: list[str] = []
    views = []
    for entry in index.recordings:
        if not entry.ok:
            problems.append(f"{entry.view}: {entry.recorder_note or 'no recording'}")
        views.append(
            ViewMedia(
                view=entry.view,
                identity=entry.identity,
                recording=root / entry.file if entry.ok else None,
                proxy=proxies.get(entry.view),
                observations=observations.get(entry.view),
                fps=_rate(entry) if entry.ok else None,
            )
        )
    recon = root / RECONSTRUCT_DIR
    return SessionMedia(
        root=root,
        plan_name=plan.name,
        views=tuple(views),
        swing_summary=_read_json(recon / SWING_SUMMARY_FILE),
        reconstruction=_read_json(recon / SESSION_RECONSTRUCTION_FILE),
        problems=tuple(problems),
    )


def flatten_numbers(payload: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """``(dotted key, value)`` rows for scalar leaves, for a key/value table."""
    rows: list[tuple[str, str]] = []
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(flatten_numbers(value, f"{name}."))
        elif isinstance(value, float):
            rows.append((name, f"{value:.4g}"))
        elif isinstance(value, bool | int | str) or value is None:
            rows.append((name, str(value)))
    return rows
