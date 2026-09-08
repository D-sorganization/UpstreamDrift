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

INGEST_INDEX_FILE = (
    "observations.json"  # rig.ingest.INGEST_INDEX_FILE; not imported (pose stack)
)
from src.motion_capture.rig.proxy import PROXIES_FILE, ProxiesIndex
from src.motion_capture.variants import list_variants, variant_dir
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
    observation_sets: dict[str, Path] | None = None  # {set dir name: view file}

    @property
    def playable(self) -> Path | None:
        """The proxy when it exists (browser-friendly H.264), else the recording."""
        return self.proxy or self.recording


@dataclass(frozen=True)
class VariantMedia:
    """One registered match of the session (#9793)."""

    name: str
    views: tuple[str, ...]
    observation_set: str
    source: dict[str, Any]
    root: Path
    has_reconstruction: bool
    has_model_fit: bool

    @property
    def label(self) -> str:
        return self.name or "(default)"


@dataclass(frozen=True)
class SessionMedia:
    root: Path
    plan_name: str
    views: tuple[ViewMedia, ...]
    swing_summary: dict[str, Any] | None
    reconstruction: dict[str, Any] | None
    problems: tuple[str, ...]
    intrinsics: Path | None = None
    reliability: dict[str, Any] | None = None
    analysis_2d: dict[str, dict[str, Any]] | None = None
    export: Path | None = None
    observation_sets: tuple[str, ...] = ()
    model_fit: dict[str, Any] | None = None
    model_comparison: dict[str, Any] | None = None
    kinetics: dict[str, Any] | None = None
    variants: tuple[VariantMedia, ...] = ()

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


def _observation_sets(root: Path) -> dict[str, dict[str, Path]]:
    """``{set name: {view: file}}`` for every ``observations*`` directory."""
    out: dict[str, dict[str, Path]] = {}
    for d in sorted(root.glob("observations*")):
        index = _read_json(d / INGEST_INDEX_FILE) if d.is_dir() else None
        if index is None:
            continue
        views = {
            row["view"]: d / row["file"]
            for row in index.get("views", [])
            if row.get("status") == "available" and row.get("file")
        }
        if views:
            out[d.name] = views
    return out


def _analysis_2d(root: Path) -> dict[str, dict[str, Any]] | None:
    d = root / "analysis_2d"
    if not d.is_dir():
        return None
    out = {}
    for path in sorted(d.glob("*.json")):
        payload = _read_json(path)
        if payload is not None:
            out[path.stem] = payload
    return out or None


def _kinetics_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The kinetics record without its per-frame torque table (for the tile)."""
    if payload is None:
        return None
    return {k: v for k, v in payload.items() if k not in ("tau", "dof_names")}


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _rate(entry: RecordingEntry) -> float | None:
    return entry.achieved_fps or float(entry.requested_mode.fps)


def load_session(root: Path) -> SessionMedia:
    """Everything the tool can show for a bundle; raises ``ValueError`` if not a bundle."""
    require(root.is_dir(), "session must be a directory", str(root))
    plan, index, _ = load_bundle(root)
    proxies, observations = _proxies(root), _observations(root)
    sets = _observation_sets(root)
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
                observation_sets={
                    name: files[entry.view]
                    for name, files in sets.items()
                    if entry.view in files
                }
                or None,
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
        intrinsics=_existing(root / "intrinsics.json"),
        reliability=_read_json(root / "reliability.json"),
        analysis_2d=_analysis_2d(root),
        export=_existing(recon / "reconstruction.trc"),
        observation_sets=tuple(sets),
        model_fit=_read_json(root / "model" / "fit_report.json"),
        model_comparison=_read_json(root / "model" / "comparison.json"),
        kinetics=_kinetics_summary(_read_json(root / "model" / "kinetics.json")),
        variants=_variants(root),
    )


def _variants(root: Path) -> tuple[VariantMedia, ...]:
    out = []
    for record in list_variants(root):
        vroot = variant_dir(root, record.name)
        out.append(
            VariantMedia(
                name=record.name,
                views=record.views,
                observation_set=record.observation_set,
                source=dict(record.source),
                root=vroot,
                has_reconstruction=(
                    vroot / RECONSTRUCT_DIR / "joints_3d_m.npy"
                ).is_file(),
                has_model_fit=(vroot / "model" / "joint_angles.json").is_file(),
            )
        )
    return tuple(out)


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
