"""Provenance for every JSON the motion-capture pipeline writes (#9792).

One block, one shape, stamped by every writer::

    "provenance": {
        "created_utc": "2026-09-08T10:15:00Z",
        "generated_by": {"package": "upstreamdrift", "module": "...",
                         "version": "...", "git_sha": "..." | null},
        "inputs": [{"path": "observations/face_on.json", "sha256": "...",
                    "bytes": 1234, "schema_version": "view-observations/1.0.0"}],
        "parameters": {...},
        "derived_from": ["reconstruct/session_reconstruction.json"]
    }

``inputs`` are the files whose bytes the output depends on (hashed);
``derived_from`` names the outputs whose provenance continues the chain, so
:func:`lineage` can walk from a model fit back to the recordings and the
detector plug-in that produced the observations. Paths are stored relative
to ``base`` (the session) when they lie inside it, so a moved session keeps
its lineage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import require
from src.shared.python.core.process_safety import narrow_catch

PACKAGE = "upstreamdrift"
STANDARD_KEYS = frozenset(
    {"created_utc", "generated_by", "inputs", "parameters", "derived_from"}
)
MAX_DEPTH = 32
_GIT_CACHE: dict[str, str | None] = {}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes. Precondition: the file exists."""
    require(path.is_file(), "file to hash must exist", str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(repo_root: Path | None = None) -> str | None:
    """``git rev-parse HEAD`` of the checkout, or ``None`` when unavailable."""
    root = repo_root or Path(__file__).resolve().parents[2]
    key = str(root)
    if key not in _GIT_CACHE:
        _GIT_CACHE[key] = None
        with narrow_catch(
            OSError, subprocess.CalledProcessError, log_message="git sha"
        ):
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            _GIT_CACHE[key] = out.stdout.strip() or None
    return _GIT_CACHE[key]


def package_version() -> str:
    with narrow_catch(importlib.metadata.PackageNotFoundError, log_message="ver"):
        return importlib.metadata.version("upstream-drift")
    return "unknown"


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relative_to(path: Path, base: Path | None) -> str:
    """``path`` relative to ``base`` when inside it, else absolute."""
    resolved = path.resolve()
    if base is not None:
        root = base.resolve()
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def _schema_of(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    with narrow_catch(OSError, ValueError, log_message="schema probe"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            value = payload.get("schema_version")
            return str(value) if value is not None else None
    return None


def input_record(path: Path, base: Path | None = None) -> dict[str, Any]:
    """``{path, sha256, bytes, schema_version}`` for one input file."""
    return {
        "path": relative_to(path, base),
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "schema_version": _schema_of(path),
    }


def stamp(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    module: str,
    inputs: Sequence[Path] = (),
    parameters: Mapping[str, Any] | None = None,
    derived_from: Sequence[Path] = (),
    base: Path | None = None,
) -> dict[str, Any]:
    """A copy of ``payload`` with ``schema_version`` (if absent) and ``provenance``.

    An existing ``provenance`` dict (a detector's own record) is kept: its
    keys stay and are also folded into ``parameters`` so the lineage shows
    them. Preconditions: non-empty schema version and module; every input
    and derived-from path exists. Postcondition: ``payload`` is not mutated.
    """
    require(schema_version.strip() != "", "schema_version must be given")
    require(module.strip() != "", "module must be given")
    for path in (*inputs, *derived_from):
        require(path.exists(), "provenance path must exist", str(path))
    out = dict(payload)
    out.setdefault("schema_version", schema_version)
    existing = out.get("provenance")
    legacy = dict(existing) if isinstance(existing, dict) else {}
    params = {**{k: v for k, v in legacy.items() if k not in STANDARD_KEYS}}
    params.update(parameters or {})
    out["provenance"] = {
        **legacy,
        "created_utc": now_utc(),
        "generated_by": {
            "package": PACKAGE,
            "module": module,
            "version": package_version(),
            "git_sha": git_sha(),
        },
        "inputs": [input_record(p, base) for p in inputs],
        "parameters": params,
        "derived_from": [relative_to(p, base) for p in derived_from],
    }
    return out


def write_stamped(path: Path, payload: Mapping[str, Any], **kwargs: Any) -> Path:
    """``write_json(path, stamp(payload, **kwargs))``."""
    return write_json(path, stamp(payload, **kwargs))


@dataclass(frozen=True)
class LineageRecord:
    """One hop of a lineage: a file and what its provenance says about it."""

    path: str
    schema_version: str | None
    created_utc: str | None
    generated_by: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[dict[str, Any], ...] = ()
    derived_from: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return self.created_utc is None


def _resolve(name: str, base: Path) -> Path:
    candidate = Path(name)
    return candidate if candidate.is_absolute() else base / candidate


def _record(path: Path, base: Path) -> LineageRecord:
    payload: Any = None
    if path.suffix.lower() == ".json" and path.is_file():
        with narrow_catch(OSError, ValueError, log_message="lineage read"):
            payload = json.loads(path.read_text(encoding="utf-8"))
    prov = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(prov, dict):
        return LineageRecord(
            path=relative_to(path, base),
            schema_version=_schema_of(path) if path.is_file() else None,
            created_utc=None,
        )
    return LineageRecord(
        path=relative_to(path, base),
        schema_version=payload.get("schema_version"),
        created_utc=prov.get("created_utc"),
        generated_by=dict(prov.get("generated_by") or {}),
        parameters=dict(prov.get("parameters") or {}),
        inputs=tuple(prov.get("inputs") or ()),
        derived_from=tuple(prov.get("derived_from") or ()),
    )


def lineage(
    path: Path, *, base: Path | None = None, max_depth: int = MAX_DEPTH
) -> list[LineageRecord]:
    """Records from ``path`` back through ``derived_from`` and ``inputs``.

    Depth-first, each file once, nearest first. Preconditions: the file
    exists; ``max_depth`` positive. A cycle raises ``ValueError``.
    """
    require(path.is_file(), "lineage root must exist", str(path))
    require(max_depth > 0, "max_depth must be positive", max_depth)
    root = (base or path.parent).resolve()
    out: list[LineageRecord] = []
    seen: set[str] = set()
    stack: list[tuple[Path, int, tuple[str, ...]]] = [(path.resolve(), 0, ())]
    while stack:
        current, depth, trail = stack.pop()
        key = current.as_posix()
        if key in trail:
            raise ValueError(f"provenance cycle at {relative_to(current, root)}")
        if key in seen:
            continue
        seen.add(key)
        record = _record(current, root)
        out.append(record)
        if depth >= max_depth:
            continue
        nexts = [*record.derived_from, *(i["path"] for i in record.inputs)]
        for name in reversed(nexts):
            stack.append((_resolve(name, root).resolve(), depth + 1, (*trail, key)))
    return out


def lineage_markdown(records: Sequence[LineageRecord]) -> str:
    """One bullet per hop, nearest first: what, when, by which tool, parameters."""
    lines = []
    for r in records:
        if r.is_leaf:
            lines.append(
                f"- `{r.path}` (leaf{', ' + r.schema_version if r.schema_version else ''})"
            )
            continue
        by = r.generated_by
        tool = (
            f"{by.get('package', '?')} {by.get('module', '?')} {by.get('version', '?')}"
        )
        sha = by.get("git_sha")
        tool += f" @{sha[:9]}" if isinstance(sha, str) else ""
        params = ", ".join(f"{k}={_short(v)}" for k, v in sorted(r.parameters.items()))
        lines.append(
            f"- `{r.path}` [{r.schema_version}] {r.created_utc} by {tool}"
            + (f" — {params}" if params else "")
        )
    return "\n".join(lines) + "\n"


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as indented JSON; returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
