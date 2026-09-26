"""Knowledge-pack manifest: what a pack indexes and how much each source is trusted.

A manifest is a small YAML document::

    id: findings
    title: Portfolio findings
    chunk_chars: 1800
    sources:
      - repo: AffineDrift
        authority: published
        include: ["articles/**/*.qmd"]
        exclude: ["**/*-bibliography.md"]
    status_overrides:
      docs/assessments/old.md: superseded

Stdlib + PyYAML only: this package is vendored by Runner_Dashboard.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Authorities in descending trust; the index breaks score ties with this order.
AUTHORITIES: tuple[str, ...] = (
    "published",
    "findings",
    "reviews",
    "product",
    "reference",
    "notes",
)
#: Passage lifecycle states. Superseded and retracted passages are hidden by default.
STATUSES: tuple[str, ...] = ("current", "draft", "superseded", "retracted")
HIDDEN_STATUSES: frozenset[str] = frozenset({"superseded", "retracted"})

DEFAULT_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 20_000

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_TOP_KEYS = frozenset({"id", "title", "chunk_chars", "sources", "status_overrides"})
_SOURCE_KEYS = frozenset({"repo", "authority", "include", "exclude"})


class ManifestError(ValueError):
    """The manifest violates the knowledge-pack contract."""


@dataclass(frozen=True)
class SourceSpec:
    """One repository slice: files matching ``include`` minus ``exclude``."""

    repo: str
    authority: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackManifest:
    """Validated manifest; build it with :func:`manifest_from_dict`."""

    id: str
    title: str
    sources: tuple[SourceSpec, ...]
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    status_overrides: Mapping[str, str] = field(default_factory=dict)

    @property
    def repos(self) -> tuple[str, ...]:
        """Distinct repositories in manifest order."""
        return tuple(dict.fromkeys(s.repo for s in self.sources))

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form; ``manifest_from_dict(m.to_dict()) == m``."""
        return {
            "id": self.id,
            "title": self.title,
            "chunk_chars": self.chunk_chars,
            "sources": [
                {
                    "repo": s.repo,
                    "authority": s.authority,
                    "include": list(s.include),
                    "exclude": list(s.exclude),
                }
                for s in self.sources
            ],
            "status_overrides": dict(self.status_overrides),
        }


def load_manifest(path: Path) -> PackManifest:
    """Read and validate a YAML manifest."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{path}: manifest must be a mapping")
    return manifest_from_dict(raw)


def manifest_from_dict(raw: Mapping[str, Any]) -> PackManifest:
    """Validate plain data into a :class:`PackManifest` or raise ManifestError."""
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ManifestError(f"unknown manifest keys: {sorted(unknown)}")
    pack_id = raw.get("id")
    if not isinstance(pack_id, str) or not _ID_RE.match(pack_id):
        raise ManifestError("id must be lowercase letters, digits, '-' or '_'")
    title = raw.get("title", pack_id)
    if not isinstance(title, str) or not title.strip():
        raise ManifestError("title must be a non-empty string")
    chunk_chars = raw.get("chunk_chars", DEFAULT_CHUNK_CHARS)
    if (
        not isinstance(chunk_chars, int)
        or isinstance(chunk_chars, bool)
        or not MIN_CHUNK_CHARS <= chunk_chars <= MAX_CHUNK_CHARS
    ):
        raise ManifestError(
            f"chunk_chars must be an integer in [{MIN_CHUNK_CHARS}, {MAX_CHUNK_CHARS}]"
        )
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ManifestError("sources must be a non-empty list")
    return PackManifest(
        id=pack_id,
        title=title.strip(),
        sources=tuple(_source(i, s) for i, s in enumerate(sources)),
        chunk_chars=chunk_chars,
        status_overrides=_overrides(raw.get("status_overrides") or {}),
    )


def _source(index: int, raw: Any) -> SourceSpec:
    where = f"sources[{index}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where} must be a mapping")
    unknown = set(raw) - _SOURCE_KEYS
    if unknown:
        raise ManifestError(f"{where}: unknown keys {sorted(unknown)}")
    repo = raw.get("repo")
    if not isinstance(repo, str) or not _REPO_RE.match(repo) or ".." in repo:
        raise ManifestError(f"{where}.repo must be a bare repository name")
    authority = raw.get("authority")
    if authority not in AUTHORITIES:
        raise ManifestError(f"{where}.authority must be one of {list(AUTHORITIES)}")
    include = _globs(f"{where}.include", raw.get("include"))
    if not include:
        raise ManifestError(f"{where}.include must list at least one glob")
    exclude = _globs(f"{where}.exclude", raw.get("exclude") or [])
    return SourceSpec(repo=repo, authority=authority, include=include, exclude=exclude)


def _globs(where: str, raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(
        not isinstance(g, str) or not g.strip() for g in raw
    ):
        raise ManifestError(f"{where} must be a list of non-empty glob strings")
    for glob in raw:
        if glob.startswith("/") or ".." in glob.split("/"):
            raise ManifestError(
                f"{where}: glob {glob!r} must stay inside the repository"
            )
    return tuple(raw)


def _overrides(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ManifestError("status_overrides must map paths to statuses")
    for path, status in raw.items():
        if not isinstance(path, str) or status not in STATUSES:
            raise ManifestError(
                f"status_overrides[{path!r}]: status must be one of {list(STATUSES)}"
            )
    return dict(raw)
