"""Wizards: a product's own knowledge pack, retrieved for each chat turn.

A product opts in with ``knowledge/wizard.yml`` at its project root::

    key: upstream_drift            # the Sidekick app_context it speaks for
    name: UpstreamDrift Wizard
    description: the expert on UpstreamDrift's features, models and results
    capabilities: [...]            # optional, shown in the system prompt
    manifest: knowledge/pack.yml   # default
    pack: .knowledge/pack.sqlite   # default; built by ``knowledge build``
    k: 5                           # passages per turn
    roots: {UpstreamDrift: .}      # default: every manifest repo is the host

Like the rest of this package it needs only the stdlib and PyYAML.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .manifest import ManifestError, load_manifest
from .pack import KnowledgePack, PackFormatError, Passage

WIZARD_FILE = Path("knowledge") / "wizard.yml"
DEFAULT_MANIFEST = "knowledge/pack.yml"
DEFAULT_PACK = ".knowledge/pack.sqlite"
DEFAULT_K = 5
MAX_K = 20
#: Seconds a freshness verdict is reused; hashing a product's docs every turn is waste.
FRESHNESS_TTL_S = 300.0
STALE_BANNER = (
    "Note: this product has changed since its knowledge pack was built, so "
    "the passages below may be out of date. Say so when it matters, and "
    "rebuild the pack with `python -m shared.python.ai.knowledge build`."
)

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_KEYS = frozenset(
    {"key", "name", "description", "capabilities", "manifest", "pack", "k", "roots"}
)


class WizardConfigError(ValueError):
    """``knowledge/wizard.yml`` violates the Wizard contract."""


@dataclass(frozen=True)
class WizardConfig:
    """Validated Wizard settings with paths resolved against the host root."""

    key: str
    name: str
    description: str
    capabilities: tuple[str, ...]
    manifest: Path
    pack: Path
    k: int
    roots: Mapping[str, Path] = field(default_factory=dict)


def load_wizard_config(project_root: Path) -> WizardConfig | None:
    """Read ``<project_root>/knowledge/wizard.yml``; None when the product has none."""
    root = Path(project_root)
    path = root / WIZARD_FILE
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise WizardConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise WizardConfigError(f"{path}: must be a mapping")
    unknown = set(raw) - _KEYS
    if unknown:
        raise WizardConfigError(f"{path}: unknown keys {sorted(unknown)}")
    key = raw.get("key")
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise WizardConfigError("key must be a lowercase app_context identifier")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise WizardConfigError("name must be a non-empty string")
    k = raw.get("k", DEFAULT_K)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        raise WizardConfigError(f"k must be an integer in [1, {MAX_K}]")
    capabilities = raw.get("capabilities") or []
    if not isinstance(capabilities, list) or not all(
        isinstance(c, str) for c in capabilities
    ):
        raise WizardConfigError("capabilities must be a list of strings")
    manifest = root / str(raw.get("manifest", DEFAULT_MANIFEST))
    return WizardConfig(
        key=key,
        name=name.strip(),
        description=str(raw.get("description") or f"the expert on {name.strip()}"),
        capabilities=tuple(capabilities),
        manifest=manifest,
        pack=root / str(raw.get("pack", DEFAULT_PACK)),
        k=k,
        roots=_roots(root, raw.get("roots"), manifest),
    )


def _roots(root: Path, raw: Any, manifest: Path) -> dict[str, Path]:
    if raw is None:
        try:
            repos = load_manifest(manifest).repos if manifest.is_file() else ()
        except ManifestError:
            repos = ()
        return {repo: root for repo in repos}
    if not isinstance(raw, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        raise WizardConfigError("roots must map repository names to paths")
    return {repo: (root / rel).resolve() for repo, rel in raw.items()}


@dataclass(frozen=True)
class KnowledgeContext:
    """What one chat turn learns from the Wizard's pack."""

    wizard: str
    passages: tuple[Passage, ...]
    stale: bool | None

    def render(self) -> str:
        """System-prompt section: numbered, cited passages plus the stale banner."""
        lines = [
            f"Knowledge from {self.wizard} (cite as [n]; answer from these first):"
        ]
        if self.stale:
            lines.append(STALE_BANNER)
        for i, passage in enumerate(self.passages, start=1):
            heading = f" — {passage.title}" if passage.title else ""
            lines.append(f"[{i}] {passage.citation}{heading}\n{passage.text}")
        if not self.passages:
            lines.append("No passage in the pack matches this question.")
        return "\n\n".join(lines)


class WizardKnowledge:
    """Retrieval for one product. Cheap to construct; the pack opens lazily."""

    def __init__(
        self,
        config: WizardConfig,
        clock: Callable[[], float] = time.monotonic,
        freshness_ttl: float = FRESHNESS_TTL_S,
    ) -> None:
        if not isinstance(config, WizardConfig):
            raise TypeError("config must be a WizardConfig")
        self.config = config
        self._clock = clock
        self._ttl = freshness_ttl
        self._checked_at: float | None = None
        self._stale: bool | None = None

    @property
    def available(self) -> bool:
        return self.config.pack.is_file()

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Current passages for ``query``; empty when the pack is unusable."""
        if not self.available:
            return []
        try:
            return KnowledgePack.open(self.config.pack).search(
                query, k or self.config.k
            )
        except PackFormatError:
            return []

    def context_for(self, query: str) -> KnowledgeContext | None:
        """The turn's knowledge; None when there is nothing worth adding."""
        if not self.available:
            return None
        passages = tuple(self.search(query))
        stale = self.is_stale()
        if not passages and not stale:
            return None
        return KnowledgeContext(wizard=self.config.name, passages=passages, stale=stale)

    def is_stale(self) -> bool | None:
        """Pack freshness, re-checked at most once per TTL; None when unknowable."""
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < self._ttl:
            return self._stale
        self._checked_at = now
        try:
            pack = KnowledgePack.open(self.config.pack)
            self._stale = pack.is_stale(self.config.roots)
        except (OSError, ValueError, PackFormatError):
            self._stale = None
        return self._stale
