"""Bridge for discovering and launching tools from the sibling Tools repository.

Reads ``tools.json`` from the Tools repository root (sibling directory or
``vendor/ud-tools/``) and exposes each entry as an :class:`ExternalTool`
that UpstreamDrift's launcher can render as a tile.

Design contracts
----------------
- :class:`ExternalTool` validates ``name`` and ``category`` are non-empty on
  construction (TypeError / ValueError).
- :func:`load_tools_from_repo` never raises — missing files / malformed JSON
  return an empty list and log a warning.
- :class:`ToolsRepoBridge` is stateless between calls; caching is the
  caller's responsibility.

Related issue: #5334.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default search paths ──────────────────────────────────────────────

_TOOLS_JSON_FILENAME = "tools.json"

# Resolved once to locate the UD repo root (two levels up from this file)
_UD_REPO_ROOT: Path = Path(__file__).resolve().parents[4]


# ── Domain model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExternalTool:
    """A launchable tool from the sibling Tools repository.

    Attributes:
        name:        Human-readable tool name.
        category:    Category used for sidebar grouping (e.g. "Robotics").
        description: One-line description shown in the launcher tile.
        launch_path: Relative path to the launch script inside the repo.
        repo_root:   Absolute path to the Tools repository root.
        repo_name:   Display name of the source repository.

    Raises:
        ValueError: If ``name`` or ``category`` is empty.
        TypeError:  If ``repo_root`` is not a :class:`~pathlib.Path`.
    """

    name: str
    category: str
    description: str
    launch_path: str
    repo_root: Path
    repo_name: str = "Tools"

    def __post_init__(self) -> None:
        if not isinstance(self.repo_root, Path):
            raise TypeError(
                f"repo_root must be a Path, got {type(self.repo_root).__name__}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("ExternalTool.name must be a non-empty string")
        if not self.category or not self.category.strip():
            raise ValueError("ExternalTool.category must be a non-empty string")

    @property
    def absolute_script_path(self) -> Path:
        """Absolute path to the launch script.

        Postcondition: returned path is always absolute.
        """
        return (self.repo_root / self.launch_path).resolve()

    @property
    def is_available(self) -> bool:
        """Return ``True`` when the launch script exists on disk."""
        return self.absolute_script_path.exists()


# ── Bridge ────────────────────────────────────────────────────────────


class ToolsRepoBridge:
    """Loads and filters :class:`ExternalTool` entries from ``tools.json``.

    Args:
        repo_root: Root directory of the Tools repository.
        repo_name: Display name used in :attr:`ExternalTool.repo_name`.

    Raises:
        TypeError: If ``repo_root`` is not a :class:`~pathlib.Path`.
    """

    def __init__(self, repo_root: Path, repo_name: str = "Tools") -> None:
        if not isinstance(repo_root, Path):
            raise TypeError(f"repo_root must be a Path, got {type(repo_root).__name__}")
        self._repo_root = repo_root
        self._repo_name = repo_name

    @property
    def tools_json_path(self) -> Path:
        """Absolute path to ``tools.json``."""
        return self._repo_root / _TOOLS_JSON_FILENAME

    def _parse_tools_json(self) -> dict[str, Any]:
        """Read and parse ``tools.json``; return empty dict on any failure."""
        path = self.tools_json_path
        if not path.exists():
            logger.debug("tools.json not found at %s", path)
            return {}
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("tools.json at %s is not a JSON object", path)
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tools.json from %s: %s", path, exc)
            return {}

    def load(self, category_filter: str | None = None) -> list[ExternalTool]:
        """Return :class:`ExternalTool` instances from the repository.

        Args:
            category_filter: When given, only tools in that category are
                returned (exact match).

        Returns:
            Sorted list of :class:`ExternalTool` objects.  Empty when
            ``tools.json`` is missing or malformed.
        """
        raw = self._parse_tools_json()
        tools: list[ExternalTool] = []

        for category, entries in raw.items():
            if category_filter is not None and category != category_filter:
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                path = entry.get("path", "")
                desc = entry.get("desc", "")
                if not name or not path:
                    logger.debug("Skipping incomplete entry in category %s", category)
                    continue
                try:
                    tools.append(
                        ExternalTool(
                            name=name,
                            category=category,
                            description=desc,
                            launch_path=path,
                            repo_root=self._repo_root,
                            repo_name=self._repo_name,
                        )
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping invalid tool entry %r: %s", name, exc)

        return sorted(tools, key=lambda t: (t.category, t.name))

    def list_categories(self) -> list[str]:
        """Return sorted list of categories present in ``tools.json``."""
        raw = self._parse_tools_json()
        return sorted(raw.keys())


# ── Convenience function ──────────────────────────────────────────────


def load_tools_from_repo(
    repo_root: Path | None = None,
    ud_root: Path | None = None,
    repo_name: str = "Tools",
) -> list[ExternalTool]:
    """Discover and load Tools-repo entries without constructing a bridge object.

    Discovery order:
    1. ``repo_root`` (explicit, highest priority).
    2. Sibling ``Tools/`` directory next to ``ud_root`` (or next to the
       UpstreamDrift repo if ``ud_root`` is ``None``).
    3. ``vendor/ud-tools/`` inside the UpstreamDrift repo.

    Args:
        repo_root: Explicit path to the Tools repository.
        ud_root:   Root of the UpstreamDrift repo, used to locate siblings.
                   Defaults to the repo root inferred from this file's location.
        repo_name: Display name used in :attr:`ExternalTool.repo_name`.

    Returns:
        List of :class:`ExternalTool` objects; empty on any discovery failure.
    """
    if repo_root is not None:
        bridge = ToolsRepoBridge(repo_root=repo_root, repo_name=repo_name)
        return bridge.load()

    effective_ud_root = ud_root if ud_root is not None else _UD_REPO_ROOT

    # Try sibling Tools/ directory
    sibling = effective_ud_root.parent / "Tools"
    if sibling.is_dir() and (sibling / _TOOLS_JSON_FILENAME).exists():
        return ToolsRepoBridge(repo_root=sibling, repo_name=repo_name).load()

    # Fall back to vendor/ud-tools/
    vendor = effective_ud_root / "vendor" / "ud-tools"
    if vendor.is_dir() and (vendor / _TOOLS_JSON_FILENAME).exists():
        return ToolsRepoBridge(repo_root=vendor, repo_name=repo_name).load()

    logger.debug(
        "Tools repository not found (tried sibling %s and vendor %s)", sibling, vendor
    )
    return []
