"""Data-driven GUI registration loader for the Tools platform.

Replaces the boilerplate pattern of 20 nearly-identical ``gui_registration.py``
files with a single canonical YAML manifest (``tool_manifest.yaml``) that is
co-located with this module.

The manifest schema mirrors the ``GUI_INFO`` dict used by each
``gui_registration.py`` so that the existing ``_gui_info_to_registration``
and ``auto_discover_guis`` infrastructure works without modification.

Usage::

    from gui_launcher.manifest_loader import load_manifest

    gui_infos = load_manifest()          # uses canonical manifest
    gui_infos = load_manifest(some_path) # uses custom manifest

Each entry in the returned list is a ``GUI_INFO``-compatible dict that can
be passed directly to ``_gui_info_to_registration``.

Closes #1863.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default manifest path — co-located with this module.
_DEFAULT_MANIFEST: Path = Path(__file__).with_name("tool_manifest.yaml")


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    """Load GUI registration data from a YAML manifest file.

    Each entry in the manifest's ``tools`` list is returned as a
    ``GUI_INFO``-compatible dict suitable for passing to
    ``gui_launcher.registry._gui_info_to_registration``.

    Args:
        path: Path to the YAML manifest.  Defaults to the canonical
            ``tool_manifest.yaml`` bundled with this package.

    Returns:
        List of GUI_INFO dicts, one per tool in the manifest.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML content is malformed or missing the
            required ``tools`` key.
    """
    import yaml  # PyYAML is a required dependency (see pyproject.toml)

    resolved = Path(path) if path is not None else _DEFAULT_MANIFEST

    if not resolved.exists():
        raise FileNotFoundError(f"GUI tool manifest not found: {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in tool manifest {resolved}: {exc}") from exc

    if not isinstance(raw, dict) or "tools" not in raw:
        raise ValueError(
            f"Tool manifest {resolved} must be a YAML mapping with a 'tools' key"
        )

    tools: list[dict[str, Any]] = raw["tools"]
    if not isinstance(tools, list):
        raise ValueError(
            f"'tools' in {resolved} must be a YAML sequence, got {type(tools).__name__}"
        )

    logger.debug("Loaded %d tool registrations from %s", len(tools), resolved)
    return tools
