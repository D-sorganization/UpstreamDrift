"""Canonical resolution and validation of the Tools repository location.

One narrow facade (:func:`resolve_tools_repo`) owns the precedence order for
locating the Tools checkout so callers never re-probe the filesystem
themselves (Law of Demeter, issue #8858):

1. Explicit ``TOOLS_REPO_PATH`` environment override (validated, unpinned).
2. Vendored ``vendor/ud-tools`` gitlink when present **and** matching the
   tracked pin (the only pinned source).
3. Sibling-walk discovery as a dev-mode fallback that logs a clear warning
   naming the resolved path and that it is unpinned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.config.tools_vendor_authority import (
    inspect_tools_vendor_authority,
)

logger = logging.getLogger(__name__)

_INVALID_TOOLS_REPO_PATH = (
    "TOOLS_REPO_PATH must point to a Tools checkout containing a src/ "
    "directory, got: {tools_root}"
)

_SOURCE_ENV = "env"
_SOURCE_VENDOR = "vendor"
_SOURCE_SIBLING = "sibling"


@dataclass(frozen=True)
class ToolsRepoResolution:
    """Where the Tools repository was found and how trustworthy it is.

    Attributes:
        path: Absolute Tools repository root (contains ``src/``).
        source: One of ``"env"``, ``"vendor"``, or ``"sibling"``.
        pinned: True only for the vendored gitlink validated against the
            tracked superproject pin.
    """

    path: Path
    source: str
    pinned: bool


def resolve_explicit_tools_root(env_value: str | None) -> Path | None:
    """Resolve an explicitly configured Tools root or return ``None``.

    Preconditions:
        ``env_value`` is a string or ``None``.

    Postconditions:
        A returned path is absolute and contains a ``src/`` directory.
        Invalid explicit paths fail closed with the canonical contract error.
    """
    if env_value is None or env_value == "":
        return None
    if not isinstance(env_value, str):
        raise TypeError("TOOLS_REPO_PATH must be a string or None")

    tools_root = Path(env_value).expanduser().resolve()
    tools_src = tools_root / "src"
    if not tools_root.is_dir() or not tools_src.is_dir():
        raise RuntimeError(_INVALID_TOOLS_REPO_PATH.format(tools_root=tools_root))
    return tools_root


def _find_sibling_tools_root(repo_root: Path) -> Path | None:
    """Walk upward for a workspace-level ``Tools`` sibling checkout.

    Normal clones sit directly beside Tools. Agent worktrees commonly sit
    below ``UpstreamDrift/.codex-worktrees/<branch>``; walking upward lets
    those source checkouts find the same workspace-level sibling.
    """
    for workspace_root in repo_root.parents:
        sibling_root = workspace_root / "Tools"
        if (sibling_root / "src").is_dir():
            return sibling_root
    return None


def resolve_tools_repo(
    repo_root: Path, env_value: str | None
) -> ToolsRepoResolution | None:
    """Resolve the Tools repository root through the canonical precedence.

    Preconditions:
        ``repo_root`` is a ``pathlib.Path`` (the UpstreamDrift checkout root);
        ``env_value`` is the raw ``TOOLS_REPO_PATH`` value or ``None``.

    Postconditions:
        A returned resolution has ``path / "src"`` as an existing directory,
        ``source`` in {"env", "vendor", "sibling"}, and ``pinned`` true only
        when the vendored gitlink validated against the tracked pin. Returns
        ``None`` when no source is available. Callers must not re-probe the
        filesystem for Tools themselves.

    Raises:
        TypeError: If ``repo_root`` is not a ``pathlib.Path``.
        RuntimeError: If an explicit ``TOOLS_REPO_PATH`` is set but invalid
            (fail closed rather than silently falling back).
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")

    explicit_root = resolve_explicit_tools_root(env_value)
    if explicit_root is not None:
        return ToolsRepoResolution(path=explicit_root, source=_SOURCE_ENV, pinned=False)

    vendor_root = repo_root / "vendor" / "ud-tools"
    if (vendor_root / "src").is_dir():
        authority = inspect_tools_vendor_authority(repo_root)
        if authority.available:
            return ToolsRepoResolution(
                path=authority.root, source=_SOURCE_VENDOR, pinned=True
            )
        logger.warning(
            "Vendored Tools at %s is unavailable (%s); falling back to "
            "sibling discovery",
            vendor_root,
            authority.reason,
        )

    sibling_root = _find_sibling_tools_root(repo_root)
    if sibling_root is not None:
        logger.warning(
            "Tools resolved via UNPINNED dev-mode sibling checkout at %s; "
            "production installs should use the vendored vendor/ud-tools "
            "gitlink or set TOOLS_REPO_PATH explicitly",
            sibling_root,
        )
        return ToolsRepoResolution(
            path=sibling_root, source=_SOURCE_SIBLING, pinned=False
        )
    return None


def resolve_tools_source_root(repo_root: Path, env_value: str | None) -> Path:
    """Select the Tools ``src`` authority used by launcher processes.

    An explicit checkout is validated before any fallback is considered.
    Otherwise the pinned vendor wins over a mutable sibling checkout.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")

    explicit_root = resolve_explicit_tools_root(env_value)
    if explicit_root is not None:
        return explicit_root / "src"

    vendor_source = repo_root / "vendor" / "ud-tools" / "src"
    if vendor_source.is_dir():
        return vendor_source

    sibling_root = _find_sibling_tools_root(repo_root)
    if sibling_root is not None:
        return sibling_root / "src"

    # Match the existing last-resort import contract. The missing path is
    # harmless in PYTHONPATH and produces the normal import error downstream.
    return vendor_source


__all__ = [
    "ToolsRepoResolution",
    "resolve_explicit_tools_root",
    "resolve_tools_repo",
    "resolve_tools_source_root",
]
