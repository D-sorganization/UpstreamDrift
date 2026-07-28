"""Canonical validation for an explicit parent Tools checkout."""

from __future__ import annotations

from pathlib import Path

_INVALID_TOOLS_REPO_PATH = (
    "TOOLS_REPO_PATH must point to a Tools checkout containing a src/ "
    "directory, got: {tools_root}"
)


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

    sibling_root = repo_root.parent / "Tools"
    if sibling_root.is_dir():
        return sibling_root / "src"

    # Match the existing last-resort import contract. The missing path is
    # harmless in PYTHONPATH and produces the normal import error downstream.
    return vendor_source


__all__ = ["resolve_explicit_tools_root", "resolve_tools_source_root"]
