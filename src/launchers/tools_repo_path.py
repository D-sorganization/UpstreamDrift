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


__all__ = ["resolve_explicit_tools_root"]
