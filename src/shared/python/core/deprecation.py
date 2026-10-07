"""Deprecated-name shims for module-level ``__getattr__`` (PEP 562)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping


def deprecated_alias_getattr(
    module_name: str, aliases: Mapping[str, type]
) -> Callable[[str], object]:
    """Build a module ``__getattr__`` that resolves deprecated names.

    Preconditions: ``aliases`` maps each deprecated name to the class that
    replaces it.
    Postconditions: reading a deprecated name emits a ``DeprecationWarning``
    naming the replacement and returns that class; any other name raises
    ``AttributeError`` as an ordinary missing module attribute would.
    """

    def __getattr__(name: str) -> object:
        replacement = aliases.get(name)
        if replacement is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        warnings.warn(
            f"{name} is deprecated and will be removed in a future release; "
            f"use {replacement.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement

    return __getattr__
