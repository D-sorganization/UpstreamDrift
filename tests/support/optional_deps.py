"""Optional-dependency skip discipline for module-level imports (issue #7158).

A blanket ``except ImportError: pytest.skip(..., allow_module_level=True)``
silently skips an entire test module when the imported *source* gains a broken
import — i.e. a real bug presents as a green, skipped suite. This helper skips
**only** when the missing module is a declared optional dependency; any other
``ImportError`` (a genuine bug in the code under test) re-raises and fails
collection loudly.

Usage::

    from tests.support.optional_deps import skip_unless_optional

    try:
        from src.api.server import app
    except ImportError as exc:
        skip_unless_optional(exc, allowed={"fastapi", "httpx"})
        raise  # unreachable when the missing module is optional
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

__all__ = ["skip_unless_optional", "missing_is_optional"]


def missing_is_optional(exc: ImportError, allowed: Iterable[str]) -> bool:
    """Return True if *exc* was raised by a missing *optional* dependency.

    The check matches the top-level package name of the missing module against
    *allowed* (so ``fastapi.testclient`` matches an allowlist entry ``fastapi``).
    """
    name = getattr(exc, "name", None) or ""
    top = name.split(".", 1)[0]
    allowed_set = set(allowed)
    if top and top in allowed_set:
        return True
    # Fall back to scanning the message for an allowed name when ``exc.name`` is
    # unset (older import machinery / re-raised wrappers).
    msg = str(exc)
    return any(mod in msg for mod in allowed_set)


def skip_unless_optional(exc: ImportError, allowed: Iterable[str]) -> None:
    """Skip the module iff *exc* is a missing optional dependency, else return.

    When the missing dependency is in *allowed*, raises ``pytest.skip`` with
    ``allow_module_level=True``. Otherwise returns, so the caller can
    ``raise`` and let the genuine import error fail collection (issue #7158).
    """
    if missing_is_optional(exc, allowed):
        pytest.skip(
            f"optional dependency unavailable: {exc}",
            allow_module_level=True,
        )
    # Not optional → a real bug. Caller re-raises.
