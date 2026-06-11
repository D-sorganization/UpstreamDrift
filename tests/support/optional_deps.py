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

import importlib
import importlib.util
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

__all__ = [
    "missing_is_optional",
    "scoped_import_with_optional_mocks",
    "skip_unless_optional",
]

_MISSING = object()


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


@contextmanager
def scoped_import_with_optional_mocks(
    module_name: str,
    module_mocks: dict[str, Any],
    *,
    module_path: Path | None = None,
    purge_modules: Iterable[str] = (),
):
    """Import *module_name* under temporary optional-dependency mocks.

    The helper removes the target module before import, installs the supplied
    dependency mocks, yields the freshly imported module, then restores both the
    dependency entries and any target modules imported under those mocks.
    """
    if not module_name:
        raise ValueError("module_name must be provided")
    if not module_mocks:
        raise ValueError("module_mocks must be provided")

    target_modules = tuple(dict.fromkeys((module_name, *purge_modules)))
    tracked_modules = tuple(dict.fromkeys((*module_mocks.keys(), *target_modules)))
    previous = {name: sys.modules.get(name, _MISSING) for name in tracked_modules}

    try:
        for name in target_modules:
            sys.modules.pop(name, None)
        sys.modules.update(module_mocks)
        importlib.invalidate_caches()
        if module_path is None:
            yield importlib.import_module(module_name)
        else:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load module spec for {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            yield module
    finally:
        for name in target_modules:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        importlib.invalidate_caches()
