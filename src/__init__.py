"""Golf Modeling Suite source package."""

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

# The pinned Tools tree, as a *fallback* import location for the shared
# namespace (UpstreamDrift#9406).
#
# Every `tools-canonical` ruling in docs/shared_tools/seam_rulings.v1.json is
# "delete UpstreamDrift's copy and let the pinned Tools tree answer", and all 36
# actionable rulings sit at `pending-cleanup` because nothing put that tree on
# the import path at runtime: deleting a child copy simply produced
# ModuleNotFoundError. This is the mechanism those rulings were waiting for.
#
# It is APPENDED, never prepended. While a child copy exists it is still found
# first, so this changes no import that resolves today -- it only answers the
# ones that would otherwise fail. Prepending would silently flip resolution for
# the 292 files that still diverge, which is exactly the ambiguity #9406 exists
# to remove.
_VENDORED_TOOLS_SRC = (
    Path(__file__).resolve().parent.parent / "vendor" / "ud-tools" / "src"
)

_CANONICAL_ALIAS_MODULES = frozenset(
    {
        "shared",
        "shared.python",
        "shared.python.import_aliases",
    }
)


def _load_downstream_shared_namespaces() -> None:
    """Attach the real downstream parents before Tools aliases add children."""
    importlib.import_module("src.shared.python")


def _restore_import_state(
    previous_modules: Mapping[str, ModuleType],
    previous_meta_path: Sequence[Any],
) -> None:
    """Restore the interpreter state captured before an alias attempt."""
    for name in tuple(sys.modules):
        if name not in previous_modules:
            sys.modules.pop(name, None)
    sys.modules.update(previous_modules)
    sys.meta_path[:] = previous_meta_path


def _install_parent_shared_aliases() -> bool:
    """Atomically install Tools-owned aliases when their module is available."""
    previous_modules = dict(sys.modules)
    previous_meta_path = list(sys.meta_path)
    try:
        from shared.python.import_aliases import install_shared_import_aliases
    except ModuleNotFoundError as exc:
        _restore_import_state(previous_modules, previous_meta_path)
        if exc.name not in _CANONICAL_ALIAS_MODULES:
            raise
        return False
    except Exception:
        _restore_import_state(previous_modules, previous_meta_path)
        raise

    try:
        _load_downstream_shared_namespaces()
        install_shared_import_aliases()
    except Exception:
        _restore_import_state(previous_modules, previous_meta_path)
        raise
    return True


def _register_vendored_tools_fallback() -> bool:
    """Append the pinned Tools tree so a retired child copy resolves upstream.

    Returns:
        True when the vendored tree was found and is on ``sys.path``.

    Postcondition:
        Appends at most one entry and never reorders ``sys.path``, so a module
        that resolves before this call resolves identically after it.
    """
    if not (_VENDORED_TOOLS_SRC / "shared" / "python").is_dir():
        # Absent in a wheel install: build_hooks.py copies the pinned tree into
        # the package itself, so there is nothing to fall back to.
        return False
    location = str(_VENDORED_TOOLS_SRC)
    if location not in sys.path:
        sys.path.append(location)
    return True


def _extend_shared_namespace_path() -> bool:
    """Let ``src.shared.python`` resolve retired child copies from the pinned tree.

    Appending to ``sys.path`` is enough for the canonical ``shared.python.X``
    spelling, but ``src.shared.python`` is a real package whose ``__path__``
    lists only UpstreamDrift's own directory -- so a deleted child copy still
    raised ``ModuleNotFoundError`` under that spelling, which is the one the
    repository actually imports.

    Returns:
        True when the pinned location was added to the package's search path.

    Postcondition:
        The vendored location is appended, so UpstreamDrift's own directory is
        still searched first and no currently-resolving import changes.
    """
    if not _VENDORED_TOOLS_FALLBACK_REGISTERED:
        return False
    module = sys.modules.get("src.shared.python")
    search_path = getattr(module, "__path__", None)
    if search_path is None:
        return False
    location = str(_VENDORED_TOOLS_SRC / "shared" / "python")
    if location in list(search_path):
        return False
    try:
        search_path.append(location)
    except AttributeError:
        module.__path__ = [*search_path, location]  # type: ignore[union-attr]
    return True


_VENDORED_TOOLS_FALLBACK_REGISTERED = _register_vendored_tools_fallback()
_PARENT_SHARED_ALIASES_INSTALLED = _install_parent_shared_aliases()
_SHARED_NAMESPACE_PATH_EXTENDED = _extend_shared_namespace_path()
