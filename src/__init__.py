"""Golf Modeling Suite source package."""

import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from importlib.abc import MetaPathFinder
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


class _VendoredToolsFallbackFinder(MetaPathFinder):
    """Resolve retired child copies from the pinned Tools tree, and only those.

    Mutating ``src.shared.python.__path__`` is not sufficient, and the reason is
    an ordering one: that package's own ``__init__`` imports submodules while it
    executes (``from . import cli_utils``, which imports
    ``src.shared.python.logging_pkg.logging_config``). Those imports run *before*
    any code that could extend the finished module's ``__path__``, so a retired
    copy still raised ``ModuleNotFoundError`` during package initialisation.

    A meta-path finder has no such window: it is consulted on every import,
    including the ones a package issues about itself. This one is **appended** to
    ``sys.meta_path``, so it is asked last -- after the normal machinery has
    failed -- which is what keeps a present child copy authoritative.
    """

    _PREFIXES = ("src.shared.python.", "shared.python.")

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> Any:
        """Return a spec from the pinned tree, or None to defer to everything else."""
        for prefix in self._PREFIXES:
            if not fullname.startswith(prefix):
                continue
            relative = fullname[len(prefix) :].replace(".", "/")
            base = _VENDORED_TOOLS_SRC / "shared" / "python" / relative
            package_init = base / "__init__.py"
            if package_init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname, package_init, submodule_search_locations=[str(base)]
                )
            module_file = base.with_suffix(".py")
            if module_file.is_file():
                return importlib.util.spec_from_file_location(fullname, module_file)
        return None


def _install_vendored_tools_fallback_finder() -> bool:
    """Append the fallback finder so retired child copies resolve upstream."""
    if not _VENDORED_TOOLS_FALLBACK_REGISTERED:
        return False
    if any(
        isinstance(finder, _VendoredToolsFallbackFinder) for finder in sys.meta_path
    ):
        return False
    sys.meta_path.append(_VendoredToolsFallbackFinder())
    return True


_VENDORED_TOOLS_FALLBACK_REGISTERED = _register_vendored_tools_fallback()
_VENDORED_TOOLS_FALLBACK_FINDER_INSTALLED = _install_vendored_tools_fallback_finder()
_PARENT_SHARED_ALIASES_INSTALLED = _install_parent_shared_aliases()
