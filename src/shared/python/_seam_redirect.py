"""Resolve retired ``src.shared.python.<root>`` spellings to the pinned Tools tree.

Phase 1 of the seam epic (UD #9406, RM #1505) deletes UpstreamDrift's shadow
copies of Tools-owned packages under ``src/shared/python``. Callers still
spell those packages ``src.shared.python.<root>``, and the Tools
``SharedImportAliasFinder`` deliberately declines that spelling for most roots
when a downstream ``src`` package exists. This module closes the gap with one
rule, the same in tests and at runtime:

* ``src.shared.python.<root>[.<sub>]`` for every root in :data:`REDIRECTED_ROOTS`
  is bound to the *same module object* as ``shared.python.<root>[.<sub>]``,
  which lives in ``vendor/ud-tools/src/shared/python`` (or the installed
  ``ud-tools`` distribution once PR-5 lands).
* A root listed with a UD-only directory (``split`` ruling in
  ``docs/shared_tools/seam_rulings.v1.json``) keeps that directory on the
  canonical package's ``__path__`` so UD-only submodules (``theme.v1``,
  ``theme.layout_metrics`` ...) resolve under both spellings, still as one object.
* If the canonical tree is not importable the import fails loudly with the
  ``git submodule update --init vendor/ud-tools`` hint instead of silently
  falling back to a stale copy.

The vendor roots are put on ``sys.path`` in the same order as
``launch_upstream_drift.py`` and the pytest ``pythonpath`` when ``shared.python``
cannot be found, so a bare ``import src.shared.python.theme`` behaves like the
launcher. Nothing here edits ``vendor/ud-tools``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path

__all__ = [
    "REDIRECTED_ROOTS",
    "SeamRedirectFinder",
    "SeamResolutionError",
    "extend_shared_python_path",
    "install",
    "installed_tools_distribution",
    "load_pinned_tools_package",
    "vendor_search_paths",
]

_LEGACY_PREFIX = "src.shared.python."
_CANONICAL_PREFIX = "shared.python."
_UD_SHARED_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _UD_SHARED_ROOT.parents[2]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "ud-tools"
SUBMODULE_HINT = "git submodule update --init vendor/ud-tools"

# root -> UD-only directory kept on the canonical __path__ (split rulings), or
# None when the UD copy is gone entirely (tools-canonical rulings). Keep this
# in sync with docs/shared_tools/seam_rulings.v1.json ``status: cleaned`` rows.
REDIRECTED_ROOTS: dict[str, Path | None] = {
    "chat_contracts": None,
    "codemap": None,
    "compatibility": None,
    "cors": None,
    "data_processing": None,
    "data_processor_io": None,
    "deprecation": None,
    "file_watcher": None,
    "logging_pkg": None,
    "notes": None,
    "plot_engine": None,
    "plot_theme": None,
    "programmatic_pid": None,
    "rotation_transforms": None,
    "safe_eval": None,
    "safe_pandas_eval": None,
    "scripting": None,
    "theme": _UD_SHARED_ROOT / "theme",
    "ui": _UD_SHARED_ROOT / "ui",
}


class SeamResolutionError(ImportError):
    """The pinned Tools tree is not importable, so a redirected root cannot load."""


def vendor_search_paths(vendor_root: Path = _VENDOR_ROOT) -> tuple[str, ...]:
    """Return the vendor roots in launcher precedence order (parent-source first)."""
    src = vendor_root / "src"
    return (
        str(src / "shared" / "python"),
        str(src),
        str(src / "python" / "src"),
    )


#: Private top-level prefix for pinned-Tools packages loaded by
#: :func:`load_pinned_tools_package`; never a shared namespace.
PINNED_TOOLS_PREFIX = "_pinned_tools__"


def load_pinned_tools_package(relative: str) -> types.ModuleType:
    """Import a pinned-Tools package under a private name (#11994).

    ``relative`` is the package's dotted path under the Tools
    ``shared/python`` root, e.g. ``"sidekick.lab.mocap"``. The package is
    executed as ``_pinned_tools__sidekick__lab__mocap`` so it is never grafted
    onto UpstreamDrift's own ``sidekick`` namespace: appending the pinned tree
    to ``sidekick.lab.__path__`` (the retired ``extend_sidekick_lab_path``)
    built the half-UpstreamDrift, half-Tools package the vendored fallback
    forbids, for the rest of the process. The package must be self-contained
    (relative imports only), which ``sidekick.lab.mocap`` is.

    Postconditions: the same module object is returned on every call, and no
    ``sidekick.*`` ``__path__`` or ``sys.modules`` entry is changed.

    Raises:
        ValueError: when ``relative`` is not a dotted package path.
        SeamResolutionError: when the package is not in the pinned tree.
    """
    parts = relative.split(".")
    if not relative or not all(part.isidentifier() for part in parts):
        raise ValueError(f"expected a dotted package path, got {relative!r}")
    name = PINNED_TOOLS_PREFIX + "__".join(parts)
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    for root in vendor_search_paths():
        init = Path(root).joinpath(*parts, "__init__.py")
        if init.is_file():
            return _exec_private_package(name, init)
    raise SeamResolutionError(
        f"{relative} is not in the pinned Tools tree; run: {SUBMODULE_HINT}"
    )


def _exec_private_package(name: str, init: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise SeamResolutionError(f"cannot load {init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def installed_tools_distribution() -> str | None:
    """Return the installed ``ud-tools`` version, or None when not installed.

    An installed distribution is preferred over ``vendor/ud-tools``: its
    ``shared`` package is a regular package on ``sys.path`` already, so
    :func:`_ensure_canonical_on_path` leaves ``sys.path`` alone.
    """
    try:
        from importlib import metadata

        return metadata.version("ud-tools")
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - metadata backends may raise anything; treat as absent
        return None


def _shared_is_regular_package() -> bool:
    """True when ``shared`` resolves to a real package (vendor or installed)."""
    loaded = sys.modules.get("shared")
    if loaded is not None:
        return bool(getattr(loaded, "__file__", None))
    try:
        spec = importlib.util.find_spec("shared")
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin not in (None, "namespace")


def _ensure_canonical_on_path() -> None:
    """Put the vendor roots first on ``sys.path`` when ``shared`` is not real.

    The repository root also contains a bare ``shared/`` model directory; from a
    plain interpreter it imports as an empty namespace package and hides the
    Tools package. Evict such a namespace so the regular package wins.
    """
    if _shared_is_regular_package():
        return
    if not (_VENDOR_ROOT / "src" / "shared" / "python").is_dir():
        return
    for entry in reversed(vendor_search_paths()):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    importlib.invalidate_caches()
    for name in [n for n in sys.modules if n == "shared" or n.startswith("shared.")]:
        if not getattr(sys.modules[name], "__file__", None):
            del sys.modules[name]


class _BoundLoader(Loader):
    """Loader that hands back an already-imported canonical module."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        del spec
        return self.module

    def exec_module(self, module: types.ModuleType) -> None:
        del module


class SeamRedirectFinder(MetaPathFinder):
    """Bind ``src.shared.python.<root>`` to the canonical ``shared.python.<root>``."""

    def _canonical(self, fullname: str) -> tuple[str, str] | None:
        if not fullname.startswith(_LEGACY_PREFIX):
            return None
        remainder = fullname[len(_LEGACY_PREFIX) :]
        root = remainder.split(".", 1)[0]
        if root not in REDIRECTED_ROOTS:
            return None
        return root, _CANONICAL_PREFIX + remainder

    def _import_canonical(
        self, fullname: str, root: str, canonical_name: str
    ) -> types.ModuleType | None:
        """Import the canonical module; fail loud when the Tools tree is absent.

        The Tools ``SharedImportAliasFinder`` may bind the *root* before this
        finder sees it, so a split root's UD-only directory is attached lazily:
        on the first miss the canonical package ``__path__`` is extended and the
        import retried once.
        """
        for attempt in (0, 1):
            try:
                # The import hook admits only REDIRECTED_ROOTS in _canonical.
                # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
                module = importlib.import_module(canonical_name)
            except ModuleNotFoundError as exc:
                if exc.name in {"shared", "shared.python", _CANONICAL_PREFIX + root}:
                    raise SeamResolutionError(
                        f"{fullname} now resolves to the pinned Tools package "
                        f"{canonical_name}, which is not importable "
                        f"({exc.name}); run `{SUBMODULE_HINT}` or install ud-tools."
                    ) from exc
                if attempt == 1 or REDIRECTED_ROOTS.get(root) is None:
                    return None
                _extend_split_path(
                    # root is selected from the explicit REDIRECTED_ROOTS allowlist.
                    root,
                    # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
                    importlib.import_module(_CANONICAL_PREFIX + root),
                )
                continue
            if "." not in canonical_name[len(_CANONICAL_PREFIX) :]:
                _extend_split_path(root, module)
            return module
        return None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        del path, target
        resolved = self._canonical(fullname)
        if resolved is None:
            return None
        root, canonical_name = resolved
        _ensure_canonical_on_path()
        module = self._import_canonical(fullname, root, canonical_name)
        if module is None:
            return None
        spec = ModuleSpec(
            fullname, _BoundLoader(module), origin=getattr(module, "__file__", None)
        )
        search = getattr(module, "__path__", None)
        if search is not None:
            spec.submodule_search_locations = list(search)
        spec.has_location = spec.origin is not None
        return spec


def _extend_split_path(root: str, module: types.ModuleType) -> None:
    ud_dir = REDIRECTED_ROOTS.get(root)
    search = getattr(module, "__path__", None)
    if ud_dir is None or search is None or not ud_dir.is_dir():
        return
    entry = str(ud_dir)
    if entry not in list(search):
        search.append(entry)


def extend_shared_python_path(path: list[str] | Sequence[str]) -> list[str]:
    """Append the pinned Tools ``shared/python`` directory to a package ``__path__``.

    ``tests/conftest.py`` puts UpstreamDrift's ``src`` ahead of the vendor roots,
    so ``shared.python`` can resolve to *this* package. Retired roots must then
    still be found under ``shared.python.<root>``: they live only in the vendor
    tree, so that directory becomes the trailing search location. UD-owned
    packages keep winning because they are found first.
    """
    entries = list(path)
    vendor_dir = _VENDOR_ROOT / "src" / "shared" / "python"
    if vendor_dir.is_dir():
        entry = str(vendor_dir)
        if entry not in entries:
            entries.append(entry)
    return entries


def install() -> SeamRedirectFinder:
    """Install the finder once per interpreter, ahead of the path finders."""
    for finder in sys.meta_path:
        # Compare by name: this module can be executed twice (as
        # ``src.shared.python._seam_redirect`` and ``shared.python._seam_redirect``).
        if type(finder).__name__ == SeamRedirectFinder.__name__:
            return finder  # type: ignore[return-value]
    finder = SeamRedirectFinder()
    sys.meta_path.insert(0, finder)
    _ensure_canonical_on_path()
    return finder
