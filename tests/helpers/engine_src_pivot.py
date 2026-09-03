"""Reversible ``sys.modules["src"]`` pivot for the 3D-Golf-Model GUI suites.

The C3D viewer and the Simscape ``three_d_gui`` tests import the engine's own
``src.apps.*`` package, whose top-level name (``src``) collides with the repo's
own top-level ``src`` package. Both suites therefore rebind ``sys.modules
["src"]`` to the engine tree while they run.

``sys.modules`` is process-global and, under ``pytest-xdist``, every worker is a
single long-lived process that collects and runs tests from *every* directory in
the suite. Anything the pivot leaves behind is therefore visible to unrelated
tests that happen to run later in the same worker.

Installing the pivot is destructive in two ways, and **both** have to be undone:

1. ``sys.modules`` — every ``src`` / ``src.*`` entry except ``src.shared*`` is
   evicted so the engine package can take the ``src`` name. Restoring only the
   bare ``src`` key (as the previous per-directory implementations did) leaves
   the repo's other ``src.*`` submodules permanently evicted. Later tests then
   re-import them and end up with *duplicate* module objects, which breaks every
   pattern that depends on module identity:
   ``importlib.reload(src.api.utils.path_validation)`` raises ``ImportError:
   module ... not in sys.modules``; ``monkeypatch.setattr("src.launchers.x.y")``
   and ``patch("src.launchers.docker_manager.secure_run")`` patch a fresh copy
   that the test's already-bound callables never consult; and ``isinstance``
   checks against a re-imported class fail.
2. ``sys.path`` — the repo root and ``<repo>/src`` are added so the engine's
   bare ``shared.python.*`` imports resolve.

:class:`EngineSrcPivot` snapshots the whole ``src`` namespace slice of
``sys.modules`` plus the ``sys.path`` entries it adds on entry, and puts both
back on exit -- the same discipline as ``patch.dict("sys.modules", ...)``, scoped
to the keys this pivot actually touches instead of the whole module table.

Each conftest owns its own instance, so the reentrancy counters of the two
directories can never interfere.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["EngineSrcPivot"]

# ``<repo>/src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src``
_ENGINE_SRC_PARTS = (
    "src",
    "engines",
    "Simscape_Multibody_Models",
    "3D_Golf_Model",
    "python",
    "src",
)

# ``src.shared.*`` is supplied by the repo tree in both worlds, so it survives
# the rebind: dropping it would make ``from src import shared`` fail, because
# the engine's ``src/__init__.py`` has no ``shared`` attribute.
_KEEP_PREFIX = "src.shared"

# Always pre-cached before the rebind, in this order, so the keep-set above is
# concretely loaded rather than merely importable.
_BASE_PRECACHE = ("src", "src.shared", "src.shared.python")


def _src_namespace() -> list[str]:
    """Return the ``sys.modules`` keys owned by the top-level ``src`` package."""
    return [name for name in sys.modules if name == "src" or name.startswith("src.")]


def _relink_to_parent(name: str, module: ModuleType) -> None:
    """Point ``parent.child`` back at the restored ``child`` module.

    ``import a.b`` binds ``b`` as an attribute of ``a`` as well as writing
    ``sys.modules["a.b"]``, and ``from a import b`` reads that *attribute*.
    A re-import inside the pivot window can therefore leave a restored parent
    holding a throwaway child, so ``from a import b`` and ``import a.b`` hand
    out different objects -- the same duplicate-module bug one level down.

    Only a stale module of the same dotted name is replaced: many packages
    re-export a *function* under the name of one of their own submodules
    (``forward_kinematics`` is both), and clobbering that with the module
    would break ``from pkg import forward_kinematics``.
    """
    parent_name, _, child = name.rpartition(".")
    if not parent_name:
        return
    parent = sys.modules.get(parent_name)
    if parent is None:
        return
    current = getattr(parent, child, None)
    if current is module:
        return
    if current is None or (
        isinstance(current, ModuleType) and getattr(current, "__name__", None) == name
    ):
        setattr(parent, child, module)


class EngineSrcPivot:
    """Bind top-level ``src`` to the engine package for a bounded window.

    Preconditions: ``repo_root`` is the repository root; ``precache`` names
    modules that must stay resolvable across the rebind.

    Postconditions: between :meth:`enter` and the matching :meth:`exit`,
    ``sys.modules["src"]`` is the engine package. After the outermost
    :meth:`exit`, ``sys.modules``' ``src`` namespace and ``sys.path`` are
    byte-for-byte what they were before the outermost :meth:`enter`.
    """

    def __init__(self, repo_root: Path, *, precache: tuple[str, ...] = ()) -> None:
        if not isinstance(repo_root, Path):
            raise TypeError(f"repo_root must be a Path, got {type(repo_root)!r}")
        self._repo_root = repo_root
        self._engine_src = repo_root.joinpath(*_ENGINE_SRC_PARTS)
        self._precache = (*_BASE_PRECACHE, *precache)
        self._depth = 0
        self._saved_modules: dict[str, ModuleType] = {}
        self._added_paths: list[str] = []

    # -- lifecycle ----------------------------------------------------------

    def enter(self) -> None:
        """Install the pivot, or bump the reentrancy depth if already active."""
        if self._depth == 0:
            self._saved_modules = {name: sys.modules[name] for name in _src_namespace()}
            self._added_paths = []
            self._install()
        self._depth += 1

    def exit(self) -> None:
        """Undo the outermost :meth:`enter`, restoring ``sys.modules``/``sys.path``."""
        self._depth -= 1
        if self._depth > 0:
            return
        self._depth = 0

        # Evict everything the pivot added under ``src.`` (the engine's own
        # ``src.apps.*``, ``src.c3d_reader``, ...), then put back every repo
        # module the install evicted.
        for name in _src_namespace():
            if name not in self._saved_modules:
                del sys.modules[name]
        sys.modules.update(self._saved_modules)
        for name, module in self._saved_modules.items():
            _relink_to_parent(name, module)
        self._saved_modules = {}

        for entry in reversed(self._added_paths):
            with contextlib.suppress(ValueError):
                sys.path.remove(entry)
        self._added_paths = []

    # -- installation -------------------------------------------------------

    def _install(self) -> None:
        if not self._engine_src.is_dir():
            return

        self._add_path(str(self._repo_root), front=True)
        for qual in self._precache:
            with contextlib.suppress(ImportError):
                sys.modules[qual] = importlib.import_module(qual)

        for name in _src_namespace():
            if name == _KEEP_PREFIX or name.startswith(_KEEP_PREFIX + "."):
                continue
            del sys.modules[name]

        # Bind ``src`` straight to the engine package rather than fighting
        # pytest's own sys.path ordering.
        spec = importlib.util.spec_from_file_location(
            "src",
            str(self._engine_src / "__init__.py"),
            submodule_search_locations=[str(self._engine_src)],
        )
        if spec is None or spec.loader is None:
            return
        src_mod = importlib.util.module_from_spec(spec)
        sys.modules["src"] = src_mod
        spec.loader.exec_module(src_mod)

        # Multi-rooted package: the engine path first so ``src.apps`` wins,
        # the repo path second so ``src.shared``/``src.tools`` still resolve.
        repo_src = self._repo_root / "src"
        src_mod.__path__ = [str(self._engine_src), str(repo_src)]

        # ``from src import shared`` walks ``src.__dict__`` before consulting
        # ``sys.modules``, so preserved children must be re-attached by hand.
        for name, module in list(sys.modules.items()):
            parts = name.split(".")
            if len(parts) == 2 and parts[0] == "src":
                setattr(src_mod, parts[1], module)

        # For the engine's bare ``shared.python.*`` imports.
        self._add_path(str(repo_src))

    def _add_path(self, entry: str, *, front: bool = False) -> None:
        if entry in sys.path:
            return
        if front:
            sys.path.insert(0, entry)
        else:
            sys.path.append(entry)
        self._added_paths.append(entry)
