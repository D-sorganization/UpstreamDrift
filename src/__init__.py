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
    """Report whether the pinned Tools tree is present to fall back to.

    Deliberately does NOT add the tree to ``sys.path``. Doing so exposes every
    top-level Tools package -- ``sidekick``, ``chat``, ``contracts`` -- as
    importable, which silently changed availability probes elsewhere:
    ``sidekick.lab.mocap`` began resolving and ``probe_tools_schema()`` flipped
    from "unavailable" to "ready" in a repository that had never declared that
    dependency reachable. The finder below answers the shared namespace on its
    own, so the fallback stays scoped to what it is meant to serve.

    Tools can live in either of two places, and a retired cluster must resolve
    from whichever is present: the pinned tree at ``vendor/ud-tools``, or an
    installed Tools distribution providing a top-level ``shared.python``.
    Tools' own downstream-consumer contracts install the distribution into a
    checkout that has no submodule at all, so gating on the vendored tree alone
    left retired clusters unresolvable there (Tools#5048).

    Returns:
        True when either Tools tree is reachable, so the finder has something
        to serve. False when neither is, where there is nothing to fall back to.
    """
    if (_VENDORED_TOOLS_SRC / "shared" / "python").is_dir():
        return True
    try:
        return importlib.util.find_spec("shared.python") is not None
    except (ImportError, ValueError):
        return False


_UD_SHARED_PYTHON = Path(__file__).resolve().parent / "shared" / "python"


def _cluster_is_still_owned(tail: str) -> bool:
    """Report whether UpstreamDrift still owns the top-level cluster in *tail*.

    The fallback exists to serve clusters that have been **wholly retired**, so
    a deleted child copy resolves upstream instead of raising. It must not fill
    a gap *inside* a cluster UpstreamDrift still owns: doing so silently builds
    a hybrid package, half UpstreamDrift and half Tools.

    That is not hypothetical. ``sidekick`` is UpstreamDrift-owned and has no
    ``lab/mocap``; the pinned tree does. Without this guard the finder served
    ``sidekick.lab.mocap`` from the pinned tree, which flipped
    ``probe_tools_schema()`` from ``unavailable`` to ``ready`` and broke two
    ``motion_capture/rig`` tests that assert the module is absent. The absence
    of a submodule inside an owned package is meaningful, not a gap to patch.
    """
    cluster = tail.split(".", 1)[0]
    if not cluster:
        return False
    return (_UD_SHARED_PYTHON / cluster).is_dir() or (
        _UD_SHARED_PYTHON / f"{cluster}.py"
    ).is_file()


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
            tail = fullname[len(prefix) :]
            if _cluster_is_still_owned(tail):
                return None
            relative = tail.replace(".", "/")
            base = _VENDORED_TOOLS_SRC / "shared" / "python" / relative
            package_init = base / "__init__.py"
            if package_init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname, package_init, submodule_search_locations=[str(base)]
                )
            module_file = base.with_suffix(".py")
            if module_file.is_file():
                return importlib.util.spec_from_file_location(fullname, module_file)
            return self._installed_tools_spec(fullname, tail)
        return None

    def _installed_tools_spec(self, fullname: str, tail: str) -> Any:
        """Resolve a retired cluster from an installed Tools distribution.

        The pinned tree is one of two places Tools can live. A consumer that
        installs Tools as a distribution -- Tools' own downstream-consumer
        contracts do exactly that -- has no ``vendor/ud-tools`` checkout at
        all, so a retired cluster has nothing to fall back to there.

        Until Tools#5048 that gap was hidden: ``SharedImportAliasFinder``
        rewrote every ``src.shared.python.<root>`` to ``shared.python.<root>``,
        which happened to cover retired clusters too. Correcting that predicate
        stops the blanket rewrite -- rightly, since it also captured clusters
        UpstreamDrift owns -- and leaves this finder to serve the retired ones
        from whichever tree is present.

        ``_cluster_is_still_owned`` has already run, so this cannot capture a
        cluster UpstreamDrift still owns.
        """
        canonical = f"shared.python.{tail}"
        if canonical in sys.modules:
            return getattr(sys.modules[canonical], "__spec__", None)
        removed = self in sys.meta_path
        if removed:
            sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(canonical)
        except (ImportError, ValueError):
            return None
        finally:
            if removed:
                sys.meta_path.append(self)
        if spec is None or spec.origin is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            spec.origin,
            submodule_search_locations=(
                list(spec.submodule_search_locations)
                if spec.submodule_search_locations is not None
                else None
            ),
        )


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
