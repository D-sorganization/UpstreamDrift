"""Guard against a directory-scoped conftest leaking its ``src`` pivot.

Regression coverage for issue #8834 / PR #9374 and its two follow-ups.

``tests/unit/c3d_viewer/ui/conftest.py`` and
``tests/unit/engines/simscape/three_d_gui/conftest.py`` both rebind the
top-level ``sys.modules["src"]`` entry to
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src``'s own
(unrelated) ``src`` package, so that the C3D-viewer engine's ``src.apps.*``
absolute imports resolve inside its own tests.

Originally that rebind happened once, unconditionally, at module import time
and was never undone. ``sys.modules`` is process-global, and under
``pytest-xdist`` a single worker process collects and runs tests from every
directory in the suite -- so whichever of these two directories got collected
first in a given worker permanently poisoned ``sys.modules["src"]`` for every
test that ran afterwards in that worker, including
``tests/scripts/test_validate_suite.py``, whose
``launch_upstream_drift._retry_parent_shared_alias_installer()`` does a bare
``import src`` and expects the real
``src/__init__.py::_install_parent_shared_aliases``.

PR #9404 scoped the rebind to the conftest's own directory but restored only
the bare ``sys.modules["src"]`` key. Installing the pivot *also* evicts every
other ``src.*`` entry (everything outside ``src.shared*``), so unrelated
suites were left re-importing the repo's modules and holding **duplicate**
module objects. That broke every pattern that depends on module identity --
``importlib.reload(src.api.utils.path_validation)`` raised ``ImportError:
module ... not in sys.modules``, ``monkeypatch.setattr("src.launchers.x.y")``
and ``patch("src.launchers.docker_manager.secure_run")`` patched a copy the
test's already-bound callables never consulted, and ``isinstance`` checks
against re-imported classes failed.

These tests therefore pin both halves of the invariant: the pivot is exposed
as a paired, reentrant enter/exit (never a bare module-level call), and the
outermost exit restores the **whole** ``src`` namespace and ``sys.path``, not
just the top-level ``src`` key.

Note: reproducing the originally-reported failure end-to-end requires the
full ``testpaths`` collection sweep, so that confirmation lives in the
investigation notes rather than as a subprocess test here, to keep this suite
fast and non-flaky.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PIVOT_CONFTESTS = (
    _REPO_ROOT / "tests" / "unit" / "c3d_viewer" / "ui" / "conftest.py",
    _REPO_ROOT
    / "tests"
    / "unit"
    / "engines"
    / "simscape"
    / "three_d_gui"
    / "conftest.py",
)

# A repo ``src.*`` module outside the pivot's ``src.shared`` keep-set, i.e. one
# the pivot install genuinely evicts. This is the exact module whose eviction
# broke ``tests/unit/utils/test_path_validation.py``.
_EVICTED_PROBE_MODULE = "src.api.utils.path_validation"

# Its parent package. ``src/api/utils/__init__.py`` deliberately does NOT
# import ``path_validation``, which is what makes the pair a faithful probe
# for the PR #9446 asymmetry: a *fresh* import of the parent leaves the child
# unbound as an attribute whenever the child's ``sys.modules`` entry already
# exists (the import system short-circuits and never rebinds it).
_EVICTED_PROBE_PARENT = "src.api.utils"


def _load_conftest(path: Path) -> ModuleType:
    """Load a directory-scoped conftest.py under a private module name.

    A distinct name per call avoids colliding with pytest's own conftest
    bookkeeping in ``sys.modules`` while still executing the file's real
    top-level code (needed to construct its ``_PIVOT``).
    """
    spec = importlib.util.spec_from_file_location(
        f"_repo_hygiene_probe_{path.parent.name}_conftest", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _src_namespace() -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }


def _restore(saved: dict[str, ModuleType], saved_path: list[str]) -> None:
    """Belt-and-suspenders cleanup so a failed assertion cannot leak."""
    for name in list(_src_namespace()):
        if name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)
    sys.path[:] = saved_path


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_conftest_defines_scoped_enter_exit_hooks(conftest_path: Path) -> None:
    """DbC: the pivot must expose a paired, reentrant enter/exit object.

    A bare module-level pivot call with no counterpart is exactly the shape of
    the original bug -- this fails loudly if either directory's conftest
    regresses back to that shape.
    """
    assert conftest_path.is_file(), f"Expected conftest at {conftest_path}"
    module = _load_conftest(conftest_path)

    pivot = getattr(module, "_PIVOT", None)
    assert pivot is not None, (
        f"{conftest_path} must build a shared EngineSrcPivot as _PIVOT so the "
        "src rebind is scoped rather than applied at import time"
    )
    assert callable(getattr(pivot, "enter", None)), (
        f"{conftest_path}'s _PIVOT must expose enter() to scope the src rebind"
    )
    assert callable(getattr(pivot, "exit", None)), (
        f"{conftest_path}'s _PIVOT must expose exit() to undo the src rebind"
    )
    for hook_name in (
        "pytest_make_collect_report",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
    ):
        assert hasattr(module, hook_name), (
            f"{conftest_path} must implement {hook_name} so the rebind is "
            "directory-scoped by pytest's own hook-consultation rules "
            "rather than leaking into unrelated tests in the same worker"
        )


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_enter_exit_restores_prior_src_identity(conftest_path: Path) -> None:
    """The rebind must be fully reversible, including when nested.

    Drives enter/exit directly (nested twice, matching how collection of a
    Package -> Module -> Function chain re-enters the same pivot) and asserts
    the top-level ``src`` identity is bit-for-bit restored after the outermost
    exit.
    """
    pivot = _load_conftest(conftest_path)._PIVOT

    saved = _src_namespace()
    saved_path = list(sys.path)
    previous = sys.modules.get("src")
    try:
        pivot.enter()
        pivoted = sys.modules.get("src")
        assert pivoted is not None, "pivot must install a src module"
        assert pivoted is not previous, (
            "pivot did not actually rebind src to the engine package"
        )

        # Nested entry (mirrors nested collectors reusing the same pivot).
        pivot.enter()
        assert sys.modules.get("src") is pivoted
        pivot.exit()
        assert sys.modules.get("src") is pivoted, (
            "an inner exit must not undo the pivot while an outer scope is still active"
        )

        pivot.exit()
        assert sys.modules.get("src") is previous, (
            "the outermost exit must restore the exact prior src identity "
            "-- this is the invariant that keeps the pivot from leaking "
            "into unrelated tests that run later in the same xdist worker"
        )
    finally:
        _restore(saved, saved_path)


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_exit_restores_the_whole_src_namespace(conftest_path: Path) -> None:
    """Exit must restore every evicted ``src.*`` entry, not just ``src``.

    Restoring only the top-level key leaves unrelated suites re-importing the
    repo's ``src.*`` modules and holding duplicate module objects, which is
    what broke ``importlib.reload`` / ``monkeypatch.setattr`` / ``isinstance``
    across ``tests/unit/launchers``, ``tests/unit/launcher`` and
    ``tests/unit/utils`` once #9404 landed.
    """
    pivot = _load_conftest(conftest_path)._PIVOT

    probe = importlib.import_module(_EVICTED_PROBE_MODULE)
    saved = _src_namespace()
    saved_path = list(sys.path)
    assert _EVICTED_PROBE_MODULE in saved, (
        f"{_EVICTED_PROBE_MODULE} must be resident before the pivot for this "
        "test to prove anything"
    )
    try:
        pivot.enter()
        assert _EVICTED_PROBE_MODULE not in sys.modules, (
            "installing the pivot is expected to evict the repo's src.* "
            "submodules -- if it no longer does, this guard needs rewriting"
        )
        pivot.exit()

        assert sys.modules.get(_EVICTED_PROBE_MODULE) is probe, (
            f"exit left {_EVICTED_PROBE_MODULE} evicted; later tests would "
            "re-import it and hold a duplicate module object"
        )
        assert _src_namespace() == saved, (
            "exit must restore the src namespace exactly -- no leftover engine "
            "modules, no evicted repo modules"
        )
        assert sys.path == saved_path, (
            "exit must also undo the sys.path entries the pivot added"
        )
    finally:
        _restore(saved, saved_path)


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_enter_rolls_back_when_install_raises(
    conftest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``enter()`` must not leave the engine ``src`` installed.

    Regression coverage for issue #9387.

    Every call site has the shape ``pivot.enter()`` followed by ``try: ...
    finally: pivot.exit()`` -- pytest hookwrappers cannot do better, because
    the undo has to be armed around the ``yield``, not around ``enter()``
    itself. So the ``finally`` only exists once ``enter()`` has *returned*,
    and an ``enter()`` that raises skips its own undo entirely.

    ``_install`` is destructive before it can fail: it evicts the repo's whole
    ``src`` namespace and binds ``sys.modules["src"]`` to the engine package
    *before* executing the engine's ``__init__.py``. One engine-side import
    error (an optional dependency missing on a CI image, say) therefore used
    to shadow the repo's ``src`` for the entire remaining life of that
    long-lived xdist worker, which is what produced both #9387 symptoms:
    ``module 'src' has no attribute '_install_parent_shared_aliases'`` in
    ``tests/scripts/test_validate_suite.py``, and ``mock.patch`` silently
    binding to duplicate ``src.launchers.*`` modules in
    ``tests/launchers/test_shot_tracer_import_gui.py``.
    """
    pivot = _load_conftest(conftest_path)._PIVOT

    probe = importlib.import_module(_EVICTED_PROBE_MODULE)
    saved = _src_namespace()
    saved_path = list(sys.path)
    previous = sys.modules.get("src")
    boom = RuntimeError("engine src/__init__.py blew up mid-install")
    real_spec_from_file_location = importlib.util.spec_from_file_location

    class _ExplodingLoader:
        """Stands in for the engine package's loader and fails in exec."""

        def create_module(self, spec: object) -> None:
            return None

        def exec_module(self, module: ModuleType) -> None:
            # The faithful failure point: _install has already evicted the repo
            # namespace *and* bound sys.modules["src"] to the engine package by
            # the time the engine's __init__ body runs.
            raise boom

    def _exploding_spec(*args: object, **kwargs: object) -> object:
        spec = real_spec_from_file_location(*args, **kwargs)
        spec.loader = _ExplodingLoader()
        return spec

    try:
        monkeypatch.setattr(importlib.util, "spec_from_file_location", _exploding_spec)
        with pytest.raises(RuntimeError) as excinfo:
            pivot.enter()
        assert excinfo.value is boom, (
            "enter() must re-raise the original failure, not mask it behind "
            "its own rollback"
        )
        monkeypatch.undo()

        assert sys.modules.get("src") is previous, (
            "a failed enter() left the engine src bound to sys.modules['src']; "
            "no caller's finally: exit() can undo it, so every later test in "
            "this xdist worker sees the wrong src package"
        )
        assert sys.modules.get(_EVICTED_PROBE_MODULE) is probe, (
            "a failed enter() left the repo's src.* submodules evicted; later "
            "tests re-import them and hold duplicate module objects"
        )
        assert _src_namespace() == saved, (
            "a failed enter() must restore the src namespace exactly"
        )
        assert sys.path == saved_path, (
            "a failed enter() must also undo the sys.path entries it added"
        )

        # The pivot must still be usable afterwards: the rollback resets the
        # reentrancy depth rather than wedging it at a half-entered value.
        pivot.enter()
        assert sys.modules.get("src") is not previous
        pivot.exit()
        assert sys.modules.get("src") is previous
    finally:
        _restore(saved, saved_path)


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_exit_without_enter_is_a_noop(conftest_path: Path) -> None:
    """An unbalanced ``exit()`` must not wipe the ``src`` namespace.

    Regression coverage for issue #9387.

    ``exit()`` restores from ``self._saved_modules``, and its first step is to
    delete every ``src``/``src.*`` key that the snapshot does not contain.
    With no pivot active that snapshot is empty, so an exit that never had a
    matching enter would delete the entire ``src`` namespace and put nothing
    back -- the same duplicate-module corruption the pivot exists to prevent,
    reached from the opposite direction. Hook impls can end up unpaired
    whenever an earlier hookwrapper in the chain raises before its own
    ``yield``, so this has to be safe by construction rather than by call-site
    discipline.
    """
    pivot = _load_conftest(conftest_path)._PIVOT

    probe = importlib.import_module(_EVICTED_PROBE_MODULE)
    saved = _src_namespace()
    saved_path = list(sys.path)
    try:
        pivot.exit()

        assert sys.modules.get(_EVICTED_PROBE_MODULE) is probe
        assert _src_namespace() == saved, (
            "exit() with no matching enter() must leave sys.modules untouched"
        )
        assert sys.path == saved_path

        # The stray exit must not have driven the depth counter negative, or
        # the next genuine enter/exit pair would no longer restore anything.
        previous = sys.modules.get("src")
        pivot.enter()
        assert sys.modules.get("src") is not previous
        pivot.exit()
        assert sys.modules.get("src") is previous
        assert _src_namespace() == saved
    finally:
        _restore(saved, saved_path)


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_exit_relinks_child_when_parent_key_was_popped(
    conftest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child-without-parent snapshot must not plant a delayed AttributeError.

    Regression coverage for the restore asymmetry documented in PR #9446.

    Genesis: a third-party cleanup (e.g. an autouse fixture that snapshots and
    pops ``sys.modules`` keys by prefix) removes a parent package's key while
    the child's key stays cached. The pivot then snapshots exactly that
    child-without-parent state. Before the fix, exit() restored the child
    orphaned-by-key; the next ``import src.api.utils`` anywhere executed a
    *fresh*, childless parent module, and the import system never rebinds a
    cached child onto it (the child's ``sys.modules`` entry short-circuits
    ``_find_and_load``) -- so the dotted-string form

        monkeypatch.setattr("src.api.utils.path_validation.<attr>", ...)

    raised ``AttributeError: module 'src.api.utils' has no attribute
    'path_validation'`` (the shape of the CI failure PR #9446 worked around by
    switching tests/unit/engines/myosuite/test_canonical_adapter.py to the
    module-object form), while the module-object form kept working. The fix
    makes exit() re-import the missing parent and re-link the restored child,
    mirroring the ``sys.modules`` layout the snapshot captured.
    """
    pivot = _load_conftest(conftest_path)._PIVOT

    child = importlib.import_module(_EVICTED_PROBE_MODULE)
    parent = importlib.import_module(_EVICTED_PROBE_PARENT)
    saved = _src_namespace()
    saved_path = list(sys.path)
    grandparent_name, _, parent_attr = _EVICTED_PROBE_PARENT.rpartition(".")
    try:
        # A third-party cleanup pops the parent's key; the child's key stays.
        del sys.modules[_EVICTED_PROBE_PARENT]

        pivot.enter()
        pivot.exit()

        restored_parent = sys.modules.get(_EVICTED_PROBE_PARENT)
        assert restored_parent is not None, (
            "exit restored a child module whose parent key had been popped "
            "without re-importing the parent -- the next fresh import of "
            f"{_EVICTED_PROBE_PARENT} can never re-link the cached child, so "
            "dotted-string patch targets under it raise AttributeError"
        )
        assert getattr(restored_parent, "path_validation", None) is child, (
            "exit must re-link the restored child as an attribute of the "
            "(re-imported) parent package, mirroring what the snapshot captured"
        )

        # Any later test's import of the parent must now hand back the linked
        # parent rather than executing a fresh, childless copy.
        reimported = importlib.import_module(_EVICTED_PROBE_PARENT)
        assert reimported.path_validation is child

        # The exact pattern that reddened CI: a dotted-string monkeypatch
        # target under the popped parent. Resolving the string walks the
        # parent and reads the child off it as an attribute.
        sentinel = object()
        monkeypatch.setattr(f"{_EVICTED_PROBE_MODULE}.validate_model_path", sentinel)
        assert child.validate_model_path is sentinel, (
            "the dotted-string form must patch the identical child module "
            "object the rest of the suite already holds"
        )
    finally:
        _restore(saved, saved_path)
        # _restore fixes sys.modules keys, not attributes: re-point the
        # grandparent at the original parent in case exit re-imported a
        # replacement while the assertions above were failing.
        grandparent = sys.modules.get(grandparent_name)
        if grandparent is not None:
            setattr(grandparent, parent_attr, parent)
