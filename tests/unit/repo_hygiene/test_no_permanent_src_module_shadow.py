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
