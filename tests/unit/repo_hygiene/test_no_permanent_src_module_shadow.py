"""Guard against a directory-scoped conftest permanently shadowing ``src``.

Regression coverage for the follow-up to issue #8834 / PR #9374.

``tests/unit/c3d_viewer/ui/conftest.py`` and
``tests/unit/engines/simscape/three_d_gui/conftest.py`` both rebind the
top-level ``sys.modules["src"]`` entry to
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src``'s own
(unrelated) ``src`` package, so that the C3D-viewer engine's ``src.apps.*``
absolute imports resolve inside its own tests.

Before this fix, that rebind happened once, unconditionally, at module
import time, and was never undone. ``sys.modules`` is process-global, and
under ``pytest-xdist`` a single worker process collects and runs tests from
every directory in the suite -- so whichever of these two directories got
collected first in a given worker permanently poisoned ``sys.modules["src"]``
for every test that ran afterwards in that same worker process, including
``tests/scripts/test_validate_suite.py``, whose
``launch_upstream_drift._retry_parent_shared_alias_installer()`` does a bare
``import src`` and expects the real
``src/__init__.py::_install_parent_shared_aliases``. When the wrong ``src``
was cached, that lookup silently returned ``None`` (post-#9374 workaround)
instead of actually installing the parent-shared aliases -- the underlying
pollution was never fixed, only the crash was suppressed.

The fix scopes the rebind to only be active while pytest is collecting or
running an item that lives under the conftest's own directory (see the
``pytest_make_collect_report`` / ``pytest_runtest_setup`` /
``pytest_runtest_teardown`` hooks there, which pytest itself only consults
for collectors/items at or below that directory). This test verifies that
the pivot's own enter/exit functions never leave ``sys.modules["src"]``
pointing at the engine package once exited -- including when nested, which
is what actually happens as pytest's collectors (Package -> Module ->
Class -> Function) and the runtest setup/teardown hooks both re-enter the
same pivot.

Note: reproducing the originally-reported failure end-to-end requires the
full ``testpaths`` collection sweep (a minimal two-item invocation does not
trigger it, because ``tests/conftest.py::pytest_configure`` happens to
import the real ``src`` package first in that narrow case) -- that
end-to-end confirmation lives in the manual investigation for this issue,
not as a subprocess test here, to keep this suite fast and non-flaky.
"""

from __future__ import annotations

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


def _load_conftest(path: Path) -> ModuleType:
    """Load a directory-scoped conftest.py under a private module name.

    A distinct name per call avoids colliding with pytest's own conftest
    bookkeeping in ``sys.modules`` while still executing the file's real
    top-level code (needed to define ``_enter_pivot``/``_exit_pivot``).
    """
    spec = importlib.util.spec_from_file_location(
        f"_repo_hygiene_probe_{path.parent.name}_conftest", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=lambda p: p.parent.name)
def test_pivot_conftest_defines_scoped_enter_exit_hooks(conftest_path: Path) -> None:
    """DbC: the pivot must expose paired, idempotent enter/exit functions.

    A bare module-level ``_pivot_sys_path()`` call with no counterpart is
    exactly the shape of the original bug -- this fails loudly if either
    directory's conftest regresses back to that shape.
    """
    assert conftest_path.is_file(), f"Expected conftest at {conftest_path}"
    module = _load_conftest(conftest_path)

    assert callable(getattr(module, "_enter_pivot", None)), (
        f"{conftest_path} must define _enter_pivot() to scope the src rebind"
    )
    assert callable(getattr(module, "_exit_pivot", None)), (
        f"{conftest_path} must define _exit_pivot() to undo the src rebind"
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

    Simulates the exact mechanism that corrupted ``sys.modules["src"]``:
    calling the pivot's enter without an eventual matching exit is the bug.
    This drives enter/exit directly (nested twice, matching how collection
    of a Package -> Module -> Function chain re-enters the same pivot) and
    asserts the top-level ``src`` identity is bit-for-bit restored after the
    outermost exit.
    """
    module = _load_conftest(conftest_path)

    previous = sys.modules.get("src")
    try:
        module._enter_pivot()
        pivoted = sys.modules.get("src")
        assert pivoted is not None, "pivot must install a src module"
        assert pivoted is not previous, (
            "pivot did not actually rebind src to the engine package"
        )

        # Nested entry (mirrors nested collectors reusing the same pivot).
        module._enter_pivot()
        assert sys.modules.get("src") is pivoted
        module._exit_pivot()
        assert sys.modules.get("src") is pivoted, (
            "an inner exit must not undo the pivot while an outer scope is still active"
        )

        module._exit_pivot()
        assert sys.modules.get("src") is previous, (
            "the outermost exit must restore the exact prior src identity "
            "-- this is the invariant that keeps the pivot from leaking "
            "into unrelated tests that run later in the same xdist worker"
        )
    finally:
        # Belt-and-suspenders: never let a failed assertion above leave
        # this test process's own sys.modules['src'] corrupted for
        # whatever collects/runs after it.
        if sys.modules.get("src") is not previous:
            if previous is not None:
                sys.modules["src"] = previous
            else:
                sys.modules.pop("src", None)
