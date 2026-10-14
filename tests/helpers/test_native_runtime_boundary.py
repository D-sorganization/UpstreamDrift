"""The pytest process never hosts both native C3D and OpenSim runtimes (#12065).

On Windows the OpenSim SDK and the Python ``ezc3d`` extension ship
ABI-incompatible ``ezc3d.dll`` builds; whichever loads first breaks the other
(reproduced in both import orders).  These tests pin the test-side boundary:
once the process hosts one runtime, a test needing the other is skipped with
an explicit boundary reason *before* any import is attempted, so no DLL is
loaded into the wrong process and no skip masquerades as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from scripts.diagnostics.frozen_capture_trc import clean_process_environment
from scripts.diagnostics.native_runtime_identity import RUNTIME_ROLES
from tests.helpers import ezc3d_support
from tests.helpers import native_runtime_boundary as boundary

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_runtimes(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A process view in which neither runtime nor a foreign ezc3d.dll is loaded."""
    for name in RUNTIME_ROLES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(boundary, "loaded_dll_path", lambda name: None)
    return monkeypatch


def _host(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_runtime_roles_are_the_diagnostic_identities() -> None:
    assert boundary.RUNTIME_ROLES is RUNTIME_ROLES
    assert set(RUNTIME_ROLES) == {"ezc3d", "opensim"}


def test_clean_process_hosts_no_runtime_and_has_no_conflict(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    assert boundary.hosted_runtimes() == ()
    for name in RUNTIME_ROLES:
        assert boundary.boundary_conflict(name) is None
        boundary.require_runtime_boundary(name)  # must not skip


@pytest.mark.parametrize(
    ("hosted", "wanted"), [("opensim", "ezc3d"), ("ezc3d", "opensim")]
)
def test_hosting_one_runtime_is_a_conflict_for_the_other(
    clean_runtimes: pytest.MonkeyPatch, hosted: str, wanted: str
) -> None:
    _host(clean_runtimes, hosted)
    assert boundary.hosted_runtimes() == (RUNTIME_ROLES[hosted],)
    assert boundary.boundary_conflict(hosted) is None
    reason = boundary.boundary_conflict(wanted)
    assert reason is not None
    assert reason.startswith(boundary.BOUNDARY_SKIP_PREFIX)
    assert RUNTIME_ROLES[hosted] in reason
    assert "not a qualified match" in reason
    with pytest.raises(pytest.skip.Exception, match="native-runtime boundary"):
        boundary.require_runtime_boundary(wanted)


@pytest.mark.parametrize("wanted", sorted(RUNTIME_ROLES))
def test_foreign_ezc3d_dll_owner_is_a_conflict(
    clean_runtimes: pytest.MonkeyPatch, wanted: str
) -> None:
    owner = r"C:\sdk\bin\ezc3d.dll"
    clean_runtimes.setattr(boundary, "loaded_dll_path", lambda name: owner)
    reason = boundary.boundary_conflict(wanted)
    assert reason is not None and owner in reason


def test_already_contaminated_process_is_a_hard_error(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    for name in RUNTIME_ROLES:
        _host(clean_runtimes, name)
    with pytest.raises(RuntimeError, match="both"):
        boundary.hosted_runtimes()


def test_unknown_runtime_is_rejected() -> None:
    with pytest.raises(ValueError, match="mujoco"):
        boundary.boundary_conflict("mujoco")


def test_require_ezc3d_never_imports_into_an_opensim_process(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    _host(clean_runtimes, "opensim")

    def _forbidden() -> types.ModuleType:
        raise AssertionError("ezc3d import attempted inside an OpenSim process")

    clean_runtimes.setattr(ezc3d_support, "import_ezc3d", _forbidden)
    with pytest.raises(pytest.skip.Exception, match="native-runtime boundary"):
        ezc3d_support.require_ezc3d()
    with pytest.raises(pytest.skip.Exception, match="native-runtime boundary"):
        ezc3d_support.ezc3d_installed()


def test_requires_opensim_marker_is_held_at_the_boundary(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    _host(clean_runtimes, "ezc3d")
    marked = types.SimpleNamespace(
        get_closest_marker=lambda name: object() if name == "requires_opensim" else None
    )
    unmarked = types.SimpleNamespace(get_closest_marker=lambda name: None)
    boundary.check_item_boundary(unmarked)  # must not skip
    with pytest.raises(pytest.skip.Exception, match="native-runtime boundary"):
        boundary.check_item_boundary(marked)


def test_real_child_process_hosting_opensim_skips_c3d_without_importing_it() -> None:
    """End to end in a clean child: no ``ezc3d`` import is even attempted."""
    child = """
import json, sys, types
attempts = []
class _Recorder:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] == "ezc3d":
            attempts.append(name)
        return None
sys.meta_path.insert(0, _Recorder())
sys.modules["opensim"] = types.ModuleType("opensim")
import pytest
from tests.helpers.ezc3d_support import require_ezc3d
try:
    require_ezc3d()
    outcome = "imported"
except pytest.skip.Exception as skip:
    outcome = str(skip.msg)
print(json.dumps({"attempts": attempts, "outcome": outcome,
                  "ezc3d_loaded": "ezc3d" in sys.modules}))
"""
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPO_ROOT,
        env=clean_process_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["attempts"] == []
    assert observed["ezc3d_loaded"] is False
    assert observed["outcome"].startswith(boundary.BOUNDARY_SKIP_PREFIX)
