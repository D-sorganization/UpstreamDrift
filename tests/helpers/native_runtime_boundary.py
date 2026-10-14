"""Keep one pytest process to one native C3D/OpenSim runtime (#12065).

On Windows the OpenSim SDK and the Python ``ezc3d`` extension ship
ABI-incompatible ``ezc3d.dll`` builds.  Whichever loads first owns the DLL name
and the other then fails with ``DLL load failed ... procedure could not be
found`` -- in *both* import orders.  A mixed pytest session therefore turned
test outcomes into a function of collection order.

This module is the test-side process boundary.  Before a test loads one
runtime it checks whether the process already hosts the other (or any foreign
``ezc3d.dll``).  If so the test is skipped with an explicit
:data:`BOUNDARY_SKIP_PREFIX` reason *before* any import is attempted, so the
wrong DLL is never loaded and the skip states that no evidence was produced.
It never catches an import failure: a runtime that fails in a clean process
still fails loudly through ``import_ezc3d``.

The runtime identities come from the clean-process diagnostic
(``scripts/diagnostics/native_runtime_identity.py``) and the DLL probe from the
Tools guard; neither is duplicated here.
"""

from __future__ import annotations

import sys
from typing import Protocol

import pytest

from scripts.diagnostics.native_runtime_identity import RUNTIME_ROLES
from src.shared.python.data_io.native_dll_guard import EZC3D_DLL, loaded_dll_path

__all__ = [
    "BOUNDARY_SKIP_PREFIX",
    "RUNTIME_ROLES",
    "boundary_conflict",
    "check_item_boundary",
    "hosted_runtimes",
    "require_runtime_boundary",
]

BOUNDARY_SKIP_PREFIX = "native-runtime boundary (#12065):"
# pytest marker -> runtime module it loads in the test process.
_MARKER_RUNTIMES = {"requires_opensim": "opensim"}


class _MarkedItem(Protocol):
    def get_closest_marker(self, name: str) -> object | None: ...


def hosted_runtimes() -> tuple[str, ...]:
    """Return the runtime identities whose module this process has imported.

    Postcondition: at most one identity; the tuple is empty in a clean process.

    Raises:
        RuntimeError: the process already hosts both runtimes, i.e. the
            boundary was crossed outside this helper; nothing it reports
            afterwards would be trustworthy evidence.
    """
    hosted = tuple(role for name, role in RUNTIME_ROLES.items() if name in sys.modules)
    if len(hosted) > 1:
        raise RuntimeError(
            f"This process hosts both native runtimes {hosted}; the C3D/OpenSim "
            "process boundary (#12065) was crossed before this check"
        )
    return hosted


def boundary_conflict(module: str) -> str | None:
    """Return why ``module`` must not load in this process, or None if it may.

    Preconditions: ``module`` is a key of :data:`RUNTIME_ROLES`.
    Postcondition: a returned reason starts with :data:`BOUNDARY_SKIP_PREFIX`
    and names what the process already hosts.  Never imports either runtime.
    """
    if module not in RUNTIME_ROLES:
        raise ValueError(
            f"Unknown native runtime {module!r}; expected one of {sorted(RUNTIME_ROLES)}"
        )
    if module in sys.modules:
        return None
    hosted = hosted_runtimes()
    owner = loaded_dll_path(EZC3D_DLL)
    if not hosted and owner is None:
        return None
    detail = f"it hosts {hosted[0]}" if hosted else "a foreign SDK is loaded"
    if owner is not None:
        detail += f" and {EZC3D_DLL} is loaded from {owner}"
    return (
        f"{BOUNDARY_SKIP_PREFIX} {module} ({RUNTIME_ROLES[module]}) cannot load in "
        f"this pytest process because {detail}; the native OpenSim and Python ezc3d "
        "builds are ABI-incompatible. Run these tests in a separate pytest process. "
        "No evidence is produced (not a qualified match)."
    )


def require_runtime_boundary(module: str) -> None:
    """Skip the calling test (or module) if ``module`` would cross the boundary.

    Preconditions: as :func:`boundary_conflict`.  Postcondition: on return the
    process may load ``module`` without contaminating a hosted runtime.
    """
    reason = boundary_conflict(module)
    if reason is not None:
        pytest.skip(reason, allow_module_level=True)


def check_item_boundary(item: _MarkedItem) -> None:
    """Hold a marked test at the boundary before it runs (``runtest_setup``).

    Postcondition: an item carrying a runtime marker (``requires_opensim``) is
    skipped when its runtime would cross the boundary; others are untouched.
    """
    for marker, module in _MARKER_RUNTIMES.items():
        if item.get_closest_marker(marker) is not None:
            require_runtime_boundary(module)
