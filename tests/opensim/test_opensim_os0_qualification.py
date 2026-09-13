"""OS-0 Qualification Test: Assert OpenSim native runtime and Moco availability.

Per OpenSim Epic #10003 Work Package OS-0:
This test serves as the mandatory red qualification test that fails when
the native OpenSim Python bindings and Moco optimization capabilities are
missing from the execution environment.
"""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.gate
def test_opensim_binding_installed() -> None:
    """Assert that the native opensim Python module is importable."""
    spec = importlib.util.find_spec("opensim")
    assert spec is not None, (
        "OS-0 RED GATE: 'opensim' module is not installed in the active environment."
    )
    import opensim  # type: ignore[import-not-found]

    assert hasattr(opensim, "Model"), (
        "OS-0 RED GATE: 'opensim.Model' class is missing from installed opensim module."
    )


@pytest.mark.gate
def test_opensim_moco_capabilities_available() -> None:
    """Assert that OpenSim Moco optimization toolchain is functional."""
    spec = importlib.util.find_spec("opensim")
    assert spec is not None, "OS-0 RED GATE: 'opensim' module is not installed."
    import opensim  # type: ignore[import-not-found]

    has_moco = hasattr(opensim, "MocoStudy") or hasattr(opensim, "MocoTrack")
    assert has_moco, (
        "OS-0 RED GATE: OpenSim Moco toolchain (MocoStudy/MocoTrack) is unavailable."
    )
