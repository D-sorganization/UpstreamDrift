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


@pytest.mark.unit
def test_baseline_historical_model_rejected_by_qualification_gate() -> None:
    """Assert that the historical matching baseline is rejected by qualification gates.

    Per OG-01 (#10395): The historical baseline model exhibits missing club attached
    geometry and unscaled arm meshes, and must fail closed with typed rationale.
    """
    from pathlib import Path
    from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
        verify_model_qualification,
    )

    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "development"
        / "opensim_tour_matching"
        / "evidence"
        / "os7_moco_g1"
        / "golf_humanoid_scaled_tour_markers_moco.osim"
    )
    if baseline_path.is_file():
        with pytest.raises(
            ValueError, match="Club body has no attached visual geometry"
        ):
            verify_model_qualification(baseline_path, require_visible_club=True)

        with pytest.raises(ValueError, match="Arm meshes remain at unit scale"):
            verify_model_qualification(
                baseline_path,
                require_visible_club=False,
                require_consistent_arm_scaling=True,
            )
