"""Opt-in real-runtime contracts for the Gepetto viewer adapter."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.live_simulation,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_gl,
]


def test_geppetto_viewer_real_model_dispatch_and_cleanup() -> None:
    """Exercise CORBA scene setup and pose dispatch when a server is running."""
    pin = pytest.importorskip("pinocchio")
    pytest.importorskip("gepetto.corbaserver")
    try:
        from src.engines.physics_engines.pinocchio.python.dtack.viz import (
            GeppettoViewer,
        )

        viewer = GeppettoViewer()
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(
            f"gepetto-gui is not reachable; start it in the conda environment: {exc}"
        )

    model = pin.buildSampleModelManipulator()
    collision_model = pin.GeometryModel()
    visual_model = pin.GeometryModel()
    try:
        viewer.load_model(model, visual_model, collision_model=collision_model)
        viewer.display(pin.neutral(model))
        windows = viewer.client.gui.getWindowList()
        assert "python-pinocchio" in windows
    finally:
        viewer.close()
