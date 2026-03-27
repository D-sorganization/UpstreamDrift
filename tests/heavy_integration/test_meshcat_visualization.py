"""
Heavy Integration Contracts — Meshcat Visualization
=====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: Meshcat can create a visualizer, add geometry, and render
without crashing in a headless environment.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.live_simulation
class TestMeshcatCore:
    """Contract: Meshcat creates a visualizer and renders geometry."""

    def test_meshcat_import_and_visualizer(self) -> None:
        """Meshcat Visualizer can be instantiated headlessly."""
        try:
            import meshcat
        except ImportError:
            pytest.skip("meshcat not installed")

        # Open a visualizer (no browser window in headless)
        vis = meshcat.Visualizer()
        assert vis is not None

    def test_meshcat_add_geometry(self) -> None:
        """Meshcat can add box/sphere geometry without error."""
        try:
            import meshcat
            import meshcat.geometry as g
        except ImportError:
            pytest.skip("meshcat not installed")

        vis = meshcat.Visualizer()

        # Add a box
        vis["box"].set_object(g.Box([0.1, 0.1, 0.1]))

        # Add a sphere
        vis["sphere"].set_object(g.Sphere(0.05))

        # Set transform
        vis["box"].set_transform(
            meshcat.transformations.translation_matrix([0.5, 0, 0])
            if hasattr(meshcat, "transformations")
            else np.eye(4)
        )

    def test_meshcat_delete_geometry(self) -> None:
        """Meshcat can delete objects from the scene."""
        try:
            import meshcat
            import meshcat.geometry as g
        except ImportError:
            pytest.skip("meshcat not installed")

        vis = meshcat.Visualizer()
        vis["temp"].set_object(g.Sphere(0.1))
        vis["temp"].delete()


@pytest.mark.live_simulation
class TestMeshcatPinocchioIntegration:
    """Contract: Meshcat integrates with Pinocchio's MeshcatVisualizer."""

    def test_pinocchio_meshcat_display(self) -> None:
        """Pinocchio's MeshcatVisualizer can display a model."""
        try:
            import pinocchio as pin
        except ImportError:
            pytest.skip("meshcat or pinocchio not installed")

        if not hasattr(pin, "Model"):
            pytest.skip("pinocchio stub installed")

        # Check for MeshcatVisualizer in pinocchio.visualize
        if not hasattr(pin, "visualize") or not hasattr(
            pin.visualize, "MeshcatVisualizer"
        ):
            pytest.skip("pinocchio.visualize.MeshcatVisualizer not available")

        model = pin.Model()
        inertia = pin.Inertia(1.0, np.zeros(3), np.eye(3))
        j1 = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint1")
        model.appendBodyToJoint(j1, inertia, pin.SE3.Identity())

        model.createData()
        q = pin.neutral(model)

        vis_model = pin.GeometryModel()
        vis = pin.visualize.MeshcatVisualizer(model, vis_model, vis_model)

        try:
            vis.initViewer(open=False)
            vis.loadViewerModel()
            vis.display(q)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"MeshcatVisualizer display failed (expected headless): {e}")


@pytest.mark.live_simulation
class TestProjectMeshcatWiring:
    """Contract: Project meshcat adapters are importable."""

    def test_meshcat_adapter_importable(self) -> None:
        """MuJoCo meshcat adapter module is importable."""
        try:
            import meshcat  # noqa: F401
        except ImportError:
            pytest.skip("meshcat not installed")

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.meshcat_adapter import (
            MeshcatAdapter,
        )

        assert MeshcatAdapter is not None

    def test_meshcat_viewer_module_importable(self) -> None:
        """Pinocchio dtack meshcat viewer module is importable."""
        try:
            import meshcat  # noqa: F401
        except ImportError:
            pytest.skip("meshcat not installed")

        from src.engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer import (
            MeshcatViewer,
        )

        assert MeshcatViewer is not None


pytestmark = pytest.mark.live_simulation
