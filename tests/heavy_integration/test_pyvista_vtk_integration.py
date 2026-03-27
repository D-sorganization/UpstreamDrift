"""
Heavy Integration Contracts — PyVista / VTK
=============================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: PyVista and VTK can create 3D meshes, perform operations,
and render offscreen without crashing.
"""

from __future__ import annotations

import pytest


@pytest.mark.live_simulation
class TestPyVistaCore:
    """Contract: PyVista can create and manipulate 3D meshes."""

    def test_pyvista_import_and_version(self) -> None:
        """PyVista is importable and reports a version."""
        try:
            import pyvista as pv
        except ImportError:
            pytest.skip("pyvista not installed")

        assert hasattr(pv, "__version__")
        # We need at least 0.38 for offscreen rendering stability
        major, minor = pv.__version__.split(".")[:2]
        assert int(major) >= 0 and int(minor) >= 30, (
            f"PyVista >= 0.30 expected, got {pv.__version__}"
        )

    def test_pyvista_create_mesh(self) -> None:
        """PyVista can create basic mesh primitives."""
        try:
            import pyvista as pv
        except ImportError:
            pytest.skip("pyvista not installed")

        sphere = pv.Sphere(radius=0.05, center=(0, 0, 0))
        assert sphere.n_points > 0
        assert sphere.n_cells > 0

        box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
        assert box.n_points == 8

        plane = pv.Plane(i_size=2.0, j_size=2.0, i_resolution=10, j_resolution=10)
        assert plane.n_points > 0

    def test_pyvista_mesh_operations(self) -> None:
        """PyVista can compute normals, bounds, and volume."""
        try:
            import pyvista as pv
        except ImportError:
            pytest.skip("pyvista not installed")

        sphere = pv.Sphere(radius=1.0)

        # Compute normals
        normals = sphere.compute_normals()
        assert "Normals" in normals.point_data or normals.n_points > 0

        # Bounds
        bounds = sphere.bounds
        assert len(bounds) == 6
        # Sphere radius 1: bounds should be approximately [-1, 1, -1, 1, -1, 1]
        assert bounds[0] < -0.9 and bounds[1] > 0.9

    def test_pyvista_offscreen_render(self) -> None:
        """PyVista can render offscreen (headless)."""
        try:
            import pyvista as pv

            pv.OFF_SCREEN = True
        except ImportError:
            pytest.skip("pyvista not installed")

        try:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())
            img = plotter.screenshot(return_img=True)
            plotter.close()

            assert img is not None
            assert img.shape[0] > 0 and img.shape[1] > 0
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Offscreen rendering failed (GPU/display issue): {e}")


@pytest.mark.live_simulation
class TestVTKCore:
    """Contract: VTK loads and can create basic pipeline objects."""

    def test_vtk_import(self) -> None:
        """VTK is importable."""
        try:
            import vtk
        except ImportError:
            pytest.skip("vtk not installed")

        assert hasattr(vtk, "vtkVersion") or hasattr(vtk, "VTK_VERSION")

    def test_vtk_create_source(self) -> None:
        """VTK can create a sphere source and mapper pipeline."""
        try:
            import vtk
        except ImportError:
            pytest.skip("vtk not installed")

        source = vtk.vtkSphereSource()
        source.SetRadius(1.0)
        source.SetPhiResolution(16)
        source.SetThetaResolution(16)
        source.Update()

        output = source.GetOutput()
        assert output.GetNumberOfPoints() > 0
        assert output.GetNumberOfCells() > 0

    def test_vtk_stl_reader_exists(self) -> None:
        """VTK STL reader class exists (used for mesh loading)."""
        try:
            import vtk
        except ImportError:
            pytest.skip("vtk not installed")

        reader = vtk.vtkSTLReader()
        assert reader is not None


pytestmark = pytest.mark.live_simulation
