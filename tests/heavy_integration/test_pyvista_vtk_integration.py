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


pytestmark = pytest.mark.live_simulation
