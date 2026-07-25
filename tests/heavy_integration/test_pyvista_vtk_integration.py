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

        # Rendering failures used to be converted into a skip, which meant this
        # contract test could never fail (#8035). A genuinely absent GPU/display
        # surfaces as a RuntimeError from VTK during plotter construction -- that
        # is the only condition worth skipping for, and it is caught narrowly.
        # Once a plotter exists, the assertions below must hold.
        try:
            plotter = pv.Plotter(off_screen=True)
        except RuntimeError as exc:
            pytest.skip(f"no offscreen rendering context available: {exc}")

        try:
            plotter.add_mesh(pv.Sphere())
            img = plotter.screenshot(return_img=True)
        finally:
            plotter.close()

        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0


pytestmark = pytest.mark.live_simulation
