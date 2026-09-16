"""
Heavy Integration Contracts — Meshcat Visualization
=====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: Meshcat can create a visualizer, add geometry, and render
without crashing in a headless environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.live_simulation, pytest.mark.requires_pinocchio]


def test_meshcat_viewer_real_model_dispatch_and_cleanup() -> None:
    """Exercise MeshCat in a clean interpreter with real geometry."""
    repo_root = Path(__file__).resolve().parents[2]
    dependency_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "names = ('pinocchio', 'meshcat', 'hppfcl'); "
                "missing = [name for name in names if "
                "importlib.util.find_spec(name) is None]; "
                "print(','.join(missing)); "
                "raise SystemExit(2 if missing else 0)"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if dependency_check.returncode == 2:
        pytest.skip(
            "Optional MeshCat integration dependencies unavailable: "
            f"{dependency_check.stdout.strip()}"
        )
    assert dependency_check.returncode == 0, dependency_check.stderr

    smoke_script = textwrap.dedent(
        """
        import numpy as np
        import pinocchio as pin
        import hppfcl

        from src.engines.physics_engines.pinocchio.python.dtack.viz import MeshCatViewer

        model = pin.buildSampleModelManipulator()
        visual_model = pin.GeometryModel()
        visual_model.addGeometryObject(
            pin.GeometryObject(
                "tip",
                model.njoints - 1,
                pin.SE3.Identity(),
                hppfcl.Sphere(0.05),
            )
        )
        viewer = None
        server_proc = None
        try:
            viewer = MeshCatViewer(zmq_url=None, open_browser=False)
            server_proc = viewer.viewer.window.server_proc
            viewer.load_model(model, visual_model)
            q0 = pin.neutral(model)
            q1 = q0.copy()
            q1[0] = 0.25
            viewer.display(q0)
            geometry_id = viewer._visualizer.visual_model.getGeometryId("tip")
            placement_0 = viewer._visualizer.visual_data.oMg[geometry_id].translation.copy()
            viewer.display(q1)
            placement_1 = viewer._visualizer.visual_data.oMg[geometry_id].translation.copy()
            assert viewer.viewer.url()
            assert not np.array_equal(placement_0, placement_1)
        finally:
            if viewer is not None:
                viewer.close()
        assert server_proc is not None
        assert server_proc.poll() is not None
        print(f"placement_0={placement_0}; placement_1={placement_1}")
        """
    )
    smoke = subprocess.run(
        [sys.executable, "-c", smoke_script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert smoke.returncode == 0, f"stdout={smoke.stdout}\nstderr={smoke.stderr}"
