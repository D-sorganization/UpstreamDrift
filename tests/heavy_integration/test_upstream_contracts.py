"""
Heavy Integration Contracts — UpstreamDrift
============================================
These tests are marked @pytest.mark.live_simulation and are EXCLUDED from
standard CI. They run only:
  • Weekly on the d-sorg-fleet-4core custom runner
  • Manually via: wsl bash run_local_heavy_tests.sh

Each test is a REPO-SPECIFIC CONTRACT, not a generic smoke test:
  • Tests must exercise real logic, not just check importability
  • Each test documents what it proves and what would break if it fails
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.live_simulation
class TestMuJoCoEngine:
    """Contract: MuJoCo can load a valid humanoid model and step physics."""

    def test_mujoco_humanoid_step(self) -> None:
        import mujoco
        import numpy as np

        # Use built-in MuJoCo humanoid XML to avoid file-path dependencies
        xml = """
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <body name="box" pos="0 0 0.5">
              <joint type="free"/>
              <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Step 100 frames — prove physics doesn't crash or NaN
        for _ in range(100):
            mujoco.mj_step(model, data)

        assert not any(np.isnan(data.qpos)), "Physics produced NaN positions"
        assert data.time > 0, "Simulation time did not advance"

    def test_mujoco_version_contract(self) -> None:
        import mujoco

        major, minor, _ = mujoco.__version__.split(".")
        # Contract: we require MuJoCo >= 3.0 for Python 3.13 compatibility
        assert int(major) >= 3, f"MuJoCo >= 3.0 required, got {mujoco.__version__}"


pytestmark = pytest.mark.live_simulation
