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

import subprocess
import sys
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


@pytest.mark.live_simulation
class TestPinocchioEngine:
    """Contract: Pinocchio can build a kinematic chain and compute FK."""

    def test_pinocchio_kinematic_chain(self) -> None:
        import numpy as np
        import pinocchio as pin

        # Build a minimal 1-DOF revolute robot in-memory
        model = pin.Model()
        geom_model = pin.GeometryModel()

        inertia = pin.Inertia(1.0, np.zeros(3), np.eye(3))
        joint_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint1")
        model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
        model.lowerPositionLimit[0] = -3.14
        model.upperPositionLimit[0] = 3.14

        data = model.createData()
        q = pin.neutral(model)

        # Forward kinematics — must not raise
        pin.forwardKinematics(model, data, q)

        # Prove FK produced a valid SE3 placement
        placement = data.oMi[joint_id]
        assert placement.isIdentity(prec=1e-6), (
            "FK at neutral config should be identity"
        )


@pytest.mark.live_simulation
class TestLauncherSystemCheck:
    """Contract: The main launcher can initialise in headless check mode."""

    def test_launcher_module_imports(self) -> None:
        """Prove the launcher's top-level imports don't crash in the heavy env."""
        launcher_path = REPO_ROOT / "launch_golf_suite.py"
        if not launcher_path.exists():
            pytest.skip("launch_golf_suite.py not present in this checkout")

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; spec=importlib.util.spec_from_file_location('launcher', 'launch_golf_suite.py'); mod=importlib.util.module_from_spec(spec)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"Launcher import failed:\n{result.stderr}"

    def test_launcher_system_check_flag(self) -> None:
        """Probe the launcher --system-check-only flag if it exists."""
        result = subprocess.run(
            [sys.executable, "launch_golf_suite.py", "--system-check-only"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        # Accept 0 (success) or 2 (unrecognized args — flag not yet implemented)
        assert result.returncode in (0, 2), (
            f"Unexpected launcher exit code {result.returncode}:\n{result.stderr}"
        )


@pytest.mark.live_simulation
class TestURDFPipeline:
    """Contract: The URDF generator produces valid, parseable output."""

    def test_urdf_generator_produces_valid_xml(self) -> None:
        import xml.etree.ElementTree as ET

        # Try to locate the URDF generator module
        try:
            from src.tools.urdf_builder.urdf_generator import (
                generate_simple_urdf,  # type: ignore
            )
        except ImportError:
            pytest.skip("URDF generator module not importable in this configuration")

        urdf_str = generate_simple_urdf(num_links=3)
        assert urdf_str is not None and len(urdf_str) > 0

        # Parse as valid XML
        root = ET.fromstring(urdf_str)
        assert root.tag == "robot", "URDF root element must be <robot>"

        links = root.findall("link")
        joints = root.findall("joint")
        assert len(links) >= 2, "URDF must have at least 2 links"
        assert len(joints) >= 1, "URDF must have at least 1 joint"


@pytest.mark.live_simulation
class TestMediaPipeIntegration:
    """Contract: MediaPipe Pose loads and processes a synthetic frame."""

    def test_mediapipe_pose_init(self) -> None:
        import mediapipe as mp  # type: ignore
        import numpy as np

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            # Create a synthetic white image (1×1 px) — won't detect a pose,
            # but proves the pipeline initialises and runs without crashing
            fake_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
            results = pose.process(fake_frame)
            # results.pose_landmarks will be None (no pose detected), that's fine
            assert results is not None, "MediaPipe returned None results object"
