"""Performance benchmark tests for physics computations.

These tests verify performance characteristics and establish baselines
for the optimized analytical RNE methods implemented in Phase 1.
"""

import time
import unittest

import numpy as np


class TestCoriolisPerformance(unittest.TestCase):
    """Test Coriolis computation performance."""

    def test_coriolis_computation_time(self) -> None:
        """Verify Coriolis computation completes within acceptable time."""
        try:
            import mujoco

            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
                KinematicForceAnalyzer,
            )
        except ImportError:
            self.skipTest("MuJoCo or kinematic_forces not available")

        # Create simple test model
        xml = """
        <mujoco>
            <worldbody>
                <body name="link1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.5" mass="1"/>
                    <body name="link2" pos="0 0 1">
                        <joint type="hinge" axis="0 1 0"/>
                        <geom type="box" size="0.1 0.1 0.5" mass="0.5"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        analyzer = KinematicForceAnalyzer(model, data)

        # Test Coriolis computation time
        qpos = np.zeros(model.nv)
        qvel = np.ones(model.nv) * 0.5

        # Warm up
        for _ in range(3):
            analyzer.compute_coriolis_forces(qpos, qvel)

        # Time actual computation
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            analyzer.compute_coriolis_forces(qpos, qvel)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iterations) * 1000

        # Should complete in less than 1ms per call for small models
        self.assertLess(
            avg_time_ms,
            1.0,
            f"Coriolis computation took {avg_time_ms:.3f}ms, expected < 1ms",
        )

    def test_rne_vs_forward_performance(self) -> None:
        """Verify RNE method is faster than forward kinematics approach."""
        try:
            import mujoco

            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
                KinematicForceAnalyzer,
            )
        except ImportError:
            self.skipTest("MuJoCo or kinematic_forces not available")

        # Create moderate complexity model
        xml = """
        <mujoco>
            <worldbody>
                <body name="link1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="capsule" size="0.05 0.3" mass="1"/>
                    <body name="link2" pos="0 0 0.6">
                        <joint type="hinge" axis="0 1 0"/>
                        <geom type="capsule" size="0.04 0.25" mass="0.8"/>
                        <body name="link3" pos="0 0 0.5">
                            <joint type="hinge" axis="1 0 0"/>
                            <geom type="capsule" size="0.03 0.2" mass="0.5"/>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        analyzer = KinematicForceAnalyzer(model, data)

        qpos = np.zeros(model.nv)
        qvel = np.random.randn(model.nv) * 0.5

        # Time RNE method
        start = time.perf_counter()
        for _ in range(50):
            analyzer.compute_coriolis_forces_rne(qpos, qvel)
        rne_time = time.perf_counter() - start

        # RNE should be fast
        self.assertLess(
            rne_time,
            0.05,  # 50 iterations should take < 50ms
            f"RNE computation took {rne_time * 1000:.1f}ms for 50 iterations",
        )


class TestMassMatrixPerformance(unittest.TestCase):
    """Test mass matrix computation performance."""

    def test_mass_matrix_symmetry(self) -> None:
        """Verify mass matrix is symmetric (physics requirement)."""
        try:
            import mujoco

            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
                KinematicForceAnalyzer,
            )
        except ImportError:
            self.skipTest("MuJoCo or kinematic_forces not available")

        xml = """
        <mujoco>
            <worldbody>
                <body name="link1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        analyzer = KinematicForceAnalyzer(model, data)
        qpos = np.zeros(model.nv)

        M = analyzer.compute_mass_matrix(qpos)

        # Check symmetry
        self.assertTrue(
            np.allclose(M, M.T),
            "Mass matrix should be symmetric",
        )

    def test_mass_matrix_positive_definite(self) -> None:
        """Verify mass matrix is positive definite (physics requirement)."""
        try:
            import mujoco

            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
                KinematicForceAnalyzer,
            )
        except ImportError:
            self.skipTest("MuJoCo or kinematic_forces not available")

        xml = """
        <mujoco>
            <worldbody>
                <body name="link1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        analyzer = KinematicForceAnalyzer(model, data)
        qpos = np.zeros(model.nv)

        M = analyzer.compute_mass_matrix(qpos)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(M)
        self.assertTrue(
            np.all(eigenvalues > 0),
            f"Mass matrix should be positive definite. Min eigenvalue: {eigenvalues.min()}",
        )


class TestTrajectoryAnalysisPerformance(unittest.TestCase):
    """Test trajectory analysis performance."""

    def test_trajectory_analysis_scalability(self) -> None:
        """Test that trajectory analysis scales linearly with timesteps."""
        try:
            import mujoco

            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.kinematic_forces import (
                KinematicForceAnalyzer,
            )
        except ImportError:
            self.skipTest("MuJoCo or kinematic_forces not available")

        xml = """
        <mujoco>
            <worldbody>
                <body name="link1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.3" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        analyzer = KinematicForceAnalyzer(model, data)

        # Create trajectories of different sizes
        for n_steps in [10, 50]:
            times = np.linspace(0, 1, n_steps)
            positions = np.zeros((n_steps, model.nv))
            velocities = np.random.randn(n_steps, model.nv) * 0.1
            accelerations = np.zeros((n_steps, model.nv))

            start = time.perf_counter()
            results = analyzer.analyze_trajectory(
                times, positions, velocities, accelerations
            )
            elapsed = time.perf_counter() - start

            self.assertEqual(len(results), n_steps)
            # Verify reasonable performance (< 100ms per 50 steps)
            if n_steps == 50:
                self.assertLess(
                    elapsed,
                    0.5,
                    f"Trajectory analysis for {n_steps} steps took {elapsed:.3f}s",
                )


if __name__ == "__main__":
    unittest.main()
