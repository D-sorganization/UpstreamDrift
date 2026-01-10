import numpy as np
import pytest

from shared.python.statistical_analysis import StatisticalAnalyzer


class TestStatisticalAnalysis:
    def test_rolling_correlation(self):
        """Test rolling correlation computation."""
        # Create two correlated signals
        t = np.linspace(0, 10, 100)
        x = np.sin(t)
        y = np.sin(t)  # Perfect correlation

        # Joint data format: (N, n_joints)
        joint_velocities = np.column_stack([x, y])
        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros_like(joint_velocities),
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_velocities),
        )

        times, corrs = analyzer.compute_rolling_correlation(
            0, 1, window_size=10, data_type="velocity"
        )

        assert len(corrs) > 0
        assert len(times) == len(corrs)
        # Should be close to 1.0 (with some numerical noise)
        np.testing.assert_allclose(corrs, 1.0, atol=1e-5)

    def test_local_divergence_rate(self):
        """Test local divergence rate (basic check)."""
        t = np.linspace(0, 10, 200)
        # Periodic signal (should have low/zero divergence)
        x = np.sin(t)

        joint_velocities = np.column_stack([x, x])  # Dummy
        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros_like(joint_velocities),
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_velocities),
        )

        times, rates = analyzer.compute_local_divergence_rate(
            joint_idx=0, window=10, data_type="velocity"
        )

        # Just check it computes something reasonable
        assert len(rates) > 0
        assert len(times) == len(rates)

    def test_coupling_angles(self):
        """Test vector coding coupling angles."""
        t = np.linspace(0, 2 * np.pi, 100)
        # Proximal: sin(t), Distal: cos(t)
        # Velocities: cos(t), -sin(t)
        # Angle should be changing
        vel1 = np.cos(t)
        vel2 = -np.sin(t)

        joint_velocities = np.column_stack([vel1, vel2])
        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros_like(joint_velocities),
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_velocities),
        )

        angles = analyzer.compute_coupling_angles(0, 1)

        assert len(angles) == 100
        assert np.all((angles >= 0) & (angles < 360))

    def test_coordination_metrics(self):
        """Test coordination metrics."""
        t = np.linspace(0, 2 * np.pi, 100)
        # In-phase: both moving positive or negative together
        vel1 = np.sin(t)
        vel2 = np.sin(t)

        joint_velocities = np.column_stack([vel1, vel2])
        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros_like(joint_velocities),
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_velocities),
        )

        metrics = analyzer.compute_coordination_metrics(0, 1)

        # Should be predominantly in-phase
        assert metrics is not None
        assert metrics.in_phase_pct > 80.0

    def test_recurrence_matrix_and_rqa(self):
        """Test Recurrence Plot and RQA metrics."""
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)

        joint_positions = x.reshape(-1, 1)
        joint_velocities = np.cos(t).reshape(-1, 1)

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_positions),
        )

        rec_mat = analyzer.compute_recurrence_matrix(threshold_ratio=0.2)
        assert rec_mat.shape == (200, 200)

        rqa = analyzer.compute_rqa_metrics(rec_mat)
        assert rqa is not None
        assert rqa.recurrence_rate > 0

    def test_swing_dna(self):
        """Test Swing DNA metric computation."""
        t = np.linspace(0, 1, 100)
        club_speed = np.zeros(100)
        club_speed[50] = 50.0  # m/s ~ 111 mph

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 3)),
            joint_velocities=np.zeros((100, 3)),
            joint_torques=np.zeros((100, 3)),
            club_head_speed=club_speed,
        )

        dna = analyzer.compute_swing_dna()
        assert dna is not None
        assert dna.speed_score > 80.0  # Should be high

    def test_phase_space_path_length(self):
        """Test phase space path length."""
        t = np.linspace(0, 1, 100)
        pos = t  # Linear position
        vel = np.ones(100)  # Constant velocity

        # Path in phase space: (0,1) to (1,1). Length = 1.

        joint_positions = pos.reshape(-1, 1)
        joint_velocities = vel.reshape(-1, 1)

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=np.zeros_like(joint_positions),
        )

        length = analyzer.compute_phase_space_path_length(0)

        # It calculates discrete sum of distances.
        # N points. distance between (t, 1) and (t+dt, 1) is dt.
        # Sum should be approx 1.

        np.testing.assert_allclose(length, 1.0, rtol=0.05)

    def test_energy_metrics(self):
        """Test energy metrics."""
        t = np.linspace(0, 1, 100)
        ke = t * 10.0  # increasing KE
        pe = (1 - t) * 10.0  # decreasing PE

        club_speed = np.zeros(100)
        club_speed[90] = 50.0  # Impact at end

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 1)),
            joint_velocities=np.zeros((100, 1)),
            joint_torques=np.zeros((100, 1)),
            club_head_speed=club_speed,
        )

        metrics = analyzer.compute_energy_metrics(ke, pe)

        assert metrics["max_total_energy"] > 0
        assert metrics["energy_efficiency"] > 0
        assert metrics["energy_drift"] == pytest.approx(0.0)

    def test_impact_detection(self):
        """Test impact detection."""
        t = np.linspace(0, 1, 100)
        club_speed = np.zeros(100)
        club_speed[50] = 50.0  # Peak

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 1)),
            joint_velocities=np.zeros((100, 1)),
            joint_torques=np.zeros((100, 1)),
            club_head_speed=club_speed,
        )

        impact_time = analyzer.detect_impact_time()
        assert impact_time == pytest.approx(t[50])

    def test_work_metrics(self):
        """Test work metrics."""
        t = np.linspace(0, 1, 100)
        # Torque = 1, Vel = 1 -> Power = 1
        # Work = integral(1) dt = 1.0

        torques = np.ones((100, 1))
        vels = np.ones((100, 1))

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 1)),
            joint_velocities=vels,
            joint_torques=torques,
        )

        work = analyzer.compute_work_metrics(0)
        assert work is not None
        assert work["net_work"] == pytest.approx(1.0, rel=0.05)
        assert work["positive_work"] == pytest.approx(1.0, rel=0.05)
        assert work["negative_work"] == 0.0

    def test_joint_power_metrics(self):
        """Test joint power metrics."""
        t = np.linspace(0, 1, 100)
        torques = np.ones((100, 1))
        vels = np.ones((100, 1))

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 1)),
            joint_velocities=vels,
            joint_torques=torques,
        )

        metrics = analyzer.compute_joint_power_metrics(0)
        assert metrics is not None
        assert metrics.peak_generation == pytest.approx(1.0)
        assert metrics.net_work == pytest.approx(1.0, rel=0.05)

    def test_impulse_metrics(self):
        """Test impulse metrics."""
        t = np.linspace(0, 1, 100)
        torques = np.ones((100, 1))

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros((100, 1)),
            joint_velocities=np.zeros((100, 1)),
            joint_torques=torques,
        )

        impulse = analyzer.compute_impulse_metrics(data_type="torque", joint_idx=0)
        assert impulse is not None
        assert impulse.net_impulse == pytest.approx(1.0, rel=0.05)
