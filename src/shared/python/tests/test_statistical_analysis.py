import numpy as np
import pytest

from src.shared.python.statistical_analysis import (
    CoordinationMetrics,
    GRFMetrics,
    JointStiffnessMetrics,
    PCAResult,
    PeakInfo,
    RQAMetrics,
    StabilityMetrics,
    StatisticalAnalyzer,
    SummaryStatistics,
    SwingPhase,
)


class TestStatisticalAnalyzer:
    @pytest.fixture
    def sample_data(self):
        # Create synthetic data
        fs = 100.0
        duration = 2.0
        t = np.arange(0, duration, 1 / fs)
        n_samples = len(t)
        n_joints = 3

        # Positions: Sinusoidal with different phases/frequencies
        joint_positions = np.zeros((n_samples, n_joints))
        joint_positions[:, 0] = np.sin(2 * np.pi * 1.0 * t)  # Joint 0
        joint_positions[:, 1] = np.sin(2 * np.pi * 1.5 * t + 0.5)  # Joint 1
        joint_positions[:, 2] = np.cos(2 * np.pi * 0.5 * t)  # Joint 2

        # Velocities: Cosine
        joint_velocities = np.zeros((n_samples, n_joints))
        joint_velocities[:, 0] = 2 * np.pi * 1.0 * np.cos(2 * np.pi * 1.0 * t)
        joint_velocities[:, 1] = 2 * np.pi * 1.5 * np.cos(2 * np.pi * 1.5 * t + 0.5)
        joint_velocities[:, 2] = -2 * np.pi * 0.5 * np.sin(2 * np.pi * 0.5 * t)

        # Torques: Proportional to acceleration (approx)
        joint_torques = np.zeros((n_samples, n_joints))
        joint_torques[:, 0] = -((2 * np.pi * 1.0) ** 2) * np.sin(2 * np.pi * 1.0 * t)
        joint_torques[:, 1] = -((2 * np.pi * 1.5) ** 2) * np.sin(
            2 * np.pi * 1.5 * t + 0.5
        )
        joint_torques[:, 2] = -((2 * np.pi * 0.5) ** 2) * np.cos(2 * np.pi * 0.5 * t)

        # Club Head Speed (Gaussian profile for swing)
        club_head_speed = 50.0 * np.exp(-10 * (t - 1.0) ** 2)  # Peak at 1.0s

        # Ground Forces
        ground_forces = np.zeros((n_samples, 3))
        ground_forces[:, 2] = 700 + 300 * np.sin(2 * np.pi * 1.0 * t)  # Vertical

        # COP Position
        cop_position = np.zeros((n_samples, 2))
        cop_position[:, 0] = 0.1 * np.sin(2 * np.pi * 0.5 * t)

        return {
            "times": t,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "club_head_speed": club_head_speed,
            "ground_forces": ground_forces,
            "cop_position": cop_position,
        }

    def test_initialization(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        assert len(analyzer.times) == 200
        assert analyzer.dt == 0.01
        assert analyzer.duration == 1.99

    def test_compute_summary_stats(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        data = sample_data["joint_positions"][:, 0]
        stats = analyzer.compute_summary_stats(data)

        assert isinstance(stats, SummaryStatistics)
        assert np.isclose(stats.max, 1.0, atol=0.01)
        assert np.isclose(stats.min, -1.0, atol=0.01)

    def test_find_peaks_in_data(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        data = sample_data["joint_positions"][:, 0]  # Sine wave
        peaks = analyzer.find_peaks_in_data(data, height=0.5)

        assert len(peaks) > 0
        assert isinstance(peaks[0], PeakInfo)
        assert peaks[0].value > 0.5

    def test_find_club_head_speed_peak(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        peak = analyzer.find_club_head_speed_peak()

        assert peak is not None
        assert np.isclose(peak.time, 1.0, atol=0.02)
        assert np.isclose(peak.value, 50.0, atol=0.1)

    def test_compute_range_of_motion(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        # Joint 0 is sin(t), range [-1, 1] rad -> [-57.3, 57.3] deg
        # ROM ~ 114.6 deg
        min_deg, max_deg, rom = analyzer.compute_range_of_motion(0)
        assert np.isclose(rom, 2 * np.rad2deg(1.0), atol=1.0)

    def test_compute_tempo(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        # With Gaussian speed peak at 1.0s
        # Start search 5 (0.05s).
        # Smooth speed.
        # This synthetic data might not have a distinct "backswing" transition unless we shape it.
        # But let's check it runs.
        tempo = analyzer.compute_tempo()
        # Might be None if transition not found
        if tempo:
            bs, ds, ratio = tempo
            assert bs > 0
            assert ds > 0

    def test_detect_swing_phases(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        phases = analyzer.detect_swing_phases()

        assert len(phases) > 0
        assert isinstance(phases[0], SwingPhase)

        # Check if Impact phase is around 1.0s
        impact_phases = [p for p in phases if p.name == "Impact"]
        if impact_phases:
            assert np.isclose(impact_phases[0].start_time, 1.0, atol=0.2)

    def test_compute_grf_metrics(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        metrics = analyzer.compute_grf_metrics()

        assert isinstance(metrics, GRFMetrics)
        assert metrics.cop_path_length > 0
        assert metrics.peak_vertical_force is not None
        assert metrics.peak_vertical_force > 0

    def test_compute_stability_metrics(self, sample_data):
        # Needs COM
        data = sample_data.copy()
        data["com_position"] = np.zeros((200, 3))  # COM at origin
        # COP is oscillating
        analyzer = StatisticalAnalyzer(**data)

        metrics = analyzer.compute_stability_metrics()
        assert isinstance(metrics, StabilityMetrics)
        assert metrics.mean_com_cop_distance > 0

    def test_compute_coordination_metrics(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        metrics = analyzer.compute_coordination_metrics(0, 1)

        assert isinstance(metrics, CoordinationMetrics)
        assert 0 <= metrics.in_phase_pct <= 100

    def test_compute_work_metrics(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        work = analyzer.compute_work_metrics(0)

        assert work is not None
        assert "net_work" in work
        # Joint 0: sin(t), Torque ~ -sin(t). Power ~ -sin*cos ~ -sin(2t). Integrate over 2s (2 cycles) -> Net ~ 0
        assert np.isclose(
            work["net_work"], 0.0, atol=50.0
        )  # Loose tolerance for discrete integration

    def test_compute_recurrence_matrix_and_rqa(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        # Using small subset to speed up
        analyzer.joint_positions = analyzer.joint_positions[:50]
        analyzer.joint_velocities = analyzer.joint_velocities[:50]
        analyzer.times = analyzer.times[:50]

        rm = analyzer.compute_recurrence_matrix()
        assert rm.shape == (50, 50)

        rqa = analyzer.compute_rqa_metrics(rm)
        if rqa:
            assert isinstance(rqa, RQAMetrics)
            assert 0 <= rqa.recurrence_rate <= 1.0

    def test_estimate_lyapunov_exponent(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        # Sine wave is predictable, LLE should be low or 0?
        # Ideally chaos detection requires more data.
        data = sample_data["joint_positions"][:, 0]
        lle = analyzer.estimate_lyapunov_exponent(data, dim=2, tau=10, window=10)
        assert isinstance(lle, float)

    def test_compute_pca(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        pca = analyzer.compute_principal_component_analysis(n_components=2)

        assert isinstance(pca, PCAResult)
        assert pca.components.shape == (2, 3)  # 2 components, 3 joints
        assert pca.projected_data.shape == (200, 2)

    def test_compute_permutation_entropy(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        data = sample_data["joint_positions"][:, 0]
        pe = analyzer.compute_permutation_entropy(data, order=3)
        assert pe >= 0

    def test_compute_fractal_dimension(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        data = sample_data["joint_positions"][:, 0]
        # Sine wave dimension is 1.0
        fd = analyzer.compute_fractal_dimension(data)
        assert np.isclose(fd, 1.0, atol=0.2)  # Higuchi approx

    def test_compute_sample_entropy(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        data = sample_data["joint_positions"][:, 0]
        sampen = analyzer.compute_sample_entropy(data, m=2, r=0.2)
        assert sampen >= 0

    def test_compute_joint_stiffness(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        # Joint 0: Pos ~ sin, Torque ~ -sin.  T = -k * theta approx.
        # Plotting T vs Theta should be linear-ish with negative slope.
        stiff = analyzer.compute_joint_stiffness(0)
        assert isinstance(stiff, JointStiffnessMetrics)
        # Slope should be negative (Torque opposes motion/displacement)
        assert stiff.stiffness < 0

    def test_compute_dynamic_stiffness(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        times, k, r2 = analyzer.compute_dynamic_stiffness(0, window_size=20)
        assert len(times) == len(k)
        assert len(k) > 0

    def test_compute_rolling_correlation(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        times, corr = analyzer.compute_rolling_correlation(0, 0, window_size=20)
        # Auto-correlation should be 1.0
        assert np.allclose(corr, 1.0)

    def test_compute_local_divergence_rate(self, sample_data):
        analyzer = StatisticalAnalyzer(**sample_data)
        times, ldr = analyzer.compute_local_divergence_rate(0, window=10)
        assert len(times) == len(ldr)

    def test_export_statistics_csv(self, sample_data, tmp_path):
        analyzer = StatisticalAnalyzer(**sample_data)
        csv_path = tmp_path / "stats.csv"
        analyzer.export_statistics_csv(str(csv_path))
        assert csv_path.exists()
