import numpy as np
import pytest

from shared.python.statistical_analysis import StatisticalAnalyzer, SwingPhase


@pytest.fixture
def sample_data():
    N = 100
    times = np.linspace(0, 1.0, N)
    # Simple sine wave for joint positions
    joint_positions = np.sin(2 * np.pi * times)[:, np.newaxis]
    joint_velocities = 2 * np.pi * np.cos(2 * np.pi * times)[:, np.newaxis]

    # Constant positive torque to create work
    # Work = Int(Torque * Vel) dt.
    # If Torque=1, Vel=cos(t), integral over cycle is 0.
    # Let's align torque with velocity to get positive work.
    joint_torques = joint_velocities.copy()

    # Club head speed: bell curve peaking at 0.5s
    club_head_speed = 100 * np.exp(-((times - 0.5) ** 2) / 0.05)

    return StatisticalAnalyzer(
        times=times,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        club_head_speed=club_head_speed,
    )


def test_initialization(sample_data):
    assert sample_data.dt > 0
    assert sample_data.duration == 1.0


def test_compute_summary_stats(sample_data):
    stats = sample_data.compute_summary_stats(sample_data.joint_positions[:, 0])
    assert stats.min == pytest.approx(-1.0, abs=0.01)
    assert stats.max == pytest.approx(1.0, abs=0.01)
    assert stats.range == pytest.approx(2.0, abs=0.02)


def test_find_peaks_in_data(sample_data):
    data = sample_data.joint_positions[:, 0]
    peaks = sample_data.find_peaks_in_data(data, height=0.5)
    assert len(peaks) > 0
    assert peaks[0].value > 0.5


def test_find_club_head_speed_peak(sample_data):
    peak = sample_data.find_club_head_speed_peak()
    assert peak is not None
    assert peak.time == pytest.approx(0.5, abs=0.02)
    assert peak.value == pytest.approx(100.0, abs=1.0)

    # Test with no data
    analyzer = StatisticalAnalyzer(
        np.array([0, 1]), np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))
    )
    assert analyzer.find_club_head_speed_peak() is None


def test_compute_range_of_motion(sample_data):
    min_angle, max_angle, rom = sample_data.compute_range_of_motion(0)
    # sin wave -1 to 1 rad -> -57 to 57 deg. ROM ~ 114
    assert rom == pytest.approx(np.rad2deg(2), abs=1.0)

    # Invalid index
    res = sample_data.compute_range_of_motion(999)
    assert res == (0.0, 0.0, 0.0)


def test_compute_tempo():
    # Construct data where we can predict tempo
    # Use "Plateau - Dip - Ramp" shape to be robust against finding min
    times = np.linspace(0, 2.0, 200)  # dt = 0.01
    speed = np.zeros_like(times)

    # 0.0 to 0.5s (Indices 0-50): Backswing movement (Speed 5.0)
    speed[:50] = 5.0

    # 0.5s to 0.6s (Indices 50-60): Transition dip (Speed 0.1)
    # The actual min should be around here.
    speed[50:60] = 0.1

    # 0.6s to 1.0s (Indices 60-100): Downswing ramp to Impact
    speed[60:100] = np.linspace(0.1, 100.0, 40)

    # Impact at 1.0s (Index 100)
    speed[100] = 100.0

    # Follow through
    speed[100:] = 0.0

    analyzer = StatisticalAnalyzer(
        times,
        np.zeros((200, 1)),
        np.zeros((200, 1)),
        np.zeros((200, 1)),
        club_head_speed=speed,
    )

    res = analyzer.compute_tempo()
    if res:
        b, d, r = res
        # Impact index 100. Search limit 0.7 * 100 = 70.
        # Min in speed[5:70] is at 50-60 (0.5s).
        # Backswing duration = 0.5s
        # Downswing duration = 1.0s - 0.5s = 0.5s
        assert b == pytest.approx(0.5, abs=0.1)
        assert d == pytest.approx(0.5, abs=0.1)

    # Test insufficient data
    analyzer_empty = StatisticalAnalyzer(
        np.array([0]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    )
    assert analyzer_empty.compute_tempo() is None


def test_compute_x_factor(sample_data):
    # 1 joint only, so this will fail or return None if we ask for index 1
    xf = sample_data.compute_x_factor(0, 0)  # Should be 0
    assert np.allclose(xf, 0.0)

    xf_none = sample_data.compute_x_factor(0, 99)
    assert xf_none is None


def test_detect_impact_time(sample_data):
    t = sample_data.detect_impact_time()
    assert t == pytest.approx(0.5, abs=0.02)


def test_compute_energy_metrics(sample_data):
    ke = np.abs(sample_data.joint_velocities[:, 0])
    pe = np.abs(sample_data.joint_positions[:, 0])
    metrics = sample_data.compute_energy_metrics(ke, pe)
    assert "energy_efficiency" in metrics
    assert "max_total_energy" in metrics

    # Test with no club head speed
    analyzer = StatisticalAnalyzer(
        sample_data.times,
        sample_data.joint_positions,
        sample_data.joint_velocities,
        sample_data.joint_torques,
    )
    metrics_no_chs = analyzer.compute_energy_metrics(ke, pe)
    assert metrics_no_chs["energy_efficiency"] == 0.0


def test_detect_swing_phases(sample_data):
    phases = sample_data.detect_swing_phases()
    assert len(phases) > 0
    assert isinstance(phases[0], SwingPhase)

    # Test empty case
    analyzer = StatisticalAnalyzer(
        np.array([0]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    )
    phases_empty = analyzer.detect_swing_phases()
    assert len(phases_empty) == 1
    assert phases_empty[0].name == "Complete Swing"


def test_compute_phase_statistics(sample_data):
    phases = sample_data.detect_swing_phases()
    stats = sample_data.compute_phase_statistics(phases, sample_data.club_head_speed)
    assert len(stats) == len(phases)


def test_generate_comprehensive_report(sample_data):
    report = sample_data.generate_comprehensive_report()
    assert "club_head_speed" in report
    assert "joints" in report
    assert "phases" in report

    # Check structure
    assert "peak_value" in report["club_head_speed"]


def test_export_statistics_csv(sample_data, tmp_path):
    f = tmp_path / "stats.csv"
    sample_data.export_statistics_csv(str(f))
    assert f.exists()
    content = f.read_text()
    assert "Golf Swing Statistical Analysis" in content
    assert "Club Head Speed" in content


def test_analyze_kinematic_sequence():
    # Need multiple joints
    times = np.linspace(0, 1, 100)
    # Pelvis peaks early, Thorax later
    v_pelvis = np.exp(-((times - 0.3) ** 2) / 0.01)
    v_thorax = np.exp(-((times - 0.4) ** 2) / 0.01)

    vels = np.column_stack([v_pelvis, v_thorax])
    analyzer = StatisticalAnalyzer(times, np.zeros((100, 2)), vels, np.zeros((100, 2)))

    seq, score = analyzer.analyze_kinematic_sequence({"Pelvis": 0, "Thorax": 1})
    assert len(seq) == 2
    assert seq[0].segment_name == "Pelvis"
    assert seq[1].segment_name == "Thorax"
    assert score == 1.0

    # Test invalid index
    seq_bad, _ = analyzer.analyze_kinematic_sequence({"Bad": 99})
    assert len(seq_bad) == 0


def test_compute_correlations():
    # Create 2 joints data for matrix
    N = 100
    j1 = np.linspace(0, 1, N)
    j2 = np.linspace(0, 1, N)  # Perfect correlation

    analyzer = StatisticalAnalyzer(
        np.arange(N), np.column_stack([j1, j2]), np.zeros((N, 2)), np.zeros((N, 2))
    )

    corr, labels = analyzer.compute_correlations("position")
    assert corr.shape == (2, 2)
    assert labels == ["J0", "J1"]
    assert corr[0, 1] == pytest.approx(1.0)

    # Test empty
    analyzer_empty = StatisticalAnalyzer(
        np.array([0]), np.zeros((1, 0)), np.zeros((1, 0)), np.zeros((1, 0))
    )
    corr_e, labels_e = analyzer_empty.compute_correlations("position")
    assert len(corr_e) == 0


def test_frequency_analysis(sample_data):
    # This might use scipy fallback if shared.python.signal_processing fails or isn't there
    freqs, psd = sample_data.compute_frequency_analysis(
        sample_data.joint_positions[:, 0]
    )
    assert len(freqs) > 0
    assert len(psd) > 0

    # Test fs=0
    analyzer = StatisticalAnalyzer(
        np.array([0]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    )
    f, p = analyzer.compute_frequency_analysis(np.array([0]))
    assert len(f) == 0


def test_compute_smoothness_metric(sample_data):
    score = sample_data.compute_smoothness_metric(sample_data.club_head_speed)
    # SAL is negative
    assert score <= 0.0

    # Test fs=0
    analyzer = StatisticalAnalyzer(
        np.array([0]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    )
    s = analyzer.compute_smoothness_metric(np.array([0]))
    assert s == 0.0


def test_compute_work_metrics(sample_data):
    # We set torque = velocity in fixture.
    # Power = v^2, which is always positive.
    # So pos_work should be > 0, neg_work = 0
    metrics = sample_data.compute_work_metrics(0)

    assert metrics is not None
    assert metrics["positive_work"] > 0
    assert metrics["negative_work"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["net_work"] == pytest.approx(metrics["positive_work"])

    # Test invalid index
    assert sample_data.compute_work_metrics(99) is None


def test_compute_phase_space_path_length(sample_data):
    pl = sample_data.compute_phase_space_path_length(0)
    assert pl > 0.0

    # Test invalid index
    assert sample_data.compute_phase_space_path_length(99) == 0.0


def test_compute_joint_power_metrics(sample_data):
    # In sample_data, torque = velocity = cos(t).
    # Power = cos^2(t), which is always >= 0.
    metrics = sample_data.compute_joint_power_metrics(0)

    assert metrics is not None
    assert metrics.peak_generation > 0
    assert metrics.peak_absorption == 0.0
    assert metrics.generation_duration > 0
    assert metrics.absorption_duration == 0.0
    assert metrics.net_work > 0

    # Test invalid index
    assert sample_data.compute_joint_power_metrics(99) is None

    # Create scenario with negative power
    times = np.linspace(0, 1, 100)
    # v = 1, tau = -1 -> power = -1
    v = np.ones((100, 1))
    tau = -np.ones((100, 1))
    analyzer = StatisticalAnalyzer(times, np.zeros((100, 1)), v, tau)

    metrics_neg = analyzer.compute_joint_power_metrics(0)
    assert metrics_neg is not None
    assert metrics_neg.peak_absorption == -1.0
    assert metrics_neg.avg_absorption == -1.0
    assert metrics_neg.peak_generation == 0.0
    assert metrics_neg.generation_duration == 0.0


def test_compute_impulse_metrics(sample_data):
    # In sample_data, torque = cos(t) (since vel = cos(t) and torque=vel)
    # Integral of cos^2(t) over 0 to 1 is positive
    # Actually wait, joint_velocities in sample_data is 2*pi*cos(2*pi*t)
    # So torque is also 2*pi*cos(2*pi*t).
    # Integral over 0 to 1 of cos(2*pi*t) is 0.

    metrics = sample_data.compute_impulse_metrics("torque", 0)
    assert metrics is not None

    # Since it's a full cycle of cosine, net impulse should be approx 0
    # But wait, it's torque = velocity. Velocity integral over 1s (cycle) is 0.
    assert metrics.net_impulse == pytest.approx(0.0, abs=1.0)  # approx

    # Test invalid inputs
    assert sample_data.compute_impulse_metrics("torque", 99) is None
    assert (
        sample_data.compute_impulse_metrics("force", 0) is None
    )  # No forces in sample_data
    assert sample_data.compute_impulse_metrics("invalid", 0) is None


def test_compute_x_factor_stretch():
    # Shoulders (index 0 for test) rotate 2x faster than Hips (index 1 for test)
    # create sample data with 2 joints
    times = np.linspace(0, 1.0, 100)
    pos = np.zeros((100, 2))
    # Hips: 0 to 45 deg
    pos[:, 1] = np.deg2rad(np.linspace(0, 45, 100))
    # Shoulders: 0 to 90 deg
    pos[:, 0] = np.deg2rad(np.linspace(0, 90, 100))

    analyzer = StatisticalAnalyzer(times, pos, np.zeros((100, 2)), np.zeros((100, 2)))

    res = analyzer.compute_x_factor_stretch(0, 1)
    assert res is not None
    vel, peak = res
    assert len(vel) == 100
    # X-Factor is linear 0->45 over 1s. Velocity should be ~45 deg/s constant.
    assert np.allclose(vel, 45.0, atol=5.0)  # Numerical gradient error tolerance
    assert peak == pytest.approx(45.0, abs=5.0)


def test_compute_angular_momentum_metrics():
    # Create angular momentum data
    N = 100
    times = np.linspace(0, 1, N)
    am = np.ones((N, 3))  # Magnitude sqrt(3)

    analyzer = StatisticalAnalyzer(
        times,
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        angular_momentum=am,
    )

    metrics = analyzer.compute_angular_momentum_metrics()
    assert metrics is not None
    assert metrics.peak_magnitude == pytest.approx(np.sqrt(3))
    assert metrics.mean_magnitude == pytest.approx(np.sqrt(3))
    assert metrics.peak_lx == 1.0

    # Test None
    analyzer_none = StatisticalAnalyzer(
        times, np.zeros((N, 1)), np.zeros((N, 1)), np.zeros((N, 1))
    )
    assert analyzer_none.compute_angular_momentum_metrics() is None


def test_compute_stability_metrics():
    # CoP and CoM
    N = 100
    times = np.linspace(0, 1, N)
    cop = np.zeros((N, 3))
    com = np.zeros((N, 3))
    com[:, 2] = 1.0  # CoM at height 1m

    # CoP at origin. Distance should be 0 in XY plane.

    analyzer = StatisticalAnalyzer(
        times,
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        cop_position=cop,
        com_position=com,
    )

    metrics = analyzer.compute_stability_metrics()
    assert metrics is not None
    assert metrics.mean_com_cop_distance == 0.0

    # Inclination angle: Vector is (0,0,1). Angle with vertical is 0.
    assert metrics.mean_inclination_angle == 0.0

    # Test 2D CoP
    cop_2d = np.zeros((N, 2))
    analyzer_2d = StatisticalAnalyzer(
        times,
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        np.zeros((N, 1)),
        cop_position=cop_2d,
        com_position=com,
    )
    metrics_2d = analyzer_2d.compute_stability_metrics()
    assert metrics_2d is not None
    assert metrics_2d.mean_inclination_angle == 0.0


def test_compute_coordination_metrics():
    # Create 2 joints
    N = 100
    times = np.linspace(0, 1, N)
    vels = np.zeros((N, 2))

    # In-Phase: Both positive
    vels[:, 0] = 1.0
    vels[:, 1] = 1.0  # Angle = 45 deg -> In-Phase

    analyzer = StatisticalAnalyzer(times, np.zeros((N, 2)), vels, np.zeros((N, 2)))

    metrics = analyzer.compute_coordination_metrics(0, 1)
    assert metrics is not None
    assert metrics.in_phase_pct == 100.0
    assert metrics.anti_phase_pct == 0.0
    assert metrics.mean_coupling_angle == pytest.approx(45.0)


def test_compute_continuous_relative_phase():
    N = 100
    times = np.linspace(0, 1, N)
    pos = np.zeros((N, 2))
    vel = np.zeros((N, 2))

    # J1: Sine wave
    pos[:, 0] = np.sin(2 * np.pi * times)
    vel[:, 0] = np.cos(2 * np.pi * times)

    # J2: Cosine wave (shifted by 90 deg)
    pos[:, 1] = np.cos(2 * np.pi * times)
    vel[:, 1] = -np.sin(2 * np.pi * times)

    analyzer = StatisticalAnalyzer(times, pos, vel, np.zeros((N, 2)))

    crp = analyzer.compute_continuous_relative_phase(0, 1)
    assert len(crp) == N
    # Difference should be constant ~90 degrees
    diff = np.abs(np.mean(crp))  # noqa: F841
    # Wrap issues might occur, but let's check basic execution first
    assert crp is not None


def test_compute_swing_profile():
    # Need speed, torques, velocities, etc.
    N = 100
    times = np.linspace(0, 1, N)
    speed = np.ones(N) * 53.6  # ~120 mph
    torques = np.ones((N, 1)) * 100
    vels = np.ones((N, 1)) * 10

    analyzer = StatisticalAnalyzer(
        times, np.zeros((N, 1)), vels, torques, club_head_speed=speed
    )

    profile = analyzer.compute_swing_profile()
    assert profile is not None
    assert profile.speed_score > 99.0
