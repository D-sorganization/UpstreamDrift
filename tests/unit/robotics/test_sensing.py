"""Unit tests for sensing module.

Tests cover:
    - Noise models (Gaussian, Brownian, Quantization)
    - Force/torque sensor simulation
    - IMU sensor simulation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.robotics.sensing.force_torque_sensor import (
    ForceTorqueSensor,
    ForceTorqueSensorConfig,
    create_ideal_sensor,
    create_realistic_sensor,
)
from src.robotics.sensing.imu_sensor import (
    IMUSensor,
    IMUSensorConfig,
    create_ideal_imu,
    create_realistic_imu,
)
from src.robotics.sensing.noise_models import (
    BandwidthLimitedNoise,
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
    QuantizationNoise,
    create_realistic_sensor_noise,
)


class TestNoiseModels:
    """Tests for noise model classes."""

    def test_gaussian_noise_shape_preserved(self) -> None:
        """Test Gaussian noise preserves input shape."""
        noise = GaussianNoise(std=0.1, seed=42)
        signal = np.array([1.0, 2.0, 3.0])

        noisy = noise.apply(signal)

        assert noisy.shape == signal.shape

    def test_gaussian_noise_statistics(self) -> None:
        """Test Gaussian noise has correct statistics."""
        noise = GaussianNoise(std=1.0, mean=0.0, seed=42)

        # Apply to many samples
        samples = []
        for _ in range(10000):
            noisy = noise.apply(np.array([0.0]))
            samples.append(noisy[0])

        samples = np.array(samples)

        # Check mean and std are approximately correct
        assert abs(np.mean(samples)) < 0.1  # Mean close to 0
        assert abs(np.std(samples) - 1.0) < 0.1  # Std close to 1

    def test_gaussian_noise_reproducibility(self) -> None:
        """Test Gaussian noise is reproducible with seed."""
        noise1 = GaussianNoise(std=0.1, seed=42)
        noise2 = GaussianNoise(std=0.1, seed=42)

        signal = np.array([1.0, 2.0, 3.0])

        noisy1 = noise1.apply(signal)
        noisy2 = noise2.apply(signal)

        assert_allclose(noisy1, noisy2)

    def test_brownian_noise_drift(self) -> None:
        """Test Brownian noise accumulates drift."""
        noise = BrownianNoise(drift_rate=0.1, seed=42)

        # Apply many times
        signal = np.array([0.0])
        for _ in range(100):
            noise.apply(signal)

        # Bias should have drifted
        assert noise.current_bias != 0.0

    def test_brownian_noise_max_bias(self) -> None:
        """Test Brownian noise respects max bias."""
        noise = BrownianNoise(drift_rate=1.0, max_bias=0.5, seed=42)

        signal = np.array([0.0])
        for _ in range(1000):
            noise.apply(signal)

        assert abs(noise.current_bias) <= 0.5

    def test_brownian_noise_reset(self) -> None:
        """Test Brownian noise reset."""
        noise = BrownianNoise(drift_rate=0.1, initial_bias=0.5, seed=42)

        noise.apply(np.array([0.0]))
        noise.reset()

        assert noise.current_bias == 0.5

    def test_quantization_noise(self) -> None:
        """Test quantization noise discretizes signal."""
        noise = QuantizationNoise(resolution=0.1)

        # Use values that don't fall on rounding boundaries
        signal = np.array([0.04, 0.16, 0.27])
        quantized = noise.apply(signal)

        # Values should be multiples of resolution (nearest)
        expected = np.array([0.0, 0.2, 0.3])
        assert_allclose(quantized, expected)

    def test_bandwidth_limited_noise(self) -> None:
        """Test bandwidth filter smooths signal."""
        noise = BandwidthLimitedNoise(
            cutoff_frequency=10.0,
            sample_rate=100.0,
        )

        # Step input
        outputs = []
        for i in range(50):
            signal = np.array([1.0]) if i > 0 else np.array([0.0])
            filtered = noise.apply(signal)
            outputs.append(filtered[0])

        outputs = np.array(outputs)

        # Should have smooth rise, not instant step
        assert outputs[1] < 1.0  # Not instant
        assert outputs[-1] > 0.9  # Eventually reaches target

    def test_bandwidth_limited_noise_order2_slower_than_order1(self) -> None:
        """Higher-order filter has steeper roll-off / slower step response."""
        order1 = BandwidthLimitedNoise(
            cutoff_frequency=10.0, sample_rate=100.0, order=1
        )
        order2 = BandwidthLimitedNoise(
            cutoff_frequency=10.0, sample_rate=100.0, order=2
        )

        outputs1: list[float] = []
        outputs2: list[float] = []
        for i in range(50):
            signal = np.array([1.0]) if i > 0 else np.array([0.0])
            outputs1.append(order1.apply(signal)[0])
            outputs2.append(order2.apply(signal)[0])

        arr1 = np.array(outputs1)
        arr2 = np.array(outputs2)

        # 2nd-order filter should respond more slowly to a step than 1st-order
        # at early samples, the 2nd-order output must lag behind the 1st-order
        assert (
            arr2[3] < arr1[3]
        ), "order=2 filter should be slower than order=1 at early samples"
        # Both should eventually converge toward 1.0
        assert arr2[-1] > 0.8

    def test_composite_noise(self) -> None:
        """Test composite noise applies all models."""
        composite = CompositeNoise(
            models=[
                GaussianNoise(std=0.1, seed=42),
                QuantizationNoise(resolution=0.01),
            ]
        )

        signal = np.array([1.0])
        noisy = composite.apply(signal)

        # Should be different from original
        assert noisy[0] != signal[0]

    def test_create_realistic_sensor_noise(self) -> None:
        """Test factory function creates valid composite."""
        noise = create_realistic_sensor_noise(
            noise_std=0.1,
            bias_drift_rate=0.01,
            quantization_bits=12,
            signal_range=10.0,
            seed=42,
        )

        signal = np.array([5.0])
        noisy = noise.apply(signal)

        assert noisy.shape == signal.shape


class TestForceTorqueSensor:
    """Tests for ForceTorqueSensor class."""

    def test_create_sensor(self) -> None:
        """Test creating a force/torque sensor."""
        config = ForceTorqueSensorConfig(
            sensor_id="test_ft",
            force_noise_std=0.1,
            torque_noise_std=0.01,
        )
        sensor = ForceTorqueSensor(config)

        assert sensor.sensor_id == "test_ft"
        assert sensor.config == config

    def test_ideal_sensor_no_noise(self) -> None:
        """Test ideal sensor adds no noise."""
        sensor = create_ideal_sensor()

        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        reading = sensor.read(true_wrench, timestamp=0.001)

        # Should be very close to true value
        assert_allclose(reading.wrench, true_wrench, atol=1e-10)

    def test_sensor_reading_shape(self) -> None:
        """Test sensor reading has correct shape."""
        sensor = ForceTorqueSensor()

        true_wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        reading = sensor.read(true_wrench, timestamp=0.001)

        assert reading.wrench.shape == (6,)
        assert reading.force.shape == (3,)
        assert reading.torque.shape == (3,)

    def test_sensor_invalid_wrench_raises(self) -> None:
        """Test sensor raises for invalid wrench shape."""
        sensor = ForceTorqueSensor()

        with pytest.raises(ValueError, match="must be 6D"):
            sensor.read(np.array([1.0, 2.0, 3.0]))

    def test_sensor_tare(self) -> None:
        """Test sensor tare functionality."""
        sensor = create_ideal_sensor()

        # First reading
        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        sensor.read(true_wrench, timestamp=0.001)

        # Tare at current reading
        sensor.tare()

        # Next reading should be offset
        reading = sensor.read(true_wrench, timestamp=0.002)
        assert_allclose(reading.wrench, np.zeros(6), atol=1e-10)

    def test_sensor_reset(self) -> None:
        """Test sensor reset clears state."""
        sensor = ForceTorqueSensor()

        sensor.read(np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0]))
        sensor.tare(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        sensor.reset()

        # After reset, tare should be zero
        reading = sensor.read(np.zeros(6))
        # Should be near zero (only noise)
        assert np.allclose(reading.wrench, 0.0, atol=1.0)  # Allow for noise

    def test_sensor_clipping(self) -> None:
        """Test sensor clips to range."""
        config = ForceTorqueSensorConfig(
            force_range=100.0,
            torque_range=10.0,
            force_noise_std=0.0,
            torque_noise_std=0.0,
        )
        sensor = ForceTorqueSensor(config)

        # Wrench exceeding range
        true_wrench = np.array([200.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        reading = sensor.read(true_wrench)

        assert abs(reading.wrench[0]) <= 100.0
        assert abs(reading.wrench[3]) <= 10.0

    def test_realistic_sensor_adds_noise(self) -> None:
        """Test realistic sensor adds noise."""
        sensor = create_realistic_sensor(quality="industrial", seed=42)

        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        reading = sensor.read(true_wrench)

        # Should be different from true value
        assert not np.allclose(reading.wrench, true_wrench)

    def test_contact_location_estimation(self) -> None:
        """Test contact location estimation."""
        sensor = ForceTorqueSensor()

        # Force at known location
        # If force is [0, 0, 10] at position [1, 0, 0]
        # Torque should be r x f = [1,0,0] x [0,0,10] = [0, -10, 0]
        wrench = np.array([0.0, 0.0, 10.0, 0.0, -10.0, 0.0])

        location = sensor.estimate_contact_location(wrench)

        assert location is not None
        # Should estimate x ≈ 1
        assert abs(location[0] - 1.0) < 0.1


class TestIMUSensor:
    """Tests for IMUSensor class."""

    def test_create_imu(self) -> None:
        """Test creating an IMU sensor."""
        config = IMUSensorConfig(
            sensor_id="test_imu",
            accel_noise_std=0.01,
            gyro_noise_std=0.001,
        )
        imu = IMUSensor(config)

        assert imu.sensor_id == "test_imu"
        assert imu.config == config

    def test_ideal_imu_no_noise(self) -> None:
        """Test ideal IMU adds no noise."""
        imu = create_ideal_imu()

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])

        reading = imu.read(accel, gyro, timestamp=0.001)

        assert_allclose(reading.linear_acceleration, accel, atol=1e-10)
        assert_allclose(reading.angular_velocity, gyro, atol=1e-10)

    def test_imu_reading_shape(self) -> None:
        """Test IMU reading has correct shape."""
        imu = IMUSensor()

        reading = imu.read(
            linear_accel=np.array([0.0, 0.0, 9.81]),
            angular_vel=np.array([0.0, 0.0, 0.1]),
        )

        assert reading.linear_acceleration.shape == (3,)
        assert reading.angular_velocity.shape == (3,)

    def test_imu_invalid_input_raises(self) -> None:
        """Test IMU raises for invalid input shape."""
        imu = IMUSensor()

        with pytest.raises(ValueError, match="must be"):
            imu.read(
                linear_accel=np.array([0.0, 0.0]),  # Wrong shape
                angular_vel=np.array([0.0, 0.0, 0.0]),
            )

    def test_imu_orientation_integration(self) -> None:
        """Test IMU integrates orientation from gyro."""
        imu = create_ideal_imu()

        # Rotate around z-axis at 1 rad/s for 1 second
        dt = 0.01
        angular_vel = np.array([0.0, 0.0, 1.0])

        for i in range(100):
            imu.read(
                linear_accel=np.zeros(3),
                angular_vel=angular_vel,
                timestamp=i * dt,
            )

        # Should have rotated ~1 radian around z
        orientation = imu.orientation

        # Convert quaternion to angle
        # For rotation around z: q = [cos(theta/2), 0, 0, sin(theta/2)]
        angle = 2 * np.arctan2(orientation[3], orientation[0])

        assert abs(angle - 1.0) < 0.1  # Approximately 1 radian

    def test_imu_reset(self) -> None:
        """Test IMU reset clears state."""
        imu = IMUSensor()

        # Do some readings
        imu.read(np.zeros(3), np.array([0, 0, 1]), timestamp=0.0)
        imu.read(np.zeros(3), np.array([0, 0, 1]), timestamp=0.1)

        imu.reset()

        # Orientation should be identity
        assert_allclose(imu.orientation, [1, 0, 0, 0])

    def test_imu_set_orientation(self) -> None:
        """Test IMU set_orientation."""
        imu = IMUSensor()

        # Set to 90 degree rotation around z
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        imu.set_orientation(q)

        assert_allclose(imu.orientation, q)

    def test_imu_clipping(self) -> None:
        """Test IMU clips to range."""
        config = IMUSensorConfig(
            accel_range=10.0,
            gyro_range=1.0,
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
        )
        imu = IMUSensor(config)

        reading = imu.read(
            linear_accel=np.array([100.0, 0.0, 0.0]),
            angular_vel=np.array([10.0, 0.0, 0.0]),
        )

        assert abs(reading.linear_acceleration[0]) <= 10.0
        assert abs(reading.angular_velocity[0]) <= 1.0

    def test_realistic_imu_adds_noise(self) -> None:
        """Test realistic IMU adds noise."""
        imu = create_realistic_imu(quality="mems", seed=42)

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        reading = imu.read(accel, gyro)

        # Should be different from true value
        assert not np.allclose(reading.linear_acceleration, accel)

    def test_gravity_in_sensor_frame(self) -> None:
        """Test gravity vector in sensor frame."""
        imu = IMUSensor()

        # At identity orientation, gravity should be in -z
        gravity = imu.get_gravity_in_sensor_frame()
        assert_allclose(gravity, [0, 0, -9.81], atol=1e-10)

        # After 90 degree rotation around y, gravity should be in -x
        q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        imu.set_orientation(q)

        gravity = imu.get_gravity_in_sensor_frame()
        assert abs(gravity[0] - 9.81) < 0.1  # ~9.81 in x direction


class TestSensorFactories:
    """Tests for sensor factory functions."""

    def test_create_ideal_ft_sensor(self) -> None:
        """Test ideal F/T sensor factory."""
        sensor = create_ideal_sensor("my_sensor")
        assert sensor.sensor_id == "my_sensor"
        assert sensor.config.force_noise_std == 0.0

    def test_create_realistic_ft_sensor_qualities(self) -> None:
        """Test realistic F/T sensor at different qualities."""
        for quality in ["research", "industrial", "consumer"]:
            sensor = create_realistic_sensor(quality=quality)
            assert sensor.config.force_noise_std > 0

    def test_create_ideal_imu(self) -> None:
        """Test ideal IMU factory."""
        imu = create_ideal_imu("my_imu")
        assert imu.sensor_id == "my_imu"
        assert imu.config.accel_noise_std == 0.0

    def test_create_realistic_imu_qualities(self) -> None:
        """Test realistic IMU at different qualities."""
        for quality in ["mems", "industrial", "tactical"]:
            imu = create_realistic_imu(quality=quality)
            assert imu.config.accel_noise_std > 0


class TestNoisySensor:
    """Tests for SensorNoiseParameters / NoisySensor wrapper."""

    def test_zero_noise_returns_raw(self) -> None:
        """With all stds set to 0, measurement equals raw signal."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=0)
        raw = np.array([1.0, -2.5, 3.14])
        out = sensor.measure(raw)
        assert_allclose(out, raw)

    def test_nonzero_noise_differs_from_raw(self) -> None:
        """Non-zero parameters produce deviation from raw with expected scale."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.1,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=42)
        raw = np.zeros(5000)
        out = sensor.measure(raw)
        # Roughly mean 0, std ~0.1
        assert abs(float(np.mean(out))) < 0.01
        assert 0.08 < float(np.std(out)) < 0.12

    def test_bias_walk_drift_accumulates(self) -> None:
        """Bias random walk std should grow as sqrt(n_steps)."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.01,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=7)
        raw = np.array([0.0])
        # Run 1000 steps, collect final bias magnitudes across many trials
        finals = []
        for trial in range(200):
            sensor = NoisySensor(params, seed=1000 + trial)
            for _ in range(1000):
                sensor.measure(raw)
            finals.append(float(np.asarray(sensor.current_bias).item()))
        # sqrt(1000) * 0.01 ~ 0.316
        final_std = float(np.std(finals))
        assert 0.2 < final_std < 0.45

    def test_scale_factor_multiplicative(self) -> None:
        """Scale factor error scales with raw signal magnitude."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.01,
        )
        sensor = NoisySensor(params, seed=3)
        # First call materializes scale factor (shape = (1,))
        sensor.measure(np.array([1.0]))
        sf = float(np.asarray(sensor.scale_factor).item())
        out_small = sensor.measure(np.array([1.0]))
        out_large = sensor.measure(np.array([100.0]))
        assert_allclose(out_small, np.array([1.0 + sf]))
        assert_allclose(out_large, np.array([100.0 * (1.0 + sf)]))

    def test_saturation_clips_output(self) -> None:
        """Saturation limit clips measurement magnitude."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
            saturation_limit=5.0,
        )
        sensor = NoisySensor(params, seed=0)
        out = sensor.measure(np.array([10.0, -10.0, 3.0]))
        assert_allclose(out, np.array([5.0, -5.0, 3.0]))

    def test_temperature_drift(self) -> None:
        """Temperature delta shifts the measurement via temp coefficient."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
            temperature_coefficient=0.1,
        )
        sensor = NoisySensor(params, seed=0)
        baseline = sensor.measure(np.array([1.0]), temperature_delta=0.0)
        hot = sensor.measure(np.array([1.0]), temperature_delta=10.0)
        assert_allclose(baseline, np.array([1.0]))
        assert_allclose(hot, np.array([1.0 + 1.0]))  # 0.1 * 10

    def test_reset_restores_state(self) -> None:
        """reset() returns the sensor to a deterministic fresh state."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.01,
            bias_initial_std=0.01,
            bias_walk_std=1e-3,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=99)
        raw = np.array([1.0, 2.0])
        first = sensor.measure(raw.copy())
        sensor.reset()
        again = sensor.measure(raw.copy())
        assert_allclose(first, again)

    def test_parameters_validate_non_negative(self) -> None:
        """SensorNoiseParameters rejects negative stds."""
        from src.robotics.sensing.noise_models import SensorNoiseParameters

        with pytest.raises(ValueError):
            SensorNoiseParameters(white_std=-0.1)
        with pytest.raises(ValueError):
            SensorNoiseParameters(saturation_limit=0.0)

    def test_presets_are_usable(self) -> None:
        """Published-datasheet presets construct and run without error."""
        from src.robotics.sensing.noise_models import (
            FORCE_TORQUE_INDUSTRIAL_DEFAULTS,
            IMU_MEMS_DEFAULTS,
            NoisySensor,
        )

        imu = NoisySensor(IMU_MEMS_DEFAULTS, seed=0)
        ft = NoisySensor(FORCE_TORQUE_INDUSTRIAL_DEFAULTS, seed=0)
        assert imu.measure(np.zeros(3)).shape == (3,)
        assert ft.measure(np.zeros(6)).shape == (6,)
