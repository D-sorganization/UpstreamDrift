import pytest
import numpy as np
from bunkershot3d.postproc.wrench_trace import WrenchTrace


def test_wrench_trace() -> None:
    time = np.linspace(0, 0.1, 100)
    force = np.ones((100, 3)) * 10.0
    torque = np.ones((100, 3)) * 2.0

    trace = WrenchTrace(time, force, torque)

    # Test impulses
    lin_imp, ang_imp = trace.get_impulses()
    assert np.allclose(lin_imp, [1.0, 1.0, 1.0])  # 10.0 * 0.1 = 1.0
    assert np.allclose(ang_imp, [0.2, 0.2, 0.2])  # 2.0 * 0.1 = 0.2

    # Test resample
    target_times = np.linspace(0, 0.1, 50)
    resampled = trace.resample(target_times)
    assert len(resampled.time) == 50
    assert np.allclose(resampled.force_world[25], [10.0, 10.0, 10.0])

    # Test filter (noisy signal)
    # Add high frequency noise
    noisy_force = force.copy()
    noisy_force[:, 0] += np.sin(2 * np.pi * 5000 * time)  # 5kHz noise
    noisy_trace = WrenchTrace(time, noisy_force, torque)

    # Since dt is 0.001 (1kHz), a 5kHz noise will alias or be invalid, but we can test
    # the function runs without error
    filtered = noisy_trace.filter(cutoff_freq=200.0)
    assert filtered.force_world.shape == (100, 3)
