"""Unit tests for the configurable torque smoothing module (#3980)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "torque_smoothing.py"
)


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location("torque_smoothing", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["torque_smoothing"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoothing_module():
    return _load_module()


def _noisy_signal(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, 201)
    base = np.sin(2.0 * np.pi * 1.5 * time) + 0.5 * np.cos(2.0 * np.pi * 0.5 * time)
    noise = 0.4 * rng.standard_normal(time.size)
    return time, base, base + noise


def _high_freq_energy(values: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(values - values.mean()))
    # Energy in the upper half of the spectrum.
    return float(np.sum(spectrum[len(spectrum) // 2 :] ** 2))


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("moving_average", {"window": 11}),
        ("savitzky_golay", {"window": 11, "polyorder": 3}),
        ("lowpass", {"cutoff_hz": 5.0, "butter_order": 4}),
        ("spline", {"spline_s": 5.0}),
    ],
)
def test_each_method_reduces_high_frequency_noise(smoothing_module, method, kwargs):
    time, _, noisy = _noisy_signal()
    config = smoothing_module.SmoothingConfig(method=method, **kwargs)
    smoothed = smoothing_module.smooth_torque(time, noisy, config)

    assert smoothed.shape == noisy.shape
    assert np.all(np.isfinite(smoothed))
    assert _high_freq_energy(smoothed) < _high_freq_energy(noisy)


def test_polynomial_residual_diagnostic_flag(smoothing_module):
    time = np.linspace(0.0, 1.0, 101)
    smoothed = np.sin(2.0 * np.pi * 4.0 * time)  # high-frequency content
    # Fit a sixth-order polynomial — it will not capture 4 Hz content.
    coeffs = np.polyfit(time, smoothed, 6)
    diagnostic = smoothing_module.polynomial_residual_diagnostic(
        time, smoothed, coeffs, threshold=0.1
    )
    assert diagnostic["exceeds_threshold"] is True
    assert diagnostic["max_abs_residual"] > 0.1
    assert diagnostic["rmse"] > 0.0

    # A low-degree polynomial fitting itself should NOT exceed.
    smooth_target = 1.0 + 2.0 * time
    coeffs_linear = np.polyfit(time, smooth_target, 6)
    ok = smoothing_module.polynomial_residual_diagnostic(
        time, smooth_target, coeffs_linear, threshold=0.01
    )
    assert ok["exceeds_threshold"] is False


def test_savitzky_golay_window_must_be_odd(smoothing_module):
    time, _, noisy = _noisy_signal()
    config = smoothing_module.SmoothingConfig(
        method="savitzky_golay", window=10, polyorder=3
    )
    with pytest.raises(ValueError, match="odd"):
        smoothing_module.smooth_torque(time, noisy, config)


def test_rejects_non_finite_input(smoothing_module):
    time = np.linspace(0.0, 1.0, 50)
    bad = np.linspace(0.0, 1.0, 50)
    bad[10] = np.nan
    with pytest.raises(ValueError):
        smoothing_module.smooth_torque(
            time,
            bad,
            smoothing_module.SmoothingConfig(method="moving_average", window=5),
        )


def test_torque_smoothing_unknown_method_raises(smoothing_module):
    time = np.linspace(0.0, 1.0, 10)
    values = np.zeros_like(time)
    config = smoothing_module.SmoothingConfig.__new__(smoothing_module.SmoothingConfig)
    object.__setattr__(config, "method", "bogus")
    object.__setattr__(config, "window", 5)
    object.__setattr__(config, "polyorder", 3)
    object.__setattr__(config, "cutoff_hz", 25.0)
    object.__setattr__(config, "butter_order", 4)
    object.__setattr__(config, "spline_s", None)
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        smoothing_module.smooth_torque(time, values, config)
