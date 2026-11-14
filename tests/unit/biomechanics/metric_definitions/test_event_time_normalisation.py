"""Tests for event-relative time normalisation (101-point resampling) (#14427)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.metric_definitions import (
    EventTimeNormalisationDescriptor,
    NormalisationKind,
    TimeNormalisedResult,
    resample_event_relative_101,
)
from src.shared.python.biomechanics.metric_definitions.schema import (
    FilterDifferentiationDescriptor,
    GolfMetricDefinitionEntry,
)

pytestmark = pytest.mark.unit


def test_red_first_import_and_definition() -> None:
    """Red-first check: NormalisationKind contains event_relative_101."""
    assert NormalisationKind.event_relative_101 == "event_relative_101"
    desc = EventTimeNormalisationDescriptor(
        event_sequence=("address", "top", "impact"),
        n_points=101,
        method="linear",
    )
    assert desc.n_points == 101
    assert desc.event_sequence == ("address", "top", "impact")


def test_linear_signal_analytic_roundtrip() -> None:
    """Known linear signal s(t) = 3*t - 2 round-trips exactly on 101-point grid."""
    time = np.linspace(0.0, 2.0, 241)  # 120 Hz sampling
    signal = 3.0 * time - 2.0
    events = {"address": 0.2, "top": 0.9, "impact": 1.2, "finish": 1.8}

    res = resample_event_relative_101(
        time, signal, events, event_sequence=("address", "impact")
    )
    assert isinstance(res, TimeNormalisedResult)
    assert len(res.grid_percent) == 101
    assert len(res.values) == 101
    assert res.grid_percent[0] == 0.0
    assert res.grid_percent[-1] == 100.0

    # Exact analytic expectation at target timestamps
    expected = 3.0 * res.target_times - 2.0
    np.testing.assert_allclose(res.values, expected, rtol=1e-12, atol=1e-12)


def test_sinusoidal_signal_analytic_resampling() -> None:
    """Sinusoid s(t) = sin(2*pi*f*t) resamples accurately within epsilon."""
    time = np.linspace(0.0, 2.0, 2001)  # Dense time array
    freq = 2.0  # 2 Hz
    signal = np.sin(2.0 * np.pi * freq * time)
    events = {"address": 0.3, "top": 0.8, "impact": 1.1}

    # Linear method
    res_lin = resample_event_relative_101(
        time, signal, events, event_sequence=("address", "impact"), method="linear"
    )
    expected_lin = np.sin(2.0 * np.pi * freq * res_lin.target_times)
    np.testing.assert_allclose(res_lin.values, expected_lin, atol=1e-4)

    # Cubic method
    res_cub = resample_event_relative_101(
        time, signal, events, event_sequence=("address", "impact"), method="cubic"
    )
    expected_cub = np.sin(2.0 * np.pi * freq * res_cub.target_times)
    np.testing.assert_allclose(res_cub.values, expected_cub, atol=1e-6)


def test_multidimensional_signal_resampling() -> None:
    """Resampling works for 2-D signal arrays of shape (M, D)."""
    time = np.linspace(0.0, 1.0, 101)
    signal = np.column_stack([time, 2.0 * time, 3.0 * time])
    events = {"address": 0.1, "impact": 0.9}

    res = resample_event_relative_101(
        time, signal, events, event_sequence=("address", "impact")
    )
    assert res.values.shape == (101, 3)
    np.testing.assert_allclose(res.values[:, 0], res.target_times, atol=1e-12)
    np.testing.assert_allclose(res.values[:, 1], 2.0 * res.target_times, atol=1e-12)


def test_chu_sequence_and_event_percentages() -> None:
    """Multi-event sequence (address -> top -> impact) reports event percentages."""
    time = np.linspace(0.0, 2.0, 201)
    signal = time**2
    events = {"address": 0.4, "top": 1.0, "impact": 1.2}

    res = resample_event_relative_101(
        time, signal, events, event_sequence=("address", "top", "impact")
    )
    assert len(res.grid_percent) == 101
    assert res.event_percentages["address"] == 0.0
    assert res.event_percentages["impact"] == 100.0
    # top at (1.0 - 0.4) / (1.2 - 0.4) = 0.6 / 0.8 = 75.0%
    assert pytest.approx(res.event_percentages["top"], abs=1e-6) == 75.0


def test_phase_points_piecewise_normalisation() -> None:
    """Piecewise normalisation anchors top event at exact designated grid point."""
    time = np.linspace(0.0, 2.0, 201)
    signal = time * 10.0
    events = {"address": 0.4, "top": 1.0, "impact": 1.2}

    res = resample_event_relative_101(
        time,
        signal,
        events,
        event_sequence=("address", "top", "impact"),
        phase_points=(51, 51),
    )
    assert len(res.grid_percent) == 101
    assert res.event_percentages["top"] == 50.0
    assert pytest.approx(res.target_times[50], abs=1e-6) == 1.0


def test_rejects_missing_required_event() -> None:
    """Missing required event raises ValueError."""
    time = np.linspace(0.0, 1.0, 101)
    signal = np.ones_like(time)
    events = {"address": 0.2}

    with pytest.raises(ValueError, match="Missing required event 'impact'"):
        resample_event_relative_101(
            time, signal, events, event_sequence=("address", "impact")
        )


def test_rejects_non_monotonic_event_times() -> None:
    """Non-monotonic or equal event timestamps raise ValueError."""
    time = np.linspace(0.0, 2.0, 201)
    signal = np.ones_like(time)
    events_reversed = {"address": 1.2, "impact": 0.4}

    with pytest.raises(ValueError, match="strictly increasing"):
        resample_event_relative_101(
            time, signal, events_reversed, event_sequence=("address", "impact")
        )

    events_equal = {"address": 0.5, "impact": 0.5}
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_event_relative_101(
            time, signal, events_equal, event_sequence=("address", "impact")
        )


def test_rejects_out_of_bounds_event_times() -> None:
    """Event times outside the time array span raise ValueError."""
    time = np.linspace(0.5, 1.5, 101)
    signal = np.ones_like(time)

    with pytest.raises(ValueError, match="out of bounds"):
        resample_event_relative_101(
            time,
            signal,
            {"address": 0.2, "impact": 1.0},
            event_sequence=("address", "impact"),
        )

    with pytest.raises(ValueError, match="out of bounds"):
        resample_event_relative_101(
            time,
            signal,
            {"address": 0.6, "impact": 1.8},
            event_sequence=("address", "impact"),
        )


def test_rejects_non_finite_inputs() -> None:
    """Non-finite values in time or signal raise ValueError."""
    time = np.linspace(0.0, 1.0, 101)
    signal = np.ones_like(time)
    events = {"address": 0.1, "impact": 0.9}

    time_with_nan = time.copy()
    time_with_nan[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_event_relative_101(
            time_with_nan, signal, events, event_sequence=("address", "impact")
        )

    signal_with_inf = signal.copy()
    signal_with_inf[10] = np.inf
    with pytest.raises(ValueError, match="finite"):
        resample_event_relative_101(
            time, signal_with_inf, events, event_sequence=("address", "impact")
        )


def test_schema_optional_time_normalisation_declaration() -> None:
    """Schema accepts opt-in time_normalisation while preserving backwards compatibility."""
    desc = EventTimeNormalisationDescriptor(
        event_sequence=("address", "top", "impact"),
        n_points=101,
        method="linear",
    )
    valid_kw = {
        "metric_id": "example.x_factor_time_norm",
        "segments": ("thorax", "pelvis"),
        "landmark_proxy": "joint_centers",
        "dimension": "3d",
        "reference_frame": "lab_target_line",
        "cardan_order": "ZXY",
        "sign_convention": "right_handed_positive_on_trail_side",
        "units": "deg",
        "event_anchors": ("address", "top", "impact"),
        "normalisation": "event_relative_101",
        "source": "measured",
        "definition_version": "1.0.0",
        "cites": ("Chu2010",),
        "filter_descriptor": FilterDifferentiationDescriptor(
            lowpass_cutoff_hz=15.0,
            method="butterworth_4th_order",
            min_sampling_hz=120.0,
        ),
        "time_normalisation": desc,
    }
    entry = GolfMetricDefinitionEntry(**valid_kw)
    assert entry.normalisation == NormalisationKind.event_relative_101
    assert entry.time_normalisation is not None
    assert entry.time_normalisation.n_points == 101

    # Without time_normalisation, defaults to None (no change to existing metrics)
    del valid_kw["time_normalisation"]
    valid_kw["normalisation"] = "none"
    plain_entry = GolfMetricDefinitionEntry(**valid_kw)
    assert plain_entry.time_normalisation is None
