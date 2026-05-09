"""Unit tests for module-level :mod:`channels` helpers (issue #4809).

Covers ``magnitude_channel``, ``derivative_channel``, and
``slice_channel`` — including all Design-by-Contract violations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plot_style import (
    DataChannel,
    derivative_channel,
    magnitude_channel,
    slice_channel,
)

# ---------------------------------------------------------------------------
# magnitude_channel
# ---------------------------------------------------------------------------


def test_magnitude_channel_two_d_round_trip_constant_velocity() -> None:
    # constant velocity (3, 4, 0) at every frame -> magnitude 5.0
    velocities = np.tile(np.array([3.0, 4.0, 0.0]), (10, 1))
    ch = magnitude_channel("speed", velocities, unit="m/s")

    assert isinstance(ch, DataChannel)
    assert ch.name == "speed"
    assert ch.unit == "m/s"
    assert ch.n_frames == 10
    assert ch.n_markers is None
    np.testing.assert_allclose(ch.values, np.full(10, 5.0))


def test_magnitude_channel_three_d_per_marker() -> None:
    # Two frames, three markers, each marker has a known 3-vector
    vectors = np.zeros((2, 3, 3))
    vectors[:, 0, :] = (3.0, 4.0, 0.0)  # |.|=5
    vectors[:, 1, :] = (1.0, 0.0, 0.0)  # |.|=1
    vectors[:, 2, :] = (0.0, 0.0, 2.0)  # |.|=2

    ch = magnitude_channel("residual", vectors)
    assert ch.n_frames == 2
    assert ch.n_markers == 3
    assert ch.is_per_marker is True
    expected = np.array([[5.0, 1.0, 2.0], [5.0, 1.0, 2.0]])
    np.testing.assert_allclose(ch.values, expected)


def test_magnitude_channel_default_unit_is_empty() -> None:
    ch = magnitude_channel("v", np.zeros((3, 3)))
    assert ch.unit == ""


def test_magnitude_channel_propagates_nan() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    ch = magnitude_channel("v", vectors)
    assert ch.values[0] == pytest.approx(1.0)
    assert math.isnan(float(ch.values[1]))


def test_magnitude_channel_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        magnitude_channel("v", [[1.0, 2.0, 3.0]])  # type: ignore[arg-type]


def test_magnitude_channel_rejects_one_d_input() -> None:
    with pytest.raises(ValueError, match="ndim"):
        magnitude_channel("v", np.array([1.0, 2.0, 3.0]))


def test_magnitude_channel_rejects_four_d_input() -> None:
    with pytest.raises(ValueError, match="ndim"):
        magnitude_channel("v", np.zeros((2, 2, 2, 3)))


def test_magnitude_channel_rejects_wrong_last_axis() -> None:
    with pytest.raises(ValueError, match="last axis"):
        magnitude_channel("v", np.zeros((4, 2)))
    with pytest.raises(ValueError, match="last axis"):
        magnitude_channel("v", np.zeros((4, 5, 2)))


# ---------------------------------------------------------------------------
# derivative_channel
# ---------------------------------------------------------------------------


def test_derivative_channel_recovers_analytic_derivative_one_d() -> None:
    # x(t) = t^2  =>  dx/dt = 2t
    dt = 0.01
    t = np.arange(0.0, 1.0, dt)
    base = DataChannel.from_array("x", t**2, unit="m")

    deriv = derivative_channel("xdot", base, timestep_s=dt)
    assert deriv.name == "xdot"
    assert deriv.unit == "m/s"
    # numpy.gradient is 2nd-order accurate in the interior
    np.testing.assert_allclose(deriv.values[2:-2], (2.0 * t)[2:-2], atol=1e-10)


def test_derivative_channel_two_d_along_frame_axis() -> None:
    # values_{t,m} = t * (m+1)  =>  d/dt = (m+1)
    dt = 0.5
    n_frames, n_markers = 8, 3
    t = np.arange(n_frames, dtype=float)
    m_scale = np.arange(1, n_markers + 1, dtype=float)
    base_vals = (t[:, None] * dt) * m_scale[None, :]
    base = DataChannel.from_array("v", base_vals)

    deriv = derivative_channel("a", base, timestep_s=dt)
    expected = np.broadcast_to(m_scale, (n_frames, n_markers))
    np.testing.assert_allclose(deriv.values, expected, atol=1e-10)


def test_derivative_channel_unit_suffix_default_when_no_unit() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]))
    deriv = derivative_channel("xdot", base, timestep_s=1.0)
    assert deriv.unit == "/s"


def test_derivative_channel_custom_unit_suffix() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]), unit="rad")
    deriv = derivative_channel("xdot", base, timestep_s=1.0, unit_suffix="/ms")
    assert deriv.unit == "rad/ms"


def test_derivative_channel_propagates_nan() -> None:
    base = DataChannel.from_array("x", np.array([0.0, np.nan, 2.0, 3.0]))
    deriv = derivative_channel("xdot", base, timestep_s=1.0)
    # NaN at idx 1 contaminates the forward edge at idx 0 and the
    # central difference at idx 2 via numpy.gradient.
    assert math.isnan(float(deriv.values[0]))
    assert math.isnan(float(deriv.values[2]))


def test_derivative_channel_rejects_non_channel() -> None:
    with pytest.raises(TypeError, match="DataChannel"):
        derivative_channel("d", np.array([1.0, 2.0]), timestep_s=0.1)  # type: ignore[arg-type]


def test_derivative_channel_rejects_zero_timestep() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="> 0"):
        derivative_channel("d", base, timestep_s=0.0)


def test_derivative_channel_rejects_negative_timestep() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="> 0"):
        derivative_channel("d", base, timestep_s=-0.1)


def test_derivative_channel_rejects_nonfinite_timestep() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        derivative_channel("d", base, timestep_s=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        derivative_channel("d", base, timestep_s=float("nan"))


def test_derivative_channel_rejects_non_numeric_timestep() -> None:
    base = DataChannel.from_array("x", np.array([0.0, 1.0]))
    with pytest.raises(TypeError, match="real number"):
        derivative_channel("d", base, timestep_s="0.1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        derivative_channel("d", base, timestep_s=True)  # type: ignore[arg-type]


def test_derivative_channel_rejects_single_frame() -> None:
    base = DataChannel.from_array("x", np.array([1.0]))
    with pytest.raises(ValueError, match="at least 2 frames"):
        derivative_channel("d", base, timestep_s=0.1)


# ---------------------------------------------------------------------------
# slice_channel
# ---------------------------------------------------------------------------


def test_slice_channel_one_d_basic_window() -> None:
    base = DataChannel.from_array("v", np.arange(10, dtype=float), unit="m")
    sliced = slice_channel(base, slice(2, 6))
    assert sliced.name == "v"
    assert sliced.unit == "m"
    np.testing.assert_array_equal(sliced.values, np.array([2.0, 3.0, 4.0, 5.0]))


def test_slice_channel_preserves_nan_tolerant_auto_range() -> None:
    base_vals = np.array([np.nan, 10.0, 5.0, np.nan, 7.0, np.nan])
    base = DataChannel.from_array("v", base_vals)
    sliced = slice_channel(base, slice(0, 4))

    lo, hi = sliced.auto_range()
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(10.0)
    # value_at preserves NaN pass-through
    assert math.isnan(sliced.value_at(0))
    assert sliced.value_at(1) == pytest.approx(10.0)


def test_slice_channel_two_d_with_marker_subset() -> None:
    base_vals = np.arange(20, dtype=float).reshape(4, 5)
    base = DataChannel.from_array("r", base_vals)

    sliced = slice_channel(base, slice(1, 3), marker_subset=[0, 2, 4])
    assert sliced.n_frames == 2
    assert sliced.n_markers == 3
    np.testing.assert_array_equal(sliced.values, base_vals[1:3][:, [0, 2, 4]])


def test_slice_channel_all_nan_window_keeps_nan_pair_auto_range() -> None:
    base_vals = np.array([1.0, np.nan, np.nan, 4.0])
    base = DataChannel.from_array("v", base_vals)
    sliced = slice_channel(base, slice(1, 3))
    lo, hi = sliced.auto_range()
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_slice_channel_open_slice_bounds_allowed() -> None:
    base = DataChannel.from_array("v", np.arange(5, dtype=float))
    sliced = slice_channel(base, slice(None, None))
    np.testing.assert_array_equal(sliced.values, base.values)


def test_slice_channel_rejects_non_channel() -> None:
    with pytest.raises(TypeError, match="DataChannel"):
        slice_channel(np.zeros(3), slice(0, 2))  # type: ignore[arg-type]


def test_slice_channel_rejects_non_slice_range() -> None:
    base = DataChannel.from_array("v", np.zeros(4))
    with pytest.raises(TypeError, match="slice"):
        slice_channel(base, (0, 2))  # type: ignore[arg-type]


def test_slice_channel_rejects_oob_start() -> None:
    base = DataChannel.from_array("v", np.zeros(4))
    with pytest.raises(ValueError, match="frame_range.start"):
        slice_channel(base, slice(-1, 2))
    with pytest.raises(ValueError, match="frame_range.start"):
        slice_channel(base, slice(99, 100))


def test_slice_channel_rejects_oob_stop() -> None:
    base = DataChannel.from_array("v", np.zeros(4))
    with pytest.raises(ValueError, match="frame_range.stop"):
        slice_channel(base, slice(0, 99))
    with pytest.raises(ValueError, match="frame_range.stop"):
        slice_channel(base, slice(0, -1))


def test_slice_channel_rejects_marker_subset_for_one_d() -> None:
    base = DataChannel.from_array("v", np.zeros(4))
    with pytest.raises(ValueError, match="per-marker"):
        slice_channel(base, slice(0, 2), marker_subset=[0])


def test_slice_channel_rejects_string_marker_subset() -> None:
    base = DataChannel.from_array("r", np.zeros((4, 3)))
    with pytest.raises(TypeError, match="sequence of ints"):
        slice_channel(base, slice(0, 2), marker_subset="012")  # type: ignore[arg-type]


def test_slice_channel_rejects_non_int_marker_index() -> None:
    base = DataChannel.from_array("r", np.zeros((4, 3)))
    with pytest.raises(TypeError, match="int"):
        slice_channel(base, slice(0, 2), marker_subset=[1.5])  # type: ignore[list-item]


def test_slice_channel_rejects_oob_marker_index() -> None:
    base = DataChannel.from_array("r", np.zeros((4, 3)))
    with pytest.raises(ValueError, match="out of range"):
        slice_channel(base, slice(0, 2), marker_subset=[5])
    with pytest.raises(ValueError, match="out of range"):
        slice_channel(base, slice(0, 2), marker_subset=[-1])


def test_slice_channel_rejects_bool_marker_index() -> None:
    base = DataChannel.from_array("r", np.zeros((4, 3)))
    with pytest.raises(TypeError, match="int"):
        slice_channel(base, slice(0, 2), marker_subset=[True])
