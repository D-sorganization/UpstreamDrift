"""Unit tests for :class:`DataChannel`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plot_style import DataChannel


# ---------- Construction ------------------------------------------------


def test_from_array_one_d_happy_path() -> None:
    values = np.array([1.0, 2.0, 3.0])
    channel = DataChannel.from_array("speed", values, unit="m/s")
    assert channel.name == "speed"
    assert channel.unit == "m/s"
    assert channel.n_frames == 3
    assert channel.n_markers is None
    assert channel.is_per_marker is False


def test_from_array_two_d_happy_path() -> None:
    values = np.zeros((4, 5))
    channel = DataChannel.from_array("residuals", values)
    assert channel.n_frames == 4
    assert channel.n_markers == 5
    assert channel.is_per_marker is True


def test_constructor_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DataChannel(name="", values=np.zeros(3))


def test_constructor_rejects_non_string_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DataChannel(name=None, values=np.zeros(3))  # type: ignore[arg-type]


def test_constructor_rejects_non_string_unit() -> None:
    with pytest.raises(TypeError, match="unit must be a string"):
        DataChannel(name="x", values=np.zeros(3), unit=123)  # type: ignore[arg-type]


def test_constructor_rejects_non_array_values() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        DataChannel(name="x", values=[1.0, 2.0])  # type: ignore[arg-type]


def test_constructor_rejects_zero_d_array() -> None:
    with pytest.raises(ValueError, match="ndim"):
        DataChannel(name="x", values=np.array(1.0))


def test_constructor_rejects_three_d_array() -> None:
    with pytest.raises(ValueError, match="ndim"):
        DataChannel(name="x", values=np.zeros((2, 3, 4)))


def test_constructor_rejects_non_numeric_dtype() -> None:
    with pytest.raises(TypeError, match="dtype must be numeric"):
        DataChannel(name="x", values=np.array(["a", "b", "c"]))


# ---------- value_at ----------------------------------------------------


def test_value_at_one_d_returns_scalar() -> None:
    channel = DataChannel.from_array("v", np.array([10.0, 20.0, 30.0]))
    assert channel.value_at(1) == 20.0


def test_value_at_one_d_ignores_marker_idx() -> None:
    channel = DataChannel.from_array("v", np.array([10.0, 20.0, 30.0]))
    assert channel.value_at(1, marker_idx=999) == 20.0


def test_value_at_oob_frame_returns_nan() -> None:
    channel = DataChannel.from_array("v", np.array([1.0, 2.0]))
    assert math.isnan(channel.value_at(5))
    assert math.isnan(channel.value_at(-1))


def test_value_at_two_d_with_marker_idx() -> None:
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    channel = DataChannel.from_array("r", values)
    assert channel.value_at(0, marker_idx=2) == 3.0
    assert channel.value_at(1, marker_idx=0) == 4.0


def test_value_at_two_d_oob_marker_returns_nan() -> None:
    values = np.array([[1.0, 2.0]])
    channel = DataChannel.from_array("r", values)
    assert math.isnan(channel.value_at(0, marker_idx=999))
    assert math.isnan(channel.value_at(0, marker_idx=-1))


def test_value_at_two_d_marker_none_returns_finite_mean() -> None:
    values = np.array([[1.0, 3.0, np.nan]])
    channel = DataChannel.from_array("r", values)
    assert channel.value_at(0) == pytest.approx(2.0)


def test_value_at_two_d_marker_none_all_nan_returns_nan() -> None:
    values = np.array([[np.nan, np.nan]])
    channel = DataChannel.from_array("r", values)
    assert math.isnan(channel.value_at(0))


def test_value_at_rejects_non_int_frame() -> None:
    channel = DataChannel.from_array("v", np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="frame_idx"):
        channel.value_at(1.5)  # type: ignore[arg-type]


def test_value_at_rejects_non_int_marker() -> None:
    channel = DataChannel.from_array("v", np.zeros((2, 3)))
    with pytest.raises(TypeError, match="marker_idx"):
        channel.value_at(0, marker_idx=1.0)  # type: ignore[arg-type]


# ---------- auto_range --------------------------------------------------


def test_auto_range_finite() -> None:
    channel = DataChannel.from_array("v", np.array([1.0, 5.0, 3.0, np.nan]))
    lo, hi = channel.auto_range()
    assert lo == 1.0
    assert hi == 5.0


def test_auto_range_all_nan_returns_nan_pair() -> None:
    channel = DataChannel.from_array("v", np.array([np.nan, np.nan]))
    lo, hi = channel.auto_range()
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_has_finite_range_true_and_false() -> None:
    finite = DataChannel.from_array("v", np.array([0.0, 1.0]))
    infinite = DataChannel.from_array("v", np.array([np.nan]))
    assert finite.has_finite_range() is True
    assert infinite.has_finite_range() is False
