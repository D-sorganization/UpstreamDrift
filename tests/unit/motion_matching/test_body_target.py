"""Unit tests for the ``BodyTarget`` and ``BodyEvent`` validation rules.

Pins the validation suite that was drafted in the closed-as-duplicate PR
#4494 but never landed when ``BodyTarget`` was bundled into PR #4504. Each
test exercises one validation rule with one failure case (or one happy
path), using ``pytest.raises(..., match=...)`` to pin the error message.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.motion_matching.body_target import (
    MAX_BODY_POSITION_NORM_M,
    BodyEvent,
    BodyTarget,
)

from ._fixtures import make_provenance


def _good_kwargs(n: int = 10, m: int = 4) -> dict:
    """Return a minimal, valid set of constructor kwargs."""
    time = np.linspace(0.0, 0.3, n)
    marker_xyz = np.zeros((n, m, 3), dtype=float)
    marker_names = tuple(f"M{i}" for i in range(m))
    return {
        "time": time,
        "marker_xyz": marker_xyz,
        "marker_names": marker_names,
        "impact_idx": n // 2,
        "events": (),
        "source": make_provenance(),
    }


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_body_target_constructs_valid_target() -> None:
    """A fully-conforming kwargs set yields an immutable ``BodyTarget``."""
    target = BodyTarget(**_good_kwargs())
    assert target.marker_xyz.shape == (10, 4, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.impact_idx = 0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Time vector rules
# --------------------------------------------------------------------------- #


def test_validation_rejects_non_monotonic_time() -> None:
    kwargs = _good_kwargs()
    bad = kwargs["time"].copy()
    bad[5] = bad[4]  # equal -> not strictly increasing
    kwargs["time"] = bad
    with pytest.raises(ValueError, match="strictly increasing"):
        BodyTarget(**kwargs)


def test_validation_rejects_non_zero_start_time() -> None:
    kwargs = _good_kwargs()
    kwargs["time"] = np.linspace(0.1, 0.3, 10)
    with pytest.raises(ValueError, match=r"time\[0\] must be 0"):
        BodyTarget(**kwargs)


def test_validation_rejects_too_few_time_samples() -> None:
    kwargs = _good_kwargs(n=1)
    with pytest.raises(ValueError, match="at least 2 samples"):
        BodyTarget(**kwargs)


def test_validation_rejects_non_1d_time() -> None:
    kwargs = _good_kwargs()
    kwargs["time"] = kwargs["time"].reshape(-1, 1)
    with pytest.raises(ValueError, match="1-D ndarray"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# marker_xyz shape, magnitude, finite-coverage
# --------------------------------------------------------------------------- #


def test_validation_rejects_wrong_marker_xyz_shape() -> None:
    kwargs = _good_kwargs()
    kwargs["marker_xyz"] = np.zeros((10, 4, 2))  # last axis != 3
    with pytest.raises(ValueError, match=r"marker_xyz must have shape"):
        BodyTarget(**kwargs)


def test_validation_rejects_oversize_marker_norm() -> None:
    kwargs = _good_kwargs()
    arr = kwargs["marker_xyz"].copy()
    arr[0, 0, 0] = MAX_BODY_POSITION_NORM_M + 1.0
    kwargs["marker_xyz"] = arr
    with pytest.raises(ValueError, match=r"\|r\| >="):
        BodyTarget(**kwargs)


def test_validation_rejects_insufficient_finite_coverage() -> None:
    kwargs = _good_kwargs(n=10, m=4)
    arr = kwargs["marker_xyz"].copy()
    # Make >50% of every frame's markers non-finite (3 of 4 NaN per frame).
    arr[:, :3, :] = np.nan
    kwargs["marker_xyz"] = arr
    with pytest.raises(ValueError, match=">=50% finite markers"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# marker_names rules
# --------------------------------------------------------------------------- #


def test_validation_rejects_too_few_marker_names() -> None:
    kwargs = _good_kwargs(n=10, m=2)
    with pytest.raises(ValueError, match="at least 3 entries"):
        BodyTarget(**kwargs)


def test_validation_rejects_empty_marker_name() -> None:
    kwargs = _good_kwargs()
    kwargs["marker_names"] = ("M0", "", "M2", "M3")
    with pytest.raises(ValueError, match="non-empty strings"):
        BodyTarget(**kwargs)


def test_validation_rejects_duplicate_marker_names() -> None:
    kwargs = _good_kwargs()
    kwargs["marker_names"] = ("M0", "M1", "M1", "M3")
    with pytest.raises(ValueError, match="must be unique"):
        BodyTarget(**kwargs)


def test_validation_rejects_non_tuple_marker_names() -> None:
    kwargs = _good_kwargs()
    kwargs["marker_names"] = ["M0", "M1", "M2", "M3"]  # list, not tuple
    with pytest.raises(ValueError, match="must be a tuple"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# impact_idx range
# --------------------------------------------------------------------------- #


def test_validation_rejects_out_of_range_impact_idx() -> None:
    kwargs = _good_kwargs(n=10)
    kwargs["impact_idx"] = 10
    with pytest.raises(ValueError, match=r"impact_idx must be in \[0, 10\)"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# events rules
# --------------------------------------------------------------------------- #


def test_validation_rejects_event_frame_out_of_range() -> None:
    kwargs = _good_kwargs(n=10)
    kwargs["events"] = (BodyEvent(label="impact", frame=99, time_s=0.05),)
    with pytest.raises(ValueError, match=r"event frame 99 out of range"):
        BodyTarget(**kwargs)


def test_validation_rejects_duplicate_event_labels() -> None:
    kwargs = _good_kwargs(n=10)
    kwargs["events"] = (
        BodyEvent(label="impact", frame=3, time_s=0.03),
        BodyEvent(label="impact", frame=4, time_s=0.04),
    )
    with pytest.raises(ValueError, match="event labels must be unique"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# source provenance type
# --------------------------------------------------------------------------- #


def test_validation_rejects_non_provenance_source() -> None:
    kwargs = _good_kwargs()
    kwargs["source"] = "not-a-provenance"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="SourceProvenance"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# coordinate_frame
# --------------------------------------------------------------------------- #


def test_validation_rejects_unknown_coordinate_frame() -> None:
    kwargs = _good_kwargs()
    kwargs["coordinate_frame"] = "y_up_left_handed"
    with pytest.raises(ValueError, match="z_up_right_handed"):
        BodyTarget(**kwargs)


# --------------------------------------------------------------------------- #
# BodyEvent direct validation
# --------------------------------------------------------------------------- #


def test_body_event_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        BodyEvent(label="", frame=0, time_s=0.0)


def test_body_event_rejects_non_int_frame() -> None:
    with pytest.raises(TypeError, match="frame must be int"):
        BodyEvent(label="impact", frame=1.5, time_s=0.0)  # type: ignore[arg-type]


def test_body_event_accepts_valid_construction() -> None:
    ev = BodyEvent(label="impact", frame=42, time_s=0.042)
    assert ev.label == "impact"
    assert ev.frame == 42
