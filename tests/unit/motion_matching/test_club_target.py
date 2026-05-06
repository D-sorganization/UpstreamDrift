"""Unit tests for the ``ClubTarget`` dataclass and validation rules."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)

from ._fixtures import make_provenance, make_target


def test_club_target_immutable() -> None:
    target = make_target()
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.impact_idx = 99  # type: ignore[misc]


def test_validation_rejects_non_monotonic_time() -> None:
    n = 10
    bad_time = np.linspace(0.0, 0.3, n)
    bad_time[5] = bad_time[4]  # equal -> not strictly increasing
    with pytest.raises(ValueError, match="strictly increasing"):
        ClubTarget(
            time=bad_time,
            butt=np.zeros((n, 3)),
            clubhead=np.zeros((n, 3)) + np.array([0, 0, 1.0]),
            club_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
            impact_idx=5,
            source=make_provenance(),
        )


def test_validation_rejects_non_zero_start() -> None:
    n = 10
    time = np.linspace(0.1, 0.3, n)
    with pytest.raises(ValueError, match="time\\[0\\] must be 0"):
        ClubTarget(
            time=time,
            butt=np.zeros((n, 3)),
            clubhead=np.zeros((n, 3)) + np.array([0, 0, 1.0]),
            club_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
            impact_idx=5,
            source=make_provenance(),
        )


def test_validation_rejects_non_unit_quaternion() -> None:
    n = 10
    time = np.linspace(0, 0.3, n)
    bad_quat = np.tile([2.0, 0, 0, 0], (n, 1))
    with pytest.raises(ValueError, match="unit-norm"):
        ClubTarget(
            time=time,
            butt=np.zeros((n, 3)),
            clubhead=np.zeros((n, 3)) + np.array([0, 0, 1.0]),
            club_quat=bad_quat,
            impact_idx=5,
            source=make_provenance(),
        )


def test_validation_rejects_nan_position() -> None:
    n = 10
    time = np.linspace(0, 0.3, n)
    butt = np.zeros((n, 3))
    butt[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ClubTarget(
            time=time,
            butt=butt,
            clubhead=np.zeros((n, 3)) + np.array([0, 0, 1.0]),
            club_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
            impact_idx=5,
            source=make_provenance(),
        )


def test_validation_rejects_implausible_radius() -> None:
    n = 10
    time = np.linspace(0, 0.3, n)
    big = np.zeros((n, 3))
    big[:, 0] = 99.0
    with pytest.raises(ValueError, match=">="):
        ClubTarget(
            time=time,
            butt=big,
            clubhead=big + np.array([0, 0, 0.1]),
            club_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
            impact_idx=5,
            source=make_provenance(),
        )


def test_validation_rejects_impact_idx_out_of_range() -> None:
    n = 10
    time = np.linspace(0, 0.3, n)
    with pytest.raises(ValueError, match="impact_idx"):
        ClubTarget(
            time=time,
            butt=np.zeros((n, 3)),
            clubhead=np.zeros((n, 3)) + np.array([0, 0, 1.0]),
            club_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
            impact_idx=999,
            source=make_provenance(),
        )


def test_source_provenance_is_frozen() -> None:
    p = make_provenance()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.format = "c3d"  # type: ignore[misc]


def test_source_provenance_required_fields() -> None:
    p = SourceProvenance(
        filename="x", format="excel", subject_id="s", trial_id="t", sha256="abc"
    )
    assert p.filename == "x"
    assert p.format == "excel"
