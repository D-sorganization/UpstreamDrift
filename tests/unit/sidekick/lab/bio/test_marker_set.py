"""Tests for the C3D marker-set detection layer (issue #4710)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sidekick.lab.bio._c3d_io import build_metadata
from sidekick.lab.bio._c3d_marker_set import (
    MarkerSet,
    MarkerSetMismatchError,
    detect_marker_set,
)

from ._synthetic import _synthetic_c3d_dict

pytestmark = pytest.mark.unit


PIG_LABELS = [
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "C7",
    "T10",
    "CLAV",
    "STRN",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "RTHI",
    "LKNE",
    "RKNE",
    "LTIB",
    "RTIB",
    "LANK",
    "RANK",
    "LHEE",
    "RHEE",
    "LTOE",
    "RTOE",
]

CGM24_LABELS = [
    *PIG_LABELS,
    "LKNM",
    "RKNM",
    "LMED",
    "RMED",
]

IOR_LABELS = [
    "R_ASIAS",
    "L_ASIAS",
    "R_AISPS",
    "L_AISPS",
    "R_KNEE",
    "L_KNEE",
    "R_ANKLE",
    "L_ANKLE",
    "C7",
    "T8",
]

GOLF_CLUSTER_LABELS = [
    "GripButt",
    "ClubHead",
    "M1",
    "M2",
    "M3",
    "M4",
]

TOUR_AVERAGE_BODY_LABELS = [
    "Marker_0:0:0",
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "LElbowOut",
    "LUArmHigh",
    "LWristTop",
    "RShoulderTop",
    "RShoulderBack",
    "RElbowOut",
    "RUArmHigh",
    "RWristTop",
    "LKneeOut",
    "LToeIn",
    "LToeOut",
    "LAnkleOut",
    "RKneeOut",
    "RToeIn",
    "RToeOut",
    "RAnkleOut",
    "Marker_2:2:1",
    "Marker_2:2:2",
    "Marker_2:2:3",
    "Marker_3:3:1",
    "Marker_3:3:2",
    "Marker_3:3:3",
    "Uname*36",
    "Uname*37",
    "Uname*38",
]


def test_detect_marker_set_returns_unknown_for_empty_labels() -> None:
    """No labels and no parameters means UNKNOWN."""
    assert detect_marker_set([], None) is MarkerSet.UNKNOWN


def test_detect_marker_set_pig_by_labels() -> None:
    """Plug-in-Gait markers should be detected via label coverage."""
    assert detect_marker_set(PIG_LABELS, {}) is MarkerSet.PLUG_IN_GAIT_41


def test_detect_marker_set_cgm24_by_labels() -> None:
    """CGM2.4 includes the medial markers and should beat PiG via priority order."""
    assert detect_marker_set(CGM24_LABELS, {}) is MarkerSet.CGM2_4


def test_detect_marker_set_ior_by_labels() -> None:
    """IOR underscore-prefixed markers should classify as IOR."""
    assert detect_marker_set(IOR_LABELS, {}) is MarkerSet.IOR


def test_detect_marker_set_golf_cluster_by_labels() -> None:
    """Files containing both grip and clubhead markers map to GOLF_CLUSTER."""
    assert detect_marker_set(GOLF_CLUSTER_LABELS, {}) is MarkerSet.GOLF_CLUSTER


def test_detect_marker_set_tour_average_body_by_labels() -> None:
    """Tour-average body marker labels classify without an explicit override."""
    assert (
        detect_marker_set(TOUR_AVERAGE_BODY_LABELS, {})
        is MarkerSet.GOLF_TOUR_AVERAGE_BODY
    )


def test_detect_marker_set_name_overrides_labels() -> None:
    """A declared SUBJECTS.MARKER_SETS name takes priority over heuristics."""
    params = {
        "SUBJECTS": {"MARKER_SETS": {"value": ["CGM2.4"]}},
    }
    # Even with PiG-only labels, the declared name wins.
    assert detect_marker_set(PIG_LABELS, params) is MarkerSet.CGM2_4


def test_detect_marker_set_model_name_used() -> None:
    """MODEL.NAME provides a fallback declared-name signal."""
    params = {"MODEL": {"NAME": {"value": ["VICON PIG"]}}}
    assert detect_marker_set(["X"], params) is MarkerSet.PLUG_IN_GAIT_41


def test_detect_marker_set_unknown_when_no_signature_fires() -> None:
    """Random labels return UNKNOWN."""
    assert detect_marker_set(["foo", "bar", "baz"], {"OTHER": {}}) is MarkerSet.UNKNOWN


def test_build_metadata_populates_marker_set() -> None:
    """build_metadata propagates the detected marker set into metadata."""
    raw = _synthetic_c3d_dict(
        n_frames=3,
        n_markers=len(PIG_LABELS),
        marker_names=PIG_LABELS,
    )
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert metadata.marker_set is MarkerSet.PLUG_IN_GAIT_41


def test_build_metadata_unknown_marker_set_default() -> None:
    """Files with unknown labels report MarkerSet.UNKNOWN."""
    raw = _synthetic_c3d_dict(
        n_frames=2,
        n_markers=3,
        marker_names=["X1", "X2", "X3"],
    )
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert metadata.marker_set is MarkerSet.UNKNOWN


def test_marker_set_mismatch_error_carries_context() -> None:
    """MarkerSetMismatchError exposes the detected set and label list."""
    err = MarkerSetMismatchError("boom", detected=MarkerSet.UNKNOWN, labels=["a", "b"])
    assert err.detected is MarkerSet.UNKNOWN
    assert err.labels == ["a", "b"]
    assert isinstance(err, ValueError)
