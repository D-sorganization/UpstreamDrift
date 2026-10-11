"""Tests for capture-O (owner) entry in TourCaptureSpec and validity policies (#11166)."""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np

from src.motion_capture.capture_registry import require_capture
from src.shared.python.motion_matching import tour_capture_contract as module
from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TOUR_CAPTURES,
    TourCapture,
    TourCaptureSpec,
    capture_kind,
    get_marker_validity_policy,
    load_tour_capture,
    verify_capture_content,
)

pytestmark = pytest.mark.unit

OWNER_SHA256 = "2569659eff1e0b00294fd3c213d332cb609d8e37c31853f4bce9eb860e99b160"


def test_owner_capture_spec_is_frozen_and_complete() -> None:
    """Synthetic contract check: verifies spec constants without requiring private data."""
    assert "owner" in TOUR_CAPTURES
    spec = TOUR_CAPTURES["owner"]
    assert isinstance(spec, TourCaptureSpec)
    assert spec.frames == 367
    assert spec.rate_hz == 240.0
    assert spec.duration_s == pytest.approx(366 / 240.0)
    assert spec.units == "m"
    assert spec.vertical_axis == "y"
    assert spec.labels == TOUR_CAPTURE.labels
    assert len(spec.labels) == 38
    assert spec.sha256 == OWNER_SHA256
    assert capture_kind(OWNER_SHA256) == "owner"


def test_owner_validity_policy() -> None:
    """Verifies marker validity policy and weights for owner capture."""
    policy = get_marker_validity_policy("owner")
    assert len(policy) == 38
    for label in TOUR_CAPTURE.labels:
        assert label in policy
        entry = policy[label]
        assert entry.valid_samples + entry.missing_samples == 367
        if label in module.MARKER_SEGMENTS["unassigned"]:
            assert entry.excluded is True
            assert entry.nominal_weight == 0.0
            assert policy.weight_for(label, is_valid=True) == 0.0
        else:
            assert entry.excluded is False
            assert entry.nominal_weight == 1.0


def test_synthetic_owner_capture_validates_shapes() -> None:
    """Synthetic capture with owner rate and frames runs in public CI."""
    t = np.arange(367) / 240.0
    points = np.zeros((367, 38, 3))
    valid = np.ones((367, 38), dtype=bool)
    capture = TourCapture(
        t, TOUR_CAPTURE.labels, points, valid, source_sha256=OWNER_SHA256
    )
    assert capture.frames == 367
    assert capture.rate_hz == pytest.approx(240.0)
    assert capture.duration_s == pytest.approx(366 / 240.0)
    assert capture.valid_count() == 367 * 38


def test_real_owner_capture_matches_spec() -> None:
    """If private data is available, loads capture-O and checks exact properties."""
    pytest.importorskip("ezc3d")
    path = require_capture("capture-O")

    kind, spec = verify_capture_content(path, expected_kind="owner")
    assert kind == "owner"
    assert spec.sha256 == OWNER_SHA256

    capture = load_tour_capture(path)
    assert capture.frames == 367
    assert capture.rate_hz == pytest.approx(240.0)
    assert capture.labels == TOUR_CAPTURE.labels
    assert capture.time_s[0] == 0.0
    assert capture.time_s[-1] == pytest.approx(366 / 240.0)
    assert capture.points_m.shape == (367, 38, 3)

    # Vertical axis is y: HeadTop sits above toes at address
    head = capture.points_m[0, capture.index("HeadTop")]
    toe = capture.points_m[0, capture.index("LToeOut")]
    assert head[1] > toe[1] + 1.0

    assert capture.valid_count() == 12773
    assert capture.source_sha256 == OWNER_SHA256
