"""Replay identity rejects ambiguous or incompatible native input packages."""

from copy import deepcopy

import pytest

from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_candidate,
)

pytestmark = pytest.mark.unit
NAMES = ("x", "y", "z", "a")
HASH = "a" * 64


@pytest.fixture
def document() -> dict:
    return {
        "schema_version": 1,
        "coordinate_names": list(NAMES),
        "model_sha256": HASH,
        "capture_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "coefficient_order": "highest-power-first",
        "time_basis": "absolute-seconds",
        "force_frame": "world",
        "duration_s": 0.8,
        "q0": [0.0] * 4,
        "qd0": [0.0] * 4,
        "coefficients": [[0.0] * 7 for _ in NAMES],
        "marker_labels": ["tip"],
        "marker_bodies": ["club"],
        "marker_offsets_m": [[0.0, 0.0, 1.0]],
    }


def test_identity_is_stable_and_detached(document: dict) -> None:
    candidate = NativeReplayCandidate.from_document(document, NAMES, HASH)
    original = candidate.sha256
    document["q0"][0] = 1
    detached = candidate.document
    detached["coefficients"][0][0] = 1
    assert candidate.sha256 == original
    assert candidate.document["q0"][0] == 0
    other = deepcopy(candidate.document)
    other["coefficients"][0][0] = 1
    assert NativeReplayCandidate.from_document(other, NAMES, HASH).sha256 != original


@pytest.mark.parametrize(
    "field,value",
    [
        ("coefficient_order", "lowest-power-first"),
        ("time_basis", "normalized"),
        ("force_frame", "hip"),
        ("model_sha256", "d" * 64),
        ("coordinate_names", ["y", "x", "z", "a"]),
        ("duration_s", 0),
        ("q0", [float("nan")] * 4),
        ("marker_offsets_m", [[0, 1]]),
    ],
)
def test_incompatible_package_rejected(
    document: dict, field: str, value: object
) -> None:
    document[field] = value
    with pytest.raises(ValueError):
        NativeReplayCandidate.from_document(document, NAMES, HASH)


def test_missing_or_unrecognized_fields_rejected(document: dict) -> None:
    document.pop("capture_sha256")
    with pytest.raises(ValueError):
        NativeReplayCandidate.from_document(document, NAMES, HASH)


def test_increment_keeps_one_absolute_time_profile(document: dict) -> None:
    import numpy as np

    candidate = NativeReplayCandidate.from_document(document, NAMES, HASH)
    increment = np.zeros((4, 7))
    increment[3, 6] = 2
    updated = increment_native_candidate(candidate, increment, basis_duration_s=0.8)
    coefficients = updated.document["coefficients"]
    for t in (0.0, 0.6, 0.8, 1.0):
        assert np.polyval(coefficients[3], t) == pytest.approx(2 * (t / 0.8) ** 6)
    assert candidate.document["coefficients"][3][0] == 0
    assert candidate.sha256 != updated.sha256
