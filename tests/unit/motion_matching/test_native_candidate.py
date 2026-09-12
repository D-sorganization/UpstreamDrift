"""Replay identity rejects ambiguous or incompatible native input packages."""

from copy import deepcopy

import pytest

from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_candidate,
    recover_native_increment,
    increment_native_bernstein,
    recover_native_bernstein,
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


def test_restart_recovers_fixed_basis_without_recentering(document: dict) -> None:
    import numpy as np

    base = NativeReplayCandidate.from_document(document, NAMES, HASH)
    delta = np.zeros((4, 7))
    delta[:, 6] = [-2, -0.7, 0.4, 2]
    updated = increment_native_candidate(base, delta, basis_duration_s=0.8)
    recovered = recover_native_increment(base, updated, basis_duration_s=0.8)
    np.testing.assert_allclose(recovered, delta, atol=1e-14)
    # The optimizer coordinate remains relative to the original envelope.
    np.testing.assert_allclose(1 + recovered[:, 6] / 10, [0.8, 0.93, 1.04, 1.2])
    for t in (0.0, 0.6, 0.8, 1.0):
        replay = increment_native_candidate(base, recovered, basis_duration_s=0.8)
        assert np.polyval(replay.document["coefficients"][2], t) == pytest.approx(
            0.4 * (t / 0.8) ** 6
        )


@pytest.mark.parametrize(
    "field,value",
    [("duration_s", 1.0), ("q0", [0.1] * 4), ("capture_sha256", "d" * 64)],
)
def test_restart_rejects_changed_noncontrol_identity(
    document: dict, field, value
) -> None:
    base = NativeReplayCandidate.from_document(document, NAMES, HASH)
    changed = deepcopy(document)
    changed[field] = value
    other = NativeReplayCandidate.from_document(changed, NAMES, HASH)
    with pytest.raises(ValueError, match="non-control"):
        recover_native_increment(base, other, basis_duration_s=0.8)


def test_bernstein_shaping_preserves_sextic_time_and_restart(document: dict) -> None:
    import numpy as np

    base = NativeReplayCandidate.from_document(document, NAMES, HASH)
    controls = np.zeros((4, 7))
    controls[0, 4:] = [2, -1, 0.7]
    result = increment_native_bernstein(base, controls, basis_duration_s=0.8)
    for t in (0, 0.3, 0.6, 0.8, 1.0):
        s = t / 0.8
        expected = 30 * s**4 * (1 - s) ** 2 - 6 * s**5 * (1 - s) + 0.7 * s**6
        assert np.polyval(result.document["coefficients"][0], t) == pytest.approx(
            expected, abs=1e-12
        )
    np.testing.assert_allclose(
        recover_native_bernstein(base, result, basis_duration_s=0.8),
        controls,
        atol=1e-12,
    )
    controls[:] = 0
    controls[:, 6] = 0.7
    power = increment_native_candidate(base, controls, basis_duration_s=0.8)
    assert (
        increment_native_bernstein(base, controls, basis_duration_s=0.8).sha256
        == power.sha256
    )
