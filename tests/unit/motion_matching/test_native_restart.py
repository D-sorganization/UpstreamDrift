"""Restart controls retain the original physical envelope and identity."""

from copy import deepcopy

import numpy as np
import pytest

from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)
from src.shared.python.motion_matching.native_restart import (
    NativeRestart,
    prepare_native_restart,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def base() -> NativeReplayCandidate:
    names = ["Px", "Py", "Pz"]
    return NativeReplayCandidate.from_document(
        {
            "schema_version": 1,
            "coordinate_names": names,
            "model_sha256": "a" * 64,
            "capture_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "coefficient_order": "highest-power-first",
            "time_basis": "absolute-seconds",
            "force_frame": "world",
            "duration_s": 0.85,
            "coefficients": np.zeros((3, 7)).tolist(),
            "q0": [0.0] * 3,
            "qd0": [0.0] * 3,
            "marker_labels": ["marker"],
            "marker_bodies": ["body"],
            "marker_offsets_m": [[0.0] * 3],
        },
        names,
        "a" * 64,
    )


def prepare(
    base: NativeReplayCandidate,
    seed: NativeReplayCandidate,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    tolerance: float = 1e-10,
) -> NativeRestart:
    return prepare_native_restart(
        base,
        seed,
        basis_duration_s=0.8,
        lower_controls=np.full((3, 7), -2.0) if lower is None else lower,
        upper_controls=np.full((3, 7), 2.0) if upper is None else upper,
        roundoff_tolerance=tolerance,
    )


def test_restart_preserves_absolute_controls_and_initial_state(
    base: NativeReplayCandidate,
) -> None:
    controls = np.linspace(-1.9, 1.9, 21).reshape(3, 7)
    seed = increment_native_bernstein(base, controls, basis_duration_s=0.8)
    result = prepare(base, seed)
    np.testing.assert_allclose(result.controls, controls, atol=1e-12)
    assert result.source_candidate_sha256 == seed.sha256
    assert result.candidate.document["q0"] == seed.document["q0"]
    assert result.candidate.document["duration_s"] == 0.85
    assert not result.controls.flags.writeable
    for time in [0, 0.6, 0.8, 0.85]:
        for actual, expected in zip(
            result.candidate.document["coefficients"],
            seed.document["coefficients"],
            strict=True,
        ):
            assert np.polyval(actual, time) == pytest.approx(
                np.polyval(expected, time), abs=1e-10
            )


def test_only_explicit_roundoff_is_snapped_to_original_bounds(
    base: NativeReplayCandidate,
) -> None:
    controls = np.full((3, 7), 2.0 + 5e-12)
    seed = increment_native_bernstein(base, controls, basis_duration_s=0.8)
    result = prepare(base, seed)
    assert np.all(result.controls <= 2.0)
    assert 0 < result.max_bound_snap <= 1e-10
    assert result.snapped_control_count == 21
    assert result.candidate.sha256 != seed.sha256


def test_outside_envelope_is_rejected_not_recentered(
    base: NativeReplayCandidate,
) -> None:
    seed = increment_native_bernstein(base, np.full((3, 7), 2.01), basis_duration_s=0.8)
    with pytest.raises(ValueError, match="bounds"):
        prepare(base, seed)


@pytest.mark.parametrize(
    "key,value", [("q0", [1.0] * 3), ("capture_sha256", "d" * 64), ("duration_s", 1.0)]
)
def test_changed_noncontrol_identity_rejected(
    base: NativeReplayCandidate, key: str, value: object
) -> None:
    document = deepcopy(base.document)
    document[key] = value
    seed = NativeReplayCandidate.from_document(document, ["Px", "Py", "Pz"], "a" * 64)
    with pytest.raises(ValueError, match="non-control"):
        prepare(base, seed)


@pytest.mark.parametrize(
    "lower,upper,tolerance",
    [
        (np.zeros(21), np.ones((3, 7)), 1e-10),
        (np.ones((3, 7)), np.zeros((3, 7)), 1e-10),
        (np.full((3, 7), np.nan), np.ones((3, 7)), 1e-10),
        (np.zeros((3, 7)), np.ones((3, 7)), -1),
        (np.zeros((3, 7)), np.ones((3, 7)), np.inf),
    ],
)
def test_invalid_restart_envelope_rejected(
    base: NativeReplayCandidate, lower: np.ndarray, upper: np.ndarray, tolerance: float
) -> None:
    with pytest.raises(ValueError):
        prepare(base, base, lower, upper, tolerance)
