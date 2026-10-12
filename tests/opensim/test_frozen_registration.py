"""Registration ownership and training-only calibration (#11912)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
    compute_capture_registration,
    register_points,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        FrozenCaptureRegistration,
    )


def _points() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.0]])


def test_registration_owns_immutable_transform_arrays() -> None:
    rotation, translation = np.eye(3), np.array([0.1, 0.2, 0.3])
    registration = CaptureRegistration(rotation, translation)
    rotation[:] = 0
    translation[:] = 100
    np.testing.assert_array_equal(registration.rotation, np.eye(3))
    np.testing.assert_array_equal(registration.translation, [0.1, 0.2, 0.3])
    for values in (registration.rotation, registration.translation):
        with pytest.raises(ValueError):
            values.flat[0] = 10
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_registration_rejects_degenerate_target_geometry() -> None:
    with pytest.raises(ValueError, match="Target.*degenerate|target.*degenerate"):
        compute_capture_registration(_points(), np.zeros((3, 3)))


def test_frozen_registration_rejects_missing_raw_source_identity() -> None:
    with pytest.raises(ValueError, match="source.*SHA-256"):
        _fit(replace(_capture(), source_sha256=None))


def _capture() -> TourCapture:
    points = np.tile(_points(), (4, 1, 1))
    points[2:] += [0.0, 0.0, 0.1]
    return TourCapture(
        np.array([0.0, 0.01, 0.03, 0.06]),
        ("A", "B", "C"),
        points,
        np.ones((4, 3), dtype=bool),
        "a" * 64,
    )


def _fit(capture: TourCapture) -> FrozenCaptureRegistration:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        fit_training_registration,
    )

    return fit_training_registration(
        capture,
        (0, 1),
        dict(zip(capture.labels, _points() + [1, 2, 3], strict=True)),
        target_geometry_sha256="b" * 64,
        anchor_provenance=dict.fromkeys(capture.labels, "original synthetic anchor"),
        target_frame="synthetic-native-world",
        source_frame="synthetic-capture",
        interpretation="declared-unverified",
    )


def test_holdout_observations_cannot_influence_training_registration() -> None:
    source = _capture()
    first = _fit(source)
    changed = source.points_m.copy()
    changed[2:] += [5, -10, 20]
    second = _fit(replace(source, points_m=changed))
    np.testing.assert_array_equal(first.transform.rotation, second.transform.rotation)
    np.testing.assert_array_equal(
        first.transform.translation, second.transform.translation
    )
    assert first.identity_sha256 == second.identity_sha256
    assert first.training_frames == (0, 1)
    assert first.training_times_s == (0.0, 0.01)
    assert not first.anatomically_qualified


def test_frozen_application_retains_clock_missingness_and_lineage() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        apply_frozen_registration,
    )

    source = _capture()
    frozen = _fit(source)
    valid = source.valid.copy()
    valid[2, 1] = False
    points = source.points_m.copy()
    points[2, 1] = np.nan
    observed = replace(source, points_m=points, valid=valid)
    result = apply_frozen_registration(
        observed, frozen, source_frame="synthetic-capture"
    )
    np.testing.assert_array_equal(result.capture.time_s, source.time_s)
    np.testing.assert_array_equal(result.capture.valid, valid)
    assert result.capture.labels == source.labels
    assert result.capture.source_sha256 == source.source_sha256
    assert result.registration_sha256 == frozen.identity_sha256
    np.testing.assert_allclose(
        result.capture.points_m[valid],
        register_points(observed.points_m[valid], frozen.transform),
        atol=1e-12,
    )
    assert np.isnan(result.capture.points_m[2, 1]).all()


@pytest.mark.parametrize("frames", [(2, 2), (-1,), (4,), (1, 0), (True,)])
def test_training_frames_are_explicit_valid_original_indices(frames: tuple) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        fit_training_registration,
    )

    source = _capture()
    with pytest.raises(ValueError, match="training"):
        fit_training_registration(
            source,
            frames,
            dict(zip(source.labels, _points(), strict=True)),
            target_geometry_sha256="b" * 64,
            anchor_provenance=dict.fromkeys(source.labels, "synthetic"),
            target_frame="native-world",
            source_frame="synthetic-capture",
            interpretation="declared-unverified",
        )


def test_missing_training_observation_is_not_filled() -> None:
    source = _capture()
    valid = source.valid.copy()
    valid[1, 0] = False
    with pytest.raises(ValueError, match="observed"):
        _fit(replace(source, valid=valid))


def test_unknown_correspondences_are_not_inferred_from_names() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        fit_training_registration,
    )

    source = _capture()
    with pytest.raises(ValueError, match="provenance|correspondence"):
        fit_training_registration(
            source,
            (0,),
            dict(zip(source.labels, _points(), strict=True)),
            target_geometry_sha256="b" * 64,
            anchor_provenance={},
            target_frame="native-world",
            source_frame="synthetic-capture",
            interpretation="declared-unverified",
        )


def test_frozen_registration_rejects_an_unrelated_observation_frame() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        apply_frozen_registration,
    )

    source = _capture()
    with pytest.raises(ValueError, match="source frame"):
        apply_frozen_registration(
            source, _fit(source), source_frame="unrelated-session"
        )


def test_training_coordinate_gauge_cannot_claim_anatomical_correspondence() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
        pelvis_coordinate_gauge,
    )
    from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS

    local = np.array(
        [[0.05, 0, -0.1], [0.05, 0, 0.1], [-0.05, 0, -0.1], [-0.05, 0, 0.1]]
    )
    rotation = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    points = np.tile(local @ rotation.T + [3, 0.9, 4], (3, 1, 1))
    source = TourCapture(
        np.array([0, 0.01, 0.02]),
        MARKER_SEGMENTS["pelvis"],
        points,
        np.ones((3, 4), bool),
        "a" * 64,
    )
    frozen = pelvis_coordinate_gauge(
        source,
        (np.eye(3), np.array([0, 0.93, 0])),
        target_left_axis_local=np.array([0, 0, -1.0]),
        target_geometry_sha256="b" * 64,
        training_frame=0,
    )
    actual = register_points(points[0], frozen.transform)
    np.testing.assert_allclose(actual, local + [0, 0.93, 0], atol=1e-12)
    assert frozen.interpretation == "coordinate-gauge"
    assert not frozen.anatomically_qualified
    changed = points.copy()
    changed[1:] += 100
    second = pelvis_coordinate_gauge(
        replace(source, points_m=changed),
        (np.eye(3), np.array([0, 0.93, 0])),
        target_left_axis_local=np.array([0, 0, -1.0]),
        target_geometry_sha256="b" * 64,
        training_frame=0,
    )
    assert second.identity_sha256 == frozen.identity_sha256
