"""Reviewed contact phases preserve image observations and native ground semantics."""

from dataclasses import asdict
import json
import numpy as np
import pytest
from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.motion_matching.historical_fit.contact_schedule import (
    ContactPinPhase,
    ContactPinSchedule,
    ScheduledConstraintOptions,
)

pytestmark = pytest.mark.unit
HASH = "sha256:" + "a" * 64


def options():
    return ConstraintOptions(
        GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1, pinned_spheres=("heel_r",)
    )


def schedule():
    return ContactPinSchedule(
        "capture",
        HASH,
        (
            ContactPinPhase((0, 1), (1, 2), ("heel_r",), (HASH,)),
            ContactPinPhase((1, 2), (1, 1), (), (HASH,)),
        ),
    )


def test_boundary_unknown_phase_and_json_roundtrip():
    wrapped = ScheduledConstraintOptions(options(), schedule())
    assert wrapped.resolve(0).pinned_spheres == ("heel_r",)
    assert wrapped.resolve(0.5).pinned_spheres == ()
    assert wrapped.resolve(1).pinned_spheres == ()
    decoded = ImageFitConfig.from_record(
        json.loads(json.dumps(asdict(ImageFitConfig(constraint_options=wrapped))))
    )
    assert decoded.constraint_options == wrapped
    with pytest.raises(ValueError):
        wrapped.resolve(1.01)


@pytest.mark.parametrize(
    "start,end,pins,hashes",
    [
        ((0, 0), (1, 1), (), (HASH,)),
        ((0, 1), (0, 1), (), (HASH,)),
        ((False, 1), (1, 1), (), (HASH,)),
        ((0, 1), (1, 1), ("x", "x"), (HASH,)),
        ((0, 1), (1, 1), (), ()),
    ],
)
def test_invalid_phase_rejected(start, end, pins, hashes):
    with pytest.raises(ValueError):
        ContactPinPhase(start, end, pins, hashes)


def test_schedule_gap_and_mutable_inputs_rejected():
    with pytest.raises(ValueError):
        ContactPinSchedule(
            "capture",
            HASH,
            (
                ContactPinPhase((0, 1), (1, 4), (), (HASH,)),
                ContactPinPhase((1, 2), (1, 1), (), (HASH,)),
            ),
        )
    with pytest.raises(ValueError):
        ContactPinSchedule("capture", HASH, list(schedule().phases))


def test_native_scheduled_rows_nonpenetration_and_jacobian():
    pytest.importorskip("mujoco")
    from tests.unit.motion_matching.test_historical_image_fit import native_problem
    from src.shared.python.motion_matching.historical_fit.solver import _Fit
    from src.shared.python.estimation import finite_difference_jacobian

    native, attachments, camera, inputs = native_problem()
    start, end = inputs.source_times[[0, -1]]
    from fractions import Fraction

    def pts(value):
        fraction = Fraction(float(value))
        return fraction.numerator, fraction.denominator

    midpoint = (start + end) / 2
    phases = (
        ContactPinPhase(pts(start), pts(midpoint), ("heel_r",), (HASH,)),
        ContactPinPhase(pts(midpoint), pts(end), (), (HASH,)),
    )
    config = ImageFitConfig(
        constraint_options=ScheduledConstraintOptions(
            options(), ContactPinSchedule("capture", HASH, phases)
        )
    )
    fit = _Fit(native, attachments, camera, inputs, config)
    q = inputs.seed.copy()
    rows = fit.constraint_linearizations(np.array([q, q]), np.array([start, end]))
    assert rows[0].row_labels == rows[1].row_labels
    for time, row in zip((start, end), rows, strict=True):
        numerical = finite_difference_jacobian(
            lambda pose, time=time: (
                fit.constraint_linearizations(pose[None, :], np.array([time]))[
                    0
                ].residual
            ),
            q,
        )
        np.testing.assert_allclose(row.jacobian, numerical, atol=2e-6, rtol=2e-5)
    assert np.isfinite(rows[1].residual).all()


@pytest.mark.parametrize(
    "record",
    [
        {"capture_id": "c", "capture_sha256": "sha256:" + "A" * 64},
        {"capture_id": "c", "capture_sha256": HASH, "status": "qualified"},
    ],
)
def test_identity_or_qualified_status_rejected(record):
    with pytest.raises(ValueError):
        ContactPinSchedule(phases=schedule().phases, **record)


def test_native_release_does_not_remove_nonpenetration():
    pytest.importorskip("mujoco")
    from dataclasses import replace
    from tests.unit.motion_matching.test_historical_image_fit import native_problem

    native, attachments, _, inputs = native_problem()
    ik = native.create_ik(attachments)
    for height in (-10.0, 10.0):
        base = replace(options(), ground=GroundPlane((0, 0, 1), height))
        pinned = ik.constraint_residual_jacobian(inputs.seed, base)
        released = ik.constraint_residual_jacobian(
            inputs.seed, replace(base, pinned_spheres=())
        )
        assert pinned.row_labels == released.row_labels
        if height < 0:
            assert np.linalg.norm(pinned.residual[6:]) > 0
            np.testing.assert_array_equal(released.residual[6:], 0)
        else:
            np.testing.assert_allclose(released.residual, pinned.residual)
            np.testing.assert_allclose(released.jacobian, pinned.jacobian)


def preserved_zero_start(native, inputs):
    from src.shared.python.motion_matching.historical_fit import ImageSplineStart

    return ImageSplineStart.from_coefficients(
        inputs.knot_times,
        np.zeros(2 * len(inputs.knot_times) * len(inputs.free_coordinates)),
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )


def test_schedule_boundary_is_constraint_probe_not_image_observation():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.historical_fit import (
        initialize_image_trajectory,
    )
    from tests.unit.motion_matching.test_historical_image_fit import native_problem

    native, attachments, camera, inputs = native_problem()
    phases = (
        ContactPinPhase((110, 1), (221, 2), ("heel_r",), (HASH,)),
        ContactPinPhase((221, 2), (111, 1), (), (HASH,)),
    )
    wrapped = ScheduledConstraintOptions(
        options(), ContactPinSchedule("capture", HASH, phases)
    )
    result = initialize_image_trajectory(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(constraint_options=wrapped),
        preserved_zero_start(native, inputs),
    )
    np.testing.assert_array_equal(result.constraint_times, [110, 110.5, 111])
    np.testing.assert_array_equal(result.source_times, inputs.source_times)
    assert result.observed_point_count == 6
    plain = initialize_image_trajectory(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(constraint_options=options()),
        preserved_zero_start(native, inputs),
    )
    assert result.rms_pixels == plain.rms_pixels


def test_schedule_mismatching_interval_and_unknown_native_pin_rejected():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.historical_fit.solver import _Fit
    from tests.unit.motion_matching.test_historical_image_fit import native_problem

    native, attachments, camera, inputs = native_problem()
    with pytest.raises(ValueError, match="interval"):
        _Fit(
            native,
            attachments,
            camera,
            inputs,
            ImageFitConfig(
                constraint_options=ScheduledConstraintOptions(options(), schedule())
            ),
        )
    invalid = ContactPinSchedule(
        "capture", HASH, (ContactPinPhase((110, 1), (111, 1), ("unknown",), (HASH,)),)
    )
    fit = _Fit(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(
            constraint_options=ScheduledConstraintOptions(options(), invalid)
        ),
    )
    with pytest.raises(ValueError, match="sphere|contact"):
        fit.constraint_linearizations(inputs.seed[None, :], np.array([110.0]))


def test_legacy_options_roundtrip_and_schedule_immutable():
    from dataclasses import FrozenInstanceError

    legacy = ImageFitConfig(constraint_options=options())
    assert ImageFitConfig.from_record(json.loads(json.dumps(asdict(legacy)))) == legacy
    reviewed = schedule()
    with pytest.raises(FrozenInstanceError):
        reviewed.capture_id = "other"
    assert reviewed.boundary_times() == (0.0, 0.5, 1.0)


def test_json_strings_cannot_masquerade_as_phase_arrays():
    record = asdict(
        ImageFitConfig(
            constraint_options=ScheduledConstraintOptions(options(), schedule())
        )
    )
    record["constraint_options"]["schedule"]["phases"] = "bad"
    with pytest.raises(ValueError, match="array"):
        ImageFitConfig.from_record(record)


def test_legacy_malformed_ground_has_specific_diagnostic():
    record = asdict(ImageFitConfig(constraint_options=options()))
    record["constraint_options"]["ground"] = "invalid"
    with pytest.raises(ValueError, match="ground"):
        ImageFitConfig.from_record(record)
