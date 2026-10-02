"""Coordinate expansion preserves polynomial motion without pretending exact restart."""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import ImageSplineStart
from src.shared.python.motion_matching.historical_fit.spline_expansion import (
    expand_image_spline_coordinates,
)

pytestmark = pytest.mark.unit


def example():
    order = tuple(f"c{index:02}" for index in range(44))
    old = ("c01", "c03", "c07")
    knots = np.array([2.0, 2.25, 4.0])
    q = np.array([[0.1, 0.2, 0.3], [0.3, 0.5, 0.4], [0.5, 0.6, 0.7]])
    v = np.array([[0.2, 0.3, 0.4], [0.1, 0.4, 0.2], [0.3, 0.1, 0.2]])
    coefficients = CubicHermiteSplineTrajectory(knots, 3).pack(q, v)
    start = ImageSplineStart.from_coefficients(
        knots, coefficients, order, old, "native-fixture"
    )
    reference = np.arange(44, dtype=float) / 100
    reference[[order.index(name) for name in old]] = q[0]
    return start, reference, ("c00", "c01", "c02", "c03", "c04", "c07")


def test_existing_coefficients_full_pose_derivatives_and_hash():
    start, reference, desired = example()
    expanded = expand_image_spline_coordinates(start, desired, reference)
    again = expand_image_spline_coordinates(start, desired, reference.copy())
    assert expanded == again
    assert expanded.expanded_start.coefficient_sha256 != start.coefficient_sha256
    assert expanded.original_coefficient_sha256 == start.coefficient_sha256
    assert expanded.expanded_start.model_sha == start.model_sha
    assert expanded.expanded_start.coordinate_order == start.coordinate_order
    old = CubicHermiteSplineTrajectory(np.asarray(start.knot_times), 3)
    new = CubicHermiteSplineTrajectory(np.asarray(start.knot_times), 6)
    oldq, oldv = old.unpack(np.asarray(start.spline_coefficients))
    newq, newv = new.unpack(np.asarray(expanded.expanded_start.spline_coefficients))
    indices = [desired.index(name) for name in start.free_coordinates]
    np.testing.assert_array_equal(newq[:, indices], oldq)
    np.testing.assert_array_equal(newv[:, indices], oldv)
    times = np.sort(
        np.concatenate((np.asarray(start.knot_times), np.linspace(2, 4, 17)))
    )
    before = old.evaluate(np.asarray(start.spline_coefficients), times)
    after = new.evaluate(np.asarray(expanded.expanded_start.spline_coefficients), times)
    for name in ("q", "v", "a"):
        left = np.tile(reference if name == "q" else np.zeros(44), (len(times), 1))
        right = left.copy()
        left[:, [start.coordinate_order.index(n) for n in start.free_coordinates]] = (
            getattr(before, name)
        )
        right[:, [start.coordinate_order.index(n) for n in desired]] = getattr(
            after, name
        )
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
    reference[:] = 99
    assert expanded.reference_pose[0] == 0
    with pytest.raises(FrozenInstanceError):
        expanded.added_coordinates = ()


@pytest.mark.parametrize(
    "desired",
    [
        ("c01", "c03"),
        ("c03", "c01", "c07"),
        ("c01", "c03", "c07", "unknown"),
        ("c01", "c03", "c07", "c01"),
        ("c01", "c03", "c07", False),
        ["c01", "c03", "c07"],
    ],
)
def test_invalid_coordinate_selection(desired):
    start, reference, _ = example()
    with pytest.raises(ValueError):
        expand_image_spline_coordinates(start, desired, reference)


@pytest.mark.parametrize("kind", ["short", "nan", "bool", "conflict", "matrix"])
def test_invalid_reference_pose(kind):
    start, reference, desired = example()
    if kind == "short":
        reference = reference[:-1]
    if kind == "nan":
        reference[0] = np.nan
    if kind == "bool":
        reference = np.ones(44, dtype=bool)
    if kind == "conflict":
        reference[1] += 1e-5
    if kind == "matrix":
        reference = reference[None, :]
    with pytest.raises(ValueError):
        expand_image_spline_coordinates(start, desired, reference)


def native_problem_with_bounds():
    pytest.importorskip("mujoco")
    from tests.unit.motion_matching.test_historical_image_fit import (
        native_problem,
        ROOT,
    )
    from src.shared.python.workspace.necromatcher_ranges import extract_authored_bounds
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    import hashlib

    native, attachments, camera, inputs = native_problem()
    definition = (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_bytes()
    xml, _ = export_full_body_mjcf(definition)
    ranges = extract_authored_bounds(
        definition,
        "sha256:" + hashlib.sha256(xml.encode()).hexdigest(),
        tuple(native.coordinate_order),
        native.coordinate_units,
    )
    return native, attachments, camera, inputs, ranges


def test_native_wrist_expansion_bounds_and_projection_sensitivity():
    from dataclasses import replace
    from src.shared.python.motion_matching.historical_fit import (
        ImageFitConfig,
        initialize_image_trajectory,
    )
    from src.shared.python.estimation import finite_difference_jacobian
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    native, attachments, camera, inputs, ranges = native_problem_with_bounds()
    wrists = ("LWInputX", "LWInputY", "RWInputX", "RWInputY")
    desired = inputs.free_coordinates + wrists
    start = ImageSplineStart.from_coefficients(
        inputs.knot_times,
        np.zeros(4),
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )
    expanded = expand_image_spline_coordinates(start, desired, inputs.seed)
    config = ImageFitConfig(
        coordinate_bounds=tuple((n, *b) for n, b in ranges.named_bounds.items())
    )
    result = initialize_image_trajectory(
        native,
        attachments,
        camera,
        replace(inputs, free_coordinates=desired),
        config,
        expanded.expanded_start,
    )
    assert result.optimizer_ran is False
    assert result.free_coordinates == desired
    ik = native.create_ik(attachments)
    q = inputs.seed.copy()
    q[native.coordinate_order.index("REInput")] = -0.3
    indices = [native.coordinate_order.index(name) for name in wrists]
    rows = ik.constraint_residual_jacobian(
        q, ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1)
    )
    assert np.linalg.matrix_rank(rows.jacobian[:6, indices]) == 4
    jac = finite_difference_jacobian(
        lambda pose: camera.project(native.marker_positions(pose, attachments)).ravel(),
        q,
    )
    np.testing.assert_array_equal(jac[:, indices], 0)


def test_mixed_boolean_reference_rejected_and_noop_is_truthful():
    start, reference, _ = example()
    mixed = reference.tolist()
    mixed[0] = False
    with pytest.raises(ValueError, match="numeric"):
        expand_image_spline_coordinates(start, start.free_coordinates, mixed)
    same = expand_image_spline_coordinates(start, start.free_coordinates, reference)
    assert same.expanded_start == start
    assert same.added_coordinates == ()
    assert same.original_coefficient_sha256 == same.expanded_start.coefficient_sha256


def test_strict_native_initializer_rejects_expanded_wrist_outside_authored_bounds():
    pytest.importorskip("mujoco")
    from dataclasses import replace
    from src.shared.python.motion_matching.historical_fit import (
        ImageFitConfig,
        initialize_image_trajectory,
    )

    native, attachments, camera, inputs, ranges = native_problem_with_bounds()
    reference = inputs.seed.copy()
    reference[native.coordinate_order.index("LWInputX")] = 2.0
    start = ImageSplineStart.from_coefficients(
        inputs.knot_times,
        np.zeros(4),
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )
    desired = inputs.free_coordinates + ("LWInputX",)
    expanded = expand_image_spline_coordinates(start, desired, reference)
    config = ImageFitConfig(
        coordinate_bounds=tuple(
            (name, *bounds) for name, bounds in ranges.named_bounds.items()
        )
    )
    with pytest.raises(ValueError, match="bounds|feasible|knot"):
        initialize_image_trajectory(
            native,
            attachments,
            camera,
            replace(inputs, seed=reference, free_coordinates=desired),
            config,
            expanded.expanded_start,
        )
