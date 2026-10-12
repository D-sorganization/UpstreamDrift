"""Native body-frame binding and rejection of placeholder physical evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
        NativeMarkerGeometry,
    )

from src.shared.python.motion_matching.pipeline.plants.opensim import (
    OpensimMatchingPlant,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ["poses", "markers", "closure", "step", "ik"])
def test_metadata_only_plant_cannot_supply_physical_evidence(operation: str) -> None:
    plant = OpensimMatchingPlant({"coordinate_order": ["angle"]})
    q = np.array([0.4])
    attachments = {"observed": ("/bodyset/segment", (0.1, 0.0, 0.0))}
    with pytest.raises(NotImplementedError):
        if operation == "poses":
            plant.frame_poses(attachments, q)
        elif operation == "markers":
            plant.marker_positions(q, attachments)
        elif operation == "closure":
            plant.closure_residuals(q)
        elif operation == "step":
            plant.step(q, q, q, 0.01)
        else:
            plant.create_ik(attachments)


def _native(
    path: Path, coordinates: tuple[str, ...] = ("/jointset/pin/angle",)
) -> NativeMarkerGeometry:
    from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
        NativeMarkerGeometry,
    )

    return NativeMarkerGeometry(path, coordinates)


def test_native_body_and_offset_frame_follow_actual_coordinates(
    native_pin: Path,
) -> None:
    provider = _native(native_pin)
    bindings = {"observed": ("/bodyset/segment/skin_frame", (0.1, 0.0, 0.0))}
    q = np.array([0.6])
    poses = provider.frame_poses(bindings, q)
    rot, trans = poses["/bodyset/segment/skin_frame"]
    expected_rotation = np.array(
        [[np.cos(0.6), -np.sin(0.6), 0], [np.sin(0.6), np.cos(0.6), 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(rot, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(
        trans, np.array([1, 0, 0]) + expected_rotation @ [0.2, 0, 0], atol=1e-12
    )
    observed = provider.marker_positions(q, bindings)
    np.testing.assert_allclose(observed[0], rot @ [0.1, 0, 0] + trans, atol=1e-12)
    np.testing.assert_allclose(provider.achieved_coordinates, q, atol=1e-10)
    provider.marker_positions(np.array([-0.4]), bindings)
    np.testing.assert_allclose(
        provider.marker_positions(q, bindings), observed, atol=1e-12
    )
    assert provider.source_sha256 == hashlib.sha256(native_pin.read_bytes()).hexdigest()
    assert len(provider.loaded_sha256) == 64
    assert provider.runtime_version


@pytest.mark.parametrize(
    "frame", ["segment", "/bodyset/torso", "/jointset/pin/angle", "/ground"]
)
def test_only_explicit_existing_body_frames_are_admitted(
    native_pin: Path, frame: str
) -> None:
    provider = _native(native_pin)
    with pytest.raises(ValueError):
        provider.marker_positions(np.array([0.1]), {"observed": (frame, (0, 0, 0))})


@pytest.mark.parametrize("q", [np.array([np.nan]), np.array([0, 0]), np.array([2.0])])
def test_native_geometry_rejects_invalid_coordinate_requests(
    native_pin: Path, q: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _native(native_pin).marker_positions(q, {"m": ("/bodyset/segment", (0, 0, 0))})


@pytest.mark.parametrize("policy", ["locked", "prescribed"])
def test_native_geometry_rejects_uncontrollable_requested_coordinate(
    native_pin: Path, policy: str
) -> None:
    import opensim as osim

    model = osim.Model(str(native_pin))
    coordinate = model.updCoordinateSet().get("angle")
    if policy == "locked":
        coordinate.setDefaultLocked(True)
    else:
        coordinate.setDefaultIsPrescribed(True)
        coordinate.setPrescribedFunction(osim.Constant(0.0))
    model.printToXML(str(native_pin))
    with pytest.raises(ValueError, match="independent"):
        _native(native_pin)


def test_matching_plant_uses_native_geometry_without_claiming_dynamics(
    native_pin: Path,
) -> None:
    plant = OpensimMatchingPlant(
        {"coordinate_order": ["/jointset/pin/angle"]}, native_model_path=native_pin
    )
    offsets = {"m": ("/bodyset/segment", (0.3, 0, 0))}
    expected = np.array([[1 + 0.3 * np.cos(0.4), 0.3 * np.sin(0.4), 0]])
    np.testing.assert_allclose(
        plant.marker_positions(np.array([0.4]), offsets), expected, atol=1e-12
    )
    poses = plant.create_ik(offsets).pose_fn(np.array([0.4]))
    np.testing.assert_allclose(poses["/bodyset/segment"][1], [1, 0, 0], atol=1e-12)
    with pytest.raises(NotImplementedError):
        plant.step(np.array([0.4]), np.zeros(1), np.zeros(1), 0.01)


def test_shared_ik_recovers_motion_through_actual_native_pose_callback(
    native_pin: Path,
) -> None:
    from src.shared.python.motion_matching.full_body_ik import (
        solve_full_body_ik_trajectory,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    native = _native(native_pin)
    offsets = {
        "a": ("/bodyset/segment", (0.2, 0, 0)),
        "b": ("/bodyset/segment", (0, 0.2, 0)),
        "c": ("/bodyset/segment", (0.2, 0.2, 0)),
    }
    expected_q = np.array([0.0, 0.2, 0.4])
    points = []
    for angle in expected_q:
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        points.append([rotation @ offset + [1, 0, 0] for _, offset in offsets.values()])
    capture = TourCapture(
        np.array([0.0, 0.01, 0.02]),
        tuple(offsets),
        np.asarray(points),
        np.ones((3, 3), dtype=bool),
    )
    q = solve_full_body_ik_trajectory(
        lambda q: native.frame_poses(offsets, q),
        offsets,
        capture,
        np.zeros(1),
        reg_weight=0,
        max_nfev=50,
    )
    np.testing.assert_allclose(q[:, 0], expected_q, atol=1e-8)


def test_actual_native_coupler_follows_dependent_coordinate_and_checks_its_range(
    native_pin: Path,
) -> None:
    import opensim as osim

    model = osim.Model(str(native_pin))
    body = osim.Body("second", 1.0, osim.Vec3(0), osim.Inertia(0.02))
    model.addBody(body)
    joint = osim.PinJoint(
        "second_pin",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint.updCoordinate().setName("dependent")
    joint.updCoordinate().setRangeMin(-1.5)
    joint.updCoordinate().setRangeMax(1.5)
    model.addJoint(joint)
    constraint = osim.CoordinateCouplerConstraint()
    independent = osim.ArrayStr()
    independent.append("angle")
    constraint.setIndependentCoordinateNames(independent)
    constraint.setDependentCoordinateName("dependent")
    constraint.setFunction(osim.LinearFunction(2.0, 0.0))
    model.addConstraint(constraint)
    model.finalizeConnections()
    model.printToXML(str(native_pin))
    with pytest.raises(ValueError, match="independent"):
        _native(native_pin, ("/jointset/second_pin/dependent",))
    provider = _native(native_pin)
    result = provider.marker_positions(
        np.array([0.6]), {"m": ("/bodyset/second", (1, 0, 0))}
    )
    np.testing.assert_allclose(result[0], [np.cos(1.2), np.sin(1.2), 0], atol=1e-9)
    with pytest.raises(ValueError, match="range"):
        provider.marker_positions(
            np.array([0.9]), {"m": ("/bodyset/second", (1, 0, 0))}
        )


def test_native_assembly_cannot_silently_change_requested_values(
    native_pin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import opensim as osim

    assemble = osim.Model.assemble

    def changing_assembly(model: Any, state: Any) -> None:
        assemble(model, state)
        model.updCoordinateSet().get("angle").setValue(state, -0.2, False)

    provider = _native(native_pin)
    monkeypatch.setattr(osim.Model, "assemble", changing_assembly)
    with pytest.raises(ValueError, match="achieve"):
        provider.marker_positions(
            np.array([0.4]), {"m": ("/bodyset/segment", (0.1, 0, 0))}
        )


def test_native_coordinate_bounds_are_source_values_and_copies(
    native_pin: Path,
) -> None:
    native = _native(native_pin)
    low, high = native.coordinate_bounds
    np.testing.assert_array_equal(low, [-1.5])
    np.testing.assert_array_equal(high, [1.5])
    low[0] = -99
    high[0] = 99
    np.testing.assert_array_equal(native.coordinate_bounds, [[-1.5], [1.5]])
