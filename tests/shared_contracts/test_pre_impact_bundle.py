"""Contract tests for the versioned pre-impact bundle (IA-U2, #9703).

The bundle is UpstreamDrift's consumer-side export record for the Tools impact
kernels. These tests pin its fail-closed construction contracts, explicit
absence, power-consistent frame changes, reduced-mode energy projection and
convention compatibility with the pinned Tools providers' public APIs.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.physics.pre_impact_bundle import (
    PRE_IMPACT_BUNDLE_SCHEMA,
    PRE_IMPACT_BUNDLE_VERSION,
    AbsentFieldError,
    FieldOrigin,
    ModalBasis,
    Pose,
    PreImpactBundle,
    PreImpactBundleError,
    Quantity,
    grip_pose_from_delivery_sample,
    project_onto_basis,
    shift_twist_reference,
    shift_wrench_origin,
    twist_to_parent,
    wrench_to_parent,
)
from tests.shared_contracts.test_tools_provider_contracts import (
    _assert_from_tools,
    _fresh_provider_import,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HASH = "a" * 64


def _rotation(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    """Rodrigues rotation used only to build independent test poses."""
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array(
        [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]]
    )
    matrix = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
    return matrix.tolist()


def _q(origin: str, value: Any) -> dict[str, Any]:
    return {"origin": origin, "value": value}


def _valid_payload() -> dict[str, Any]:
    half = math.sqrt(0.5)
    return {
        "schema": PRE_IMPACT_BUNDLE_SCHEMA,
        "version": PRE_IMPACT_BUNDLE_VERSION,
        "units": "SI",
        "provenance": {
            "equipment_id": "driver-synthetic-01",
            "ball_id": "ball-synthetic-01",
            "calibration_id": "calibration-none-synthetic",
            "model_tier": "rigid_head_reduced_modal_shaft",
            "source_hashes": {"delivery": _HASH, "shaft": "b" * 64},
        },
        "timebase": {
            "sample_times_s": [0.0, 0.0005, 0.001, 0.0015],
            "event_time_s": 0.001,
            "interval_s": [0.0005, 0.0015],
            "interpolant": "cubic_hermite",
            "time_uncertainty_s": _q("identified", 2.0e-6),
            "interpolation_error_m": _q("synthetic", 1.0e-5),
        },
        "frames": {
            "head": {
                "parent_frame_id": "world",
                "frame_id": "head",
                "rotation": _rotation((0.2, 1.0, -0.3), 0.7),
                "translation_m": [1.1, 0.05, -0.2],
            },
            "grip": {
                "parent_frame_id": "world",
                "frame_id": "grip",
                "quaternion_wxyz": [half, 0.0, half, 0.0],
                "translation_m": [0.2, 0.9, -0.1],
            },
        },
        "head": {
            "mass_kg": _q("measured", 0.2),
            "inertia_com_kg_m2": _q(
                "identified",
                [
                    [4.0e-4, 1.0e-5, -2.0e-5],
                    [1.0e-5, 5.0e-4, 3.0e-5],
                    [-2.0e-5, 3.0e-5, 3.0e-4],
                ],
            ),
            "linear_velocity_mps": _q("measured", [45.0, -2.0, 1.5]),
            "angular_velocity_rad_s": _q("measured", [3.0, -12.0, 40.0]),
            "contact_offset_m": _q("prescribed", [0.02, 0.005, -0.003]),
            "contact_normal": _q("prescribed", [1.0, 0.0, 0.0]),
            "face_curvature_1_m": _q("absent", None),
        },
        "ball": {
            "position_m": _q("measured", [1.13, 0.02, -0.2]),
            "velocity_mps": _q("measured", [0.0, 0.0, 0.0]),
            "spin_rad_s": _q("absent", None),
        },
        "shaft": {
            "basis": {
                "basis_id": "synthetic-cantilever-3dof",
                "version": "1",
                "normalization": "mass",
                "dimension": 2,
                "generalized_mass": [[1.0, 0.0], [0.0, 1.0]],
                "generalized_stiffness": [[900.0, 0.0], [0.0, 25000.0]],
            },
            "basis_id": "synthetic-cantilever-3dof",
            "basis_version": "1",
            "amplitudes": _q("synthetic", [0.01, -0.002]),
            "velocities": _q("synthetic", [0.3, 0.05]),
            "axial_stations_m": [0.0, 0.5, 1.0],
            "axial_force_n": _q("prescribed", [120.0, 80.0, 40.0]),
        },
        "hands": [
            {
                "hand": "lead",
                "frame_id": "grip",
                "origin_m": [0.0, 0.0, 0.05],
                "force_n": _q("identified", [10.0, -40.0, 150.0]),
                "moment_n_m": _q("identified", [1.5, 0.3, -2.0]),
                "stiffness": _q("absent", None),
                "damping": _q("absent", None),
                "assumptions": ["rigid hand-grip attachment point declared"],
            },
            {
                "hand": "trail",
                "frame_id": "grip",
                "origin_m": [0.0, 0.0, 0.14],
                "force_n": _q("absent", None),
                "moment_n_m": _q("absent", None),
                "stiffness": _q("absent", None),
                "damping": _q("absent", None),
                "assumptions": ["trail-hand wrench not identified"],
            },
        ],
        "constraints": ["free detached head during contact interval"],
    }


def _bundle(payload: dict[str, Any] | None = None) -> PreImpactBundle:
    return PreImpactBundle.from_dict(payload or _valid_payload())


# --- round trips -------------------------------------------------------------


def test_valid_bundle_round_trips_through_dict_and_json() -> None:
    bundle = _bundle()
    assert bundle.schema == PRE_IMPACT_BUNDLE_SCHEMA
    assert bundle.version == PRE_IMPACT_BUNDLE_VERSION
    canonical = bundle.to_json()
    again = PreImpactBundle.from_json(canonical)
    assert again.to_json() == canonical
    assert PreImpactBundle.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()
    assert json.loads(canonical)["version"] == 1


def test_unknown_version_schema_and_fields_are_refused() -> None:
    for mutate in (
        lambda p: p.__setitem__("version", 2),
        lambda p: p.__setitem__("schema", "other.bundle"),
        lambda p: p.__setitem__("extra", 1),
        lambda p: p["head"].__setitem__("extra", 1),
    ):
        payload = _valid_payload()
        mutate(payload)
        with pytest.raises(PreImpactBundleError):
            _bundle(payload)


def test_arrays_are_copied_and_read_only() -> None:
    payload = _valid_payload()
    velocity = np.array([45.0, -2.0, 1.5])
    payload["head"]["linear_velocity_mps"] = _q("measured", velocity)
    bundle = _bundle(payload)
    velocity[0] = 0.0
    stored = bundle.head.linear_velocity_mps.value
    assert stored[0] == 45.0
    with pytest.raises(ValueError):
        stored[0] = 1.0


def test_every_field_declares_an_origin() -> None:
    bundle = _bundle()
    origins = bundle.field_origins()
    assert origins["head.mass_kg"] is FieldOrigin.MEASURED
    assert origins["ball.spin_rad_s"] is FieldOrigin.ABSENT
    assert origins["shaft.amplitudes"] is FieldOrigin.SYNTHETIC
    assert origins["hands.trail.force_n"] is FieldOrigin.ABSENT
    assert all(isinstance(value, FieldOrigin) for value in origins.values())


# --- invalid inputs ----------------------------------------------------------


def _set(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    payload = _valid_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


_INVALID = {
    "bad_units": (("units",), "imperial"),
    "non_finite_velocity": (
        ("head", "linear_velocity_mps"),
        _q("measured", [math.inf, 0.0, 0.0]),
    ),
    "nan_mass": (("head", "mass_kg"), _q("measured", math.nan)),
    "negative_mass": (("head", "mass_kg"), _q("measured", -0.2)),
    "string_coercion": (("head", "mass_kg"), _q("measured", "0.2")),
    "non_monotonic_time": (
        ("timebase", "sample_times_s"),
        [0.0, 0.001, 0.0005, 0.0015],
    ),
    "event_outside_interval": (("timebase", "event_time_s"), 0.0016),
    "interval_extrapolates": (("timebase", "interval_s"), [0.0005, 0.002]),
    "reflection_rotation": (
        ("frames", "head", "rotation"),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    ),
    "non_orthonormal_rotation": (
        ("frames", "head", "rotation"),
        [[1.001, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ),
    "non_unit_quaternion": (
        ("frames", "grip", "quaternion_wxyz"),
        [1.0, 0.1, 0.0, 0.0],
    ),
    "wrong_parent_frame": (("frames", "grip", "parent_frame_id"), "head"),
    "asymmetric_inertia": (
        ("head", "inertia_com_kg_m2"),
        _q("identified", [[4e-4, 1e-4, 0], [0, 5e-4, 0], [0, 0, 3e-4]]),
    ),
    "indefinite_inertia": (
        ("head", "inertia_com_kg_m2"),
        _q("identified", [[4e-4, 0, 0], [0, -5e-4, 0], [0, 0, 3e-4]]),
    ),
    "triangle_violation": (
        ("head", "inertia_com_kg_m2"),
        _q("identified", [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 5e-4]]),
    ),
    "non_unit_normal": (("head", "contact_normal"), _q("prescribed", [2.0, 0.0, 0.0])),
    "mismatched_basis_id": (("shaft", "basis_id"), "other-basis"),
    "mismatched_basis_version": (("shaft", "basis_version"), "2"),
    "amplitude_dimension": (("shaft", "amplitudes"), _q("synthetic", [0.01])),
    "velocity_dimension": (("shaft", "velocities"), _q("synthetic", [0.1, 0.2, 0.3])),
    "non_spd_modal_mass": (
        ("shaft", "basis", "generalized_mass"),
        [[1.0, 0.0], [0.0, 0.0]],
    ),
    "axial_station_mismatch": (("shaft", "axial_force_n"), _q("prescribed", [1.0])),
    "unknown_origin": (("head", "mass_kg"), _q("guessed", 0.2)),
    "absent_with_value": (("head", "mass_kg"), _q("absent", 0.2)),
    "present_without_value": (("head", "mass_kg"), _q("measured", None)),
    "bad_hash": (("provenance", "source_hashes"), {"delivery": "xyz"}),
    "absent_required_mass": (("head", "mass_kg"), _q("absent", None)),
}


#: The classified reason each invalid case must be refused for.
_EXPECTED_CODE = {
    "absent_required_mass": "absent_required",
    "absent_with_value": "origin",
    "amplitude_dimension": "basis_mismatch",
    "asymmetric_inertia": "asymmetric",
    "axial_station_mismatch": "shape",
    "bad_hash": "hash",
    "bad_units": "units",
    "event_outside_interval": "event_outside_interval",
    "indefinite_inertia": "not_positive_definite",
    "interval_extrapolates": "extrapolation",
    "mismatched_basis_id": "basis_mismatch",
    "mismatched_basis_version": "basis_mismatch",
    "nan_mass": "non_finite",
    "negative_mass": "out_of_range",
    "non_finite_velocity": "non_finite",
    "non_monotonic_time": "non_monotonic_time",
    "non_orthonormal_rotation": "improper_rotation",
    "non_spd_modal_mass": "not_positive_definite",
    "non_unit_normal": "non_unit_vector",
    "non_unit_quaternion": "non_unit_quaternion",
    "present_without_value": "origin",
    "reflection_rotation": "improper_rotation",
    "string_coercion": "type",
    "triangle_violation": "triangle_inequality",
    "unknown_origin": "origin",
    "velocity_dimension": "basis_mismatch",
    "wrong_parent_frame": "frame_mismatch",
}


@pytest.mark.parametrize("case", sorted(_INVALID))
def test_invalid_inputs_are_refused_with_classified_error(case: str) -> None:
    path, value = _INVALID[case]
    with pytest.raises(PreImpactBundleError) as info:
        _bundle(_set(path, value))
    assert info.value.code == _EXPECTED_CODE[case]


def test_duplicate_hand_is_refused() -> None:
    payload = _valid_payload()
    payload["hands"][1]["hand"] = "lead"
    with pytest.raises(PreImpactBundleError):
        _bundle(payload)


def test_invariants_hold_under_python_optimize() -> None:
    script = (
        "from tests.shared_contracts.test_pre_impact_bundle import _valid_payload\n"
        "from src.shared.python.physics.pre_impact_bundle import "
        "PreImpactBundle, PreImpactBundleError\n"
        "p = _valid_payload(); p['timebase']['event_time_s'] = 9.0\n"
        "try:\n    PreImpactBundle.from_dict(p)\n"
        "except PreImpactBundleError:\n    print('refused')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.stdout.strip().endswith("refused"), result.stderr[-2000:]


# --- explicit absence --------------------------------------------------------


def test_absent_field_access_raises_and_is_never_zero() -> None:
    bundle = _bundle()
    spin = bundle.ball.spin_rad_s
    assert spin.is_absent
    with pytest.raises(AbsentFieldError):
        _ = spin.value
    with pytest.raises(AbsentFieldError):
        np.asarray(spin)
    with pytest.raises(AbsentFieldError):
        float(bundle.head.face_curvature_1_m)
    with pytest.raises(AbsentFieldError):
        bundle.hand_wrench_in("trail", "world")


def test_absent_elastic_state_is_not_zero_energy() -> None:
    payload = _valid_payload()
    payload["shaft"]["amplitudes"] = _q("absent", None)
    bundle = _bundle(payload)
    with pytest.raises(AbsentFieldError):
        bundle.shaft.modal_energy_j()


def test_quantity_constructor_enforces_origin_value_pairing() -> None:
    assert Quantity.absent().origin is FieldOrigin.ABSENT
    with pytest.raises(PreImpactBundleError):
        Quantity(FieldOrigin.ABSENT, np.zeros(3))
    with pytest.raises(PreImpactBundleError):
        Quantity(FieldOrigin.MEASURED, None)


# --- frames and wrench power -------------------------------------------------


def _power(force, moment, linear, angular) -> float:
    return float(np.dot(force, linear) + np.dot(moment, angular))


def test_wrench_power_is_invariant_across_world_head_and_grip() -> None:
    bundle = _bundle()
    rng = np.random.default_rng(9703)
    force, moment = rng.normal(size=3) * 100, rng.normal(size=3) * 5
    linear, angular = rng.normal(size=3) * 20, rng.normal(size=3) * 30
    reference = _power(force, moment, linear, angular)
    for source in ("world", "head", "grip"):
        for target in ("world", "head", "grip"):
            pose = bundle.pose_between(target, source)
            f2, m2 = wrench_to_parent(pose, force, moment)
            v2, w2 = twist_to_parent(pose, linear, angular)
            assert math.isclose(
                _power(f2, m2, v2, w2), reference, rel_tol=1e-12, abs_tol=0
            )


def test_moment_about_shifted_origin_preserves_power() -> None:
    rng = np.random.default_rng(1)
    force, moment = rng.normal(size=3) * 80, rng.normal(size=3) * 3
    linear, angular = rng.normal(size=3) * 15, rng.normal(size=3) * 25
    old, new = rng.normal(size=3), rng.normal(size=3)
    f2, m2 = shift_wrench_origin(force, moment, old, new)
    v2, w2 = shift_twist_reference(linear, angular, old, new)
    np.testing.assert_allclose(m2, moment + np.cross(old - new, force), rtol=1e-14)
    assert math.isclose(
        _power(f2, m2, v2, w2),
        _power(force, moment, linear, angular),
        rel_tol=1e-12,
        abs_tol=0,
    )


def test_hand_wrench_transform_matches_manual_composition() -> None:
    bundle = _bundle()
    hand = bundle.hand("lead")
    grip = bundle.pose_between("world", "grip")
    force, moment = bundle.hand_wrench_in("lead", "world")
    rotation = np.asarray(grip.rotation)
    at_grip_origin = hand.moment_n_m.value + np.cross(hand.origin_m, hand.force_n.value)
    expected_force = rotation @ hand.force_n.value
    expected_moment = rotation @ at_grip_origin + np.cross(
        grip.translation_m, expected_force
    )
    np.testing.assert_allclose(force, expected_force, rtol=1e-13)
    np.testing.assert_allclose(moment, expected_moment, rtol=1e-13)


def test_pose_inverse_and_composition_round_trip() -> None:
    bundle = _bundle()
    head_from_grip = bundle.pose_between("head", "grip")
    identity = head_from_grip.compose(head_from_grip.inverse())
    np.testing.assert_allclose(identity.rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(identity.translation_m, 0.0, atol=1e-14)
    with pytest.raises(PreImpactBundleError):
        bundle.pose_between("world", "ball")


# --- reduced-mode projection -------------------------------------------------


def _full_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.0]])
    stiffness = np.array(
        [[4000.0, -1500.0, 0.0], [-1500.0, 3000.0, -800.0], [0.0, -800.0, 900.0]]
    )
    # Mass-normalised modes of K x = w^2 M x via the Cholesky factor M = L L^T.
    inverse_factor = np.linalg.inv(np.linalg.cholesky(mass))
    _, vectors = np.linalg.eigh(inverse_factor @ stiffness @ inverse_factor.T)
    shapes = inverse_factor.T @ vectors[:, :2]
    return mass, stiffness, shapes


def _basis(shapes: np.ndarray, mass: np.ndarray, stiffness: np.ndarray) -> ModalBasis:
    return ModalBasis(
        basis_id="synthetic-cantilever-3dof",
        version="1",
        normalization="mass",
        dimension=2,
        generalized_mass=shapes.T @ mass @ shapes,
        generalized_stiffness=shapes.T @ stiffness @ shapes,
    )


def test_in_span_projection_preserves_represented_energy() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    amplitudes, velocities = np.array([0.02, -0.004]), np.array([0.5, 0.1])
    result = project_onto_basis(
        basis, shapes, mass, stiffness, shapes @ amplitudes, shapes @ velocities
    )
    np.testing.assert_allclose(result.amplitudes, amplitudes, rtol=1e-12)
    np.testing.assert_allclose(result.velocities, velocities, rtol=1e-12)
    assert result.displacement_residual < 1e-12
    assert result.velocity_residual < 1e-12
    full = result.full_kinetic_energy_j + result.full_potential_energy_j
    represented = (
        result.represented_kinetic_energy_j + result.represented_potential_energy_j
    )
    assert math.isclose(represented, full, rel_tol=1e-12)
    expected = 0.5 * velocities @ velocities + 0.5 * amplitudes @ (
        basis.generalized_stiffness @ amplitudes
    )
    assert math.isclose(represented, expected, rel_tol=1e-12)


def test_out_of_span_projection_reports_residual() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    state = np.array([0.01, 0.0, 0.03])
    result = project_onto_basis(basis, shapes, mass, stiffness, state, state)
    assert result.displacement_residual > 1e-3
    assert result.represented_kinetic_energy_j <= result.full_kinetic_energy_j


def test_projection_refuses_inconsistent_basis() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    with pytest.raises(PreImpactBundleError):
        project_onto_basis(
            basis, shapes * 2.0, mass, stiffness, shapes[:, 0], shapes[:, 0]
        )


def test_bundle_modal_energy_uses_declared_quadratic_form() -> None:
    bundle = _bundle()
    q, qd = np.array([0.01, -0.002]), np.array([0.3, 0.05])
    expected = 0.5 * qd @ qd + 0.5 * (900.0 * q[0] ** 2 + 25000.0 * q[1] ** 2)
    assert math.isclose(bundle.shaft.modal_energy_j(), expected, rel_tol=1e-14)


# --- Tools provider convention compatibility ---------------------------------


def test_head_record_feeds_tools_rigid_contact_body() -> None:
    bundle = _bundle()
    with _fresh_provider_import("golf_club"):
        mobility = importlib.import_module("golf_club.impact_mobility")
        _assert_from_tools(Path(mobility.__file__))
        body = mobility.RigidContactBody(**bundle.head.tools_contact_body_fields())
        normal = tuple(bundle.head.contact_normal.value)
        assert mobility.normal_effective_mass(body, normal) > 0.0


def test_twist_convention_matches_tools_grip_state_transform() -> None:
    bundle = _bundle()
    grip = bundle.pose_between("world", "grip")
    linear, angular = np.array([1.0, -2.0, 0.5]), np.array([3.0, 0.2, -1.0])
    with _fresh_provider_import("golf_club"):
        types = importlib.import_module("golf_club.types")
        grip_module = importlib.import_module("golf_club.grip_impedance")
        _assert_from_tools(Path(grip_module.__file__))
        transform = types.RigidTransform(
            from_frame_id="grip",
            to_frame_id="world",
            rotation=np.asarray(grip.rotation),
            translation_m=np.asarray(grip.translation_m),
        )
        state = grip_module.GripPortState(
            "grip", np.zeros(6), np.concatenate([linear, angular]), np.zeros(6)
        )
        mapped = np.asarray(grip_module.transform_grip_state(state, transform).velocity)
    v2, w2 = twist_to_parent(grip, linear, angular)
    np.testing.assert_allclose(mapped, np.concatenate([v2, w2]), rtol=1e-13, atol=1e-15)


def test_grip_pose_adapts_tools_delivery_sample() -> None:
    with _fresh_provider_import("swing_sim"):
        trajectory = importlib.import_module("swing_sim.delivery_interchange")
        _assert_from_tools(Path(trajectory.__file__))
        half = math.sqrt(0.5)
        sample = trajectory.TrajectorySample(
            time_s=0.001,
            position_m=(0.2, 0.9, -0.1),
            quaternion_wxyz=(half, 0.0, half, 0.0),
            linear_velocity_mps=(30.0, 0.0, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 40.0),
        )
        pose = grip_pose_from_delivery_sample(sample, world_frame_id="world")
        np.testing.assert_allclose(pose.rotation, sample.rotation_matrix(), atol=1e-15)
        np.testing.assert_allclose(pose.translation_m, sample.position_m)
    assert isinstance(pose, Pose)
    assert pose.frame_id == "grip"


def test_modal_state_immutability() -> None:
    bundle = _bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.shaft.basis_id = "x"  # type: ignore[misc]
    assert copy.deepcopy(bundle).to_json() == bundle.to_json()
