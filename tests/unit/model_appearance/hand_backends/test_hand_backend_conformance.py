"""Conformance contract every grip hand visual backend must pass (ADR-0057 A4).

The checks are structural: identity, mesh validity, frame, skin colour, shaft
roll, resizing, handedness, determinism and registry hygiene. They run for
every registered backend (disabled ones included) and for a test-only
reference backend, so the contract is exercised before any real backend
lands. Appearance is judged separately, from renders (epic #15150).
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends import (
    HandJointAngles,
    HandSide,
    HandVisualBackend,
    HandVisualResult,
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends import registry

pytestmark = pytest.mark.unit

GRIP_RADIUS_M = 0.012
ROLL_DEG = 20.0
ROLL_ATOL_M = 1e-6
HANDEDNESS_REL_TOL = 0.05
MIN_FACE_AREA_M2 = 1e-14
BIGGER_HAND_SCALE = 1.15
SIDES: tuple[HandSide, ...] = ("lead", "trail")
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

REFERENCE_NAME = "conformance_reference"
REFERENCE_SKIN_RGB = (0.8, 0.6, 0.5)


def _box(lo: np.ndarray, hi: np.ndarray) -> Mesh:
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return Mesh(corners, faces)


def _reference_builder(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
) -> HandVisualResult:
    """A palm block beside the grip, sized, mirrored and rolled per the contract."""
    length = grip_spec.hand_size.length_m
    breadth = grip_spec.hand_size.breadth_m
    side_sign = 1.0 if side == "lead" else -1.0
    lo = np.array([grip_radius_m, 0.0, 0.0])
    hi = np.array([grip_radius_m + 0.25 * breadth, 0.6 * length, side_sign * breadth])
    mesh = _box(np.minimum(lo, hi), np.maximum(lo, hi))
    vertices = mesh.vertices.copy()
    if grip_spec.handedness == "left":
        vertices[:, 1] *= -1.0
    roll = apply_shaft_roll(
        np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, resolved_shaft_roll_deg(grip_spec, side)
    )[:3, :3]
    faces = mesh.faces if grip_spec.handedness != "left" else mesh.faces[:, ::-1]
    return HandVisualResult(
        backend=REFERENCE_NAME,
        side=side,
        mesh=Mesh(vertices @ roll.T, faces.copy()),
        skin_rgb=skin_rgb or REFERENCE_SKIN_RGB,
        grip_radius_m=grip_radius_m,
    )


REFERENCE_BACKEND = HandVisualBackend(
    name=REFERENCE_NAME,
    builder=_reference_builder,
    description="Test-only palm block exercising the conformance contract",
)


def _backends() -> list[HandVisualBackend]:
    registered = [
        registry.get_hand_visual_backend(name)
        for name in list_hand_visual_backends(include_disabled=True)
    ]
    return [REFERENCE_BACKEND, *registered]


def _build(
    backend: HandVisualBackend, spec: GripSpec, side: HandSide, **kwargs: object
) -> HandVisualResult:
    return backend.builder(spec, side, None, grip_radius_m=GRIP_RADIUS_M, **kwargs)


def _roll_spec(spec: GripSpec, side: HandSide, roll_deg: float) -> GripSpec:
    hand = replace(getattr(spec, side), shaft_roll_deg=roll_deg)
    return replace(spec, **{side: hand})


def _extent(mesh: Mesh) -> np.ndarray:
    return mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)


BACKENDS = _backends()
IDS = [backend.name for backend in BACKENDS]


@pytest.fixture(params=BACKENDS, ids=IDS)
def backend(request: pytest.FixtureRequest) -> HandVisualBackend:
    return request.param


@pytest.mark.parametrize("side", SIDES)
def test_identity(backend: HandVisualBackend, side: HandSide) -> None:
    result = _build(backend, GripSpec.legacy(), side)
    assert isinstance(result, HandVisualResult)
    assert result.backend == backend.name
    assert result.side == side
    assert result.grip_radius_m == pytest.approx(GRIP_RADIUS_M)


@pytest.mark.parametrize("side", SIDES)
def test_mesh_is_valid(backend: HandVisualBackend, side: HandSide) -> None:
    mesh = _build(backend, GripSpec.legacy(), side).mesh
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4
    assert np.all(np.isfinite(vertices))
    assert faces.ndim == 2 and faces.shape[1] == 3 and len(faces) >= 1
    assert np.issubdtype(faces.dtype, np.integer)
    assert faces.min() >= 0 and faces.max() < len(vertices)
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    assert np.all(areas > MIN_FACE_AREA_M2), "degenerate (zero-area) faces"
    assert np.all(_extent(mesh) > 0.0)


@pytest.mark.parametrize("side", SIDES)
def test_mesh_sits_beside_the_grip(backend: HandVisualBackend, side: HandSide) -> None:
    spec = GripSpec.legacy()
    centroid = _build(backend, spec, side).mesh.vertices.mean(axis=0)
    axis = APPEARANCE_GRIP_SHAFT_AXIS
    radial = centroid - np.dot(centroid, axis) * axis
    assert np.linalg.norm(radial) < spec.hand_size.length_m


@pytest.mark.parametrize("side", SIDES)
def test_skin_colour_default_and_override(
    backend: HandVisualBackend, side: HandSide
) -> None:
    default = _build(backend, GripSpec.legacy(), side).skin_rgb
    assert len(default) == 3 and all(0.0 <= c <= 1.0 for c in default)
    chosen = (0.31, 0.22, 0.17)
    assert (
        tuple(_build(backend, GripSpec.legacy(), side, skin_rgb=chosen).skin_rgb)
        == chosen
    )


@pytest.mark.parametrize("side", SIDES)
def test_shaft_roll_is_a_rigid_rotation_per_hand(
    backend: HandVisualBackend, side: HandSide
) -> None:
    base_spec = GripSpec.legacy()
    rolled_spec = _roll_spec(base_spec, side, ROLL_DEG)
    base = _build(backend, base_spec, side).mesh
    rolled = _build(backend, rolled_spec, side).mesh
    rotation = apply_shaft_roll(
        np.eye(3),
        APPEARANCE_GRIP_SHAFT_AXIS,
        resolved_shaft_roll_deg(rolled_spec, side),
    )[:3, :3]
    np.testing.assert_allclose(
        rolled.vertices, base.vertices @ rotation.T, atol=ROLL_ATOL_M
    )
    other: HandSide = "trail" if side == "lead" else "lead"
    np.testing.assert_allclose(
        _build(backend, rolled_spec, other).mesh.vertices,
        _build(backend, base_spec, other).mesh.vertices,
        atol=ROLL_ATOL_M,
    )


@pytest.mark.parametrize("side", SIDES)
def test_bigger_hand_size_gives_a_bigger_hand(
    backend: HandVisualBackend, side: HandSide
) -> None:
    spec = GripSpec.legacy()
    size = spec.hand_size
    bigger = replace(
        spec,
        hand_size=replace(
            size,
            length_m=size.length_m * BIGGER_HAND_SCALE,
            breadth_m=size.breadth_m * BIGGER_HAND_SCALE,
        ),
    )
    assert (
        _extent(_build(backend, bigger, side).mesh).max()
        > _extent(_build(backend, spec, side).mesh).max()
    )


def test_left_handed_lead_hand_matches_right_handed_extent(
    backend: HandVisualBackend,
) -> None:
    right = _extent(_build(backend, GripSpec.legacy(), "lead").mesh)
    left = _extent(_build(backend, GripSpec.legacy(handedness="left"), "lead").mesh)
    np.testing.assert_allclose(np.sort(left), np.sort(right), rtol=HANDEDNESS_REL_TOL)


def test_builds_are_deterministic(backend: HandVisualBackend) -> None:
    first = _build(backend, GripSpec.legacy(), "lead").mesh
    second = _build(backend, GripSpec.legacy(), "lead").mesh
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, second.faces)


def test_registry_hygiene(backend: HandVisualBackend) -> None:
    assert NAME_PATTERN.match(backend.name)
    if backend.name == "myohand":
        assert backend.default_enabled is False
