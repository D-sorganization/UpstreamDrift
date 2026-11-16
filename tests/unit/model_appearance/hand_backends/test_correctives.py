"""Corrective shapes on synthetic meshes (epic #15150): no hand asset needed."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.correctives import (
    FoldRelaxSettings,
    PadConformSettings,
    PressSettings,
    conform_pads,
    distance_gradient,
    lift_out,
    yield_to_points,
    press_flat,
    relax_folds,
    sharpened_faces,
    vertex_neighbours,
)

pytestmark = pytest.mark.unit

_N = 7
_CELL = 0.004


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """A flat ``_N`` x ``_N`` sheet in the xy-plane, wound to face ``+z``."""
    xs, ys = np.meshgrid(np.arange(_N) * _CELL, np.arange(_N) * _CELL, indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(_N * _N)])
    faces = []
    for i in range(_N - 1):
        for j in range(_N - 1):
            a, b, c, d = (
                i * _N + j,
                (i + 1) * _N + j,
                (i + 1) * _N + j + 1,
                i * _N + j + 1,
            )
            faces += [(a, b, c), (a, c, d)]
    return vertices, np.array(faces, dtype=np.int64)


def _facing_down(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.flatnonzero(normal[:, 2] <= 0.0)


def _folded_grid() -> tuple[np.ndarray, np.ndarray, int]:
    vertices, faces = _grid()
    centre = (_N // 2) * _N + _N // 2
    vertices[centre, 0] += 1.6 * _CELL  # dragged across its neighbour
    return vertices, faces, centre


def test_neighbours_are_the_edge_connected_vertices() -> None:
    _, faces = _grid()
    neighbours = vertex_neighbours(faces, _N * _N)
    assert sorted(neighbours[0].tolist()) == [1, _N, _N + 1]
    inner = (_N // 2) * _N + _N // 2
    assert len(neighbours[inner]) == 6
    assert all(inner in neighbours[j] for j in neighbours[inner])


def test_relaxing_unfolds_the_sheet_and_leaves_the_rest_alone() -> None:
    vertices, faces, centre = _folded_grid()

    def folded(v: np.ndarray) -> np.ndarray:
        return _facing_down(v, faces)

    assert len(folded(vertices)) > 0
    relaxed = relax_folds(vertices, faces, folded, FoldRelaxSettings(8, 1, 0.5))
    assert len(folded(relaxed)) == 0
    moved = np.flatnonzero(np.abs(relaxed - vertices).max(axis=1) > 0.0)
    assert centre in moved
    # Nothing farther than a few cells from the fold moves.
    reach = np.abs(vertices[moved] - vertices[centre]).max()
    assert reach <= 4.0 * _CELL
    assert len(moved) < len(vertices)


def test_fold_relaxation_is_switched_off_by_zero_iterations() -> None:
    vertices, faces, _ = _folded_grid()
    out = relax_folds(
        vertices, faces, lambda v: _facing_down(v, faces), FoldRelaxSettings(0, 1, 0.5)
    )
    np.testing.assert_array_equal(out, vertices)
    assert out is not vertices


def test_an_unfolded_sheet_is_returned_unchanged() -> None:
    vertices, faces = _grid()
    out = relax_folds(
        vertices, faces, lambda v: _facing_down(v, faces), FoldRelaxSettings(8, 2, 0.5)
    )
    np.testing.assert_array_equal(out, vertices)


def _floor(points: np.ndarray) -> np.ndarray:
    """Distance to a floor at ``z = 0`` (the obstacle lies below)."""
    return points[:, 2].copy()


def _ramp() -> tuple[np.ndarray, np.ndarray]:
    """The sheet, facing down at the floor, 2 mm above it rising to 20 mm."""
    vertices, faces = _grid()
    vertices[:, 2] = 0.002 + vertices[:, 0] * (0.018 / ((_N - 1) * _CELL))
    return vertices, faces[:, ::-1].copy()


def _normals_down(vertices: np.ndarray) -> np.ndarray:
    return np.tile([0.0, 0.0, -1.0], (len(vertices), 1))


_PAD = PadConformSettings(
    bones=(),
    reach_m=0.010,
    clearance_m=0.0003,
    facing_full=0.5,
    smooth_iterations=2,
    passes=3,
)


def test_distance_gradient_points_away_from_the_obstacle() -> None:
    points = np.array([[0.1, -0.2, 0.004], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(
        distance_gradient(_floor, points, 1e-5), [[0, 0, 1], [0, 0, 1]], atol=1e-9
    )


def test_pad_skin_near_the_obstacle_settles_onto_it() -> None:
    vertices, faces = _ramp()
    pad_ids = np.arange(len(vertices), dtype=np.int64)
    out = conform_pads(vertices, faces, pad_ids, _floor, _PAD, _normals_down)
    before, after = _floor(vertices), _floor(out)
    assert (after >= _PAD.clearance_m - 1e-12).all(), "pad pushed into the obstacle"
    assert (after <= before + 1e-12).all(), "pad moved away from the obstacle"
    nearest = before < 0.004
    assert (after[nearest] < 0.5 * before[nearest]).all()
    # Skin well outside the reach stays where skinning put it.
    far = before > _PAD.reach_m + 2 * 0.003
    assert far.any()
    np.testing.assert_allclose(out[far], vertices[far], atol=2e-4)
    # The pad stays a smooth ramp: gaps keep their order along the sheet.
    rows = after.reshape(_N, _N).mean(axis=1)
    assert (np.diff(rows) >= 0.0).all()


def test_only_pad_vertices_that_face_the_obstacle_move() -> None:
    vertices, faces = _ramp()
    pad_ids = np.arange(0, 2 * _N, dtype=np.int64)
    out = conform_pads(vertices, faces, pad_ids, _floor, _PAD, _normals_down)
    np.testing.assert_array_equal(out[2 * _N :], vertices[2 * _N :])
    assert np.abs(out[: 2 * _N] - vertices[: 2 * _N]).max() > 1e-4
    everyone = np.arange(len(vertices), dtype=np.int64)
    back = conform_pads(
        vertices, faces, everyone, _floor, _PAD, lambda v: -_normals_down(v)
    )
    np.testing.assert_array_equal(back, vertices)


def test_pad_conform_is_switched_off_by_zero_reach_or_zero_passes() -> None:
    vertices, faces = _ramp()
    ids = np.arange(len(vertices), dtype=np.int64)
    for off in (
        dataclasses.replace(_PAD, reach_m=0.0),
        dataclasses.replace(_PAD, passes=0),
    ):
        out = conform_pads(vertices, faces, ids, _floor, off, _normals_down)
        np.testing.assert_array_equal(out, vertices)


def test_bad_corrective_settings_raise() -> None:
    for bad in ((-1, 1, 0.5), (1, -1, 0.5), (1, 1, 0.0), (1, 1, 1.5)):
        with pytest.raises(ValueError):
            FoldRelaxSettings(*bad)
    for bad in (
        {"reach_m": -0.001},
        {"clearance_m": -0.001},
        {"facing_full": 0.0},
        {"smooth_iterations": -1},
        {"passes": -1},
    ):
        with pytest.raises(ValueError):
            PadConformSettings(**{**_PAD.__dict__, **bad})


def test_skin_just_inside_the_obstacle_is_lifted_to_its_surface() -> None:
    vertices, _ = _grid()
    vertices[:, 2] = np.linspace(-0.004, 0.004, len(vertices))
    out = lift_out(vertices, _floor, max_depth_m=0.001)
    shallow = (vertices[:, 2] < 0.0) & (vertices[:, 2] >= -0.001)
    assert shallow.any()
    np.testing.assert_allclose(out[shallow, 2], 0.0, atol=1e-9)
    np.testing.assert_array_equal(out[shallow, :2], vertices[shallow, :2])
    # Skin outside, and skin deeper than the limit (a posing fault), stays.
    np.testing.assert_array_equal(out[~shallow], vertices[~shallow])
    assert (~shallow & (vertices[:, 2] < 0.0)).any()


def test_lifting_out_is_switched_off_by_zero_depth_and_rejects_a_negative() -> None:
    vertices, _ = _grid()
    vertices[:, 2] = -0.0005
    out = lift_out(vertices, _floor, max_depth_m=0.0)
    np.testing.assert_array_equal(out, vertices)
    assert out is not vertices
    with pytest.raises(ValueError):
        lift_out(vertices, _floor, max_depth_m=-0.001)


def _tent(height: float) -> tuple[np.ndarray, np.ndarray]:
    """The sheet with its middle row lifted into a ridge of ``height``."""
    vertices, faces = _grid()
    vertices[(_N // 2) * _N : (_N // 2 + 1) * _N, 2] = height
    return vertices, faces


def test_faces_at_an_edge_that_sharpened_past_the_limit_are_reported() -> None:
    rest, faces = _grid()
    posed, _ = _tent(1.5 * _CELL)  # each side of the ridge tilts 56 degrees
    assert len(sharpened_faces(rest, rest, faces, 40.0)) == 0
    found = sharpened_faces(posed, rest, faces, 40.0)
    assert len(found) > 0
    ridge = (_N // 2) * _CELL
    centres = posed[faces[found]].mean(axis=1)
    assert (np.abs(centres[:, 0] - ridge) <= 2.0 * _CELL).all()
    # An edge that was already that sharp at rest is not a new crease.
    assert len(sharpened_faces(posed, posed, faces, 40.0)) == 0
    assert len(sharpened_faces(posed, rest, faces, 170.0)) == 0


def test_relaxing_a_sharpened_crease_brings_it_under_the_limit() -> None:
    rest, faces = _grid()
    posed, _ = _tent(1.5 * _CELL)

    def sharp(v: np.ndarray) -> np.ndarray:
        return sharpened_faces(v, rest, faces, 40.0)

    relaxed = relax_folds(posed, faces, sharp, FoldRelaxSettings(12, 1, 0.5))
    assert len(sharp(relaxed)) == 0
    assert 0.0 < relaxed[:, 2].max() < posed[:, 2].max()


def _bowl(depth: float) -> tuple[np.ndarray, np.ndarray]:
    """The sheet as skin facing the floor (``-z``), dipping ``depth`` into it."""
    vertices, faces = _grid()
    centre = vertices[(_N // 2) * _N + _N // 2].copy()
    reach = np.linalg.norm(vertices[:, :2] - centre[:2], axis=1)
    vertices[:, 2] = 3.0 * depth * (reach / reach.max()) ** 2 - depth
    return vertices, faces[:, ::-1].copy()


def _down_normals(faces: np.ndarray):
    def normals(v: np.ndarray) -> np.ndarray:
        tri = v[faces]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        out = np.zeros_like(v)
        for k in range(3):
            np.add.at(out, faces[:, k], face)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return normals


def test_pressed_skin_lies_flat_and_the_skin_beside_it_bulges_sideways() -> None:
    vertices, faces = _bowl(0.002)
    settings = PressSettings(max_depth_m=0.003, bulge_gain=0.5, bulge_rings=1)
    out = press_flat(vertices, faces, _floor, settings, _down_normals(faces))
    pressed = vertices[:, 2] < 0.0
    assert pressed.any() and not pressed.all()
    np.testing.assert_allclose(out[pressed, 2], 0.0, atol=1e-9)
    assert out[:, 2].min() >= -1e-12
    centre = vertices[(_N // 2) * _N + _N // 2, :2]
    outward = vertices[:, :2] - centre
    moved = ~pressed & (np.abs(out - vertices).max(axis=1) > 0.0)
    assert moved.any()
    # The bulge pushes skin away from the contact patch, never into the floor.
    assert (
        np.einsum("ij,ij->i", (out - vertices)[moved, :2], outward[moved]) > 0
    ).all()
    assert (out[moved, 2] >= vertices[moved, 2] - 1e-12).all()
    # It reaches one ring past the pressed skin and no farther.
    neighbours = vertex_neighbours(faces, len(vertices))
    ring = {int(j) for i in np.flatnonzero(pressed) for j in neighbours[i]}
    assert set(np.flatnonzero(moved).tolist()) <= ring


def test_press_without_bulge_is_lift_out_and_bad_settings_raise() -> None:
    vertices, faces = _bowl(0.002)
    flat = PressSettings(max_depth_m=0.003, bulge_gain=0.0, bulge_rings=2)
    out = press_flat(vertices, faces, _floor, flat, _down_normals(faces))
    np.testing.assert_array_equal(out, lift_out(vertices, _floor, max_depth_m=0.003))
    off = PressSettings(max_depth_m=0.0, bulge_gain=0.5, bulge_rings=2)
    unchanged = press_flat(vertices, faces, _floor, off, _down_normals(faces))
    np.testing.assert_array_equal(unchanged, vertices)
    for bad in ((-0.001, 0.5, 1), (0.001, -0.1, 1), (0.001, 0.5, -1)):
        with pytest.raises(ValueError):
            PressSettings(*bad)


def test_skin_gives_way_to_a_point_standing_through_a_large_triangle() -> None:
    """A coarse sheet (outside is +z) over a point 1 mm under its middle."""
    sheet = np.array([[0.0, 0, 0], [0.006, 0, 0], [0.003, 0.006, 0], [0.03, 0.03, 0]])
    up = lambda v: np.tile([0.0, 0.0, 1.0], (len(v), 1))  # noqa: E731
    point = np.array([[0.003, 0.002, -0.001]])
    out = yield_to_points(sheet, up, point, max_depth_m=0.003, passes=2)
    np.testing.assert_allclose(out[:3, 2], -0.001, atol=1e-12)
    np.testing.assert_array_equal(out[3], sheet[3])
    np.testing.assert_array_equal(out[:, :2], sheet[:, :2])
    # A point above the skin, or deeper than the limit, moves nothing.
    for z in (0.001, -0.005):
        still = yield_to_points(
            sheet, up, point * [1, 1, 0] + [0, 0, z], max_depth_m=0.003, passes=2
        )
        np.testing.assert_array_equal(still, sheet)
    off = yield_to_points(sheet, up, point, max_depth_m=0.0, passes=2)
    np.testing.assert_array_equal(off, sheet)
    with pytest.raises(ValueError):
        yield_to_points(sheet, up, point, max_depth_m=-0.001, passes=2)


def test_skin_stands_a_clearance_off_points_that_all_but_touch_it() -> None:
    """In a hollow between vertices a point can be inside the skin and still
    above its nearest vertex: skin within the clearance gives way too."""
    sheet = np.array([[0.0, 0, 0], [0.006, 0, 0], [0.003, 0.006, 0], [0.03, 0.03, 0]])
    up = lambda v: np.tile([0.0, 0.0, 1.0], (len(v), 1))  # noqa: E731
    point = np.array([[0.003, 0.002, 0.00005]])
    limits = {"max_depth_m": 0.003, "passes": 3}
    out = yield_to_points(sheet, up, point, clearance_m=0.0003, **limits)
    np.testing.assert_allclose(out[:3, 2], -0.00025, atol=1e-12)
    np.testing.assert_array_equal(out[3], sheet[3])
    far = yield_to_points(
        sheet, up, point + [0, 0, 0.001], clearance_m=0.0003, **limits
    )
    np.testing.assert_array_equal(far, sheet)
    np.testing.assert_array_equal(yield_to_points(sheet, up, point, **limits), sheet)
    with pytest.raises(ValueError):
        yield_to_points(sheet, up, point, clearance_m=-0.0001, **limits)
