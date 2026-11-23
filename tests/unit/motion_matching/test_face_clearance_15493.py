"""Static finite-patch distance is not first-contact evidence."""

import numpy as np
import pytest

from src.shared.python.motion_matching.face_clearance import (
    face_clearance,
    require_initial_separation,
)

pytestmark = pytest.mark.unit

TRIANGLES = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


@pytest.mark.parametrize(
    ("centre", "expected", "closest"),
    [
        ([0.2, 0.2, 2.0], 2.0, [0.2, 0.2, 0.0]),
        ([0.5, -1.0, 0.0], 1.0, [0.5, 0.0, 0.0]),
        ([-1.0, -1.0, 0.0], np.sqrt(2.0), [0.0, 0.0, 0.0]),
    ],
)
def test_nearest_interior_edge_and_vertex(centre, expected, closest):
    sample = face_clearance(TRIANGLES, centre, 0.25)
    assert sample.gap_m == pytest.approx(expected - 0.25)
    np.testing.assert_allclose(sample.closest_point_m, closest, atol=1e-12)
    assert sample.triangle_index == 0


def test_multiple_triangles_and_overlap_guard():
    triangles = np.concatenate((TRIANGLES + [0, 0, 5], TRIANGLES))
    assert face_clearance(triangles, [0.2, 0.2, 1], 0.25).triangle_index == 1
    assert require_initial_separation(triangles, [0.2, 0.2, 1], 0.25).gap_m > 0
    for height in (0.25, 0.1, 0):
        with pytest.raises(ValueError, match="initial"):
            require_initial_separation(triangles, [0.2, 0.2, height], 0.25)
    with pytest.raises(ValueError, match="initial"):
        require_initial_separation(triangles, [0.2, 0.2, 1], 0.25, margin_m=0.8)


def test_unresolvable_world_coordinates_are_rejected():
    translated = TRIANGLES + 1e15
    with pytest.raises(ValueError, match="resolution"):
        face_clearance(translated, [1e15 + 0.2, 1e15 + 0.2, 1e15 + 2], 0.25)


def test_overlap_with_pinned_v2_face_patch_is_rejected():
    import json
    from pathlib import Path

    from src.shared.python.motion_matching.club_face_target import (
        face_triangles_in_frame,
    )

    path = Path(__file__).resolve().parents[3] / (
        "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    )
    triangles = face_triangles_in_frame(json.loads(path.read_text(encoding="utf-8")))
    triangle = triangles[len(triangles) // 2]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    centre = triangle.mean(axis=0) + normal * 0.01
    with pytest.raises(ValueError, match="initial"):
        require_initial_separation(triangles, centre, 0.02135)


def test_closest_point_agrees_with_independent_constrained_minimization():
    from scipy.optimize import minimize

    rng = np.random.default_rng(15493)
    for _ in range(12):
        triangle = rng.normal(size=(3, 3))
        point = rng.normal(size=3)
        a, b, c = triangle

        def objective(uv, a=a, b=b, c=c, point=point):
            return np.sum((a + uv[0] * (b - a) + uv[1] * (c - a) - point) ** 2)

        result = minimize(
            objective,
            [0.25, 0.25],
            method="SLSQP",
            bounds=[(0, 1), (0, 1)],
            constraints=[{"type": "ineq", "fun": lambda uv: 1 - uv.sum()}],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        assert result.success
        measured = face_clearance(triangle[None, :, :], point, 0.01)
        assert measured.gap_m + 0.01 == pytest.approx(np.sqrt(result.fun), abs=2e-6)


@pytest.mark.parametrize(
    ("triangles", "centre", "radius"),
    [
        (np.zeros((0, 3, 3)), [0, 0, 0], 1),
        (np.zeros((1, 3, 3)), [0, 0, 1], 1),
        (TRIANGLES, [np.nan, 0, 0], 1),
        (TRIANGLES, [0, 0, 1], 0),
        (TRIANGLES, [0, 0, 1], np.inf),
    ],
)
def test_bad_geometry_fails_closed(triangles, centre, radius):
    with pytest.raises(ValueError):
        face_clearance(triangles, centre, radius)
