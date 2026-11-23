"""Bracketed sphere contact against finite, moving triangle patches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import contact_event
from src.shared.python.motion_matching.contact_event import (
    find_contact_event as _find_contact_event,
)

pytestmark = pytest.mark.unit

_TRIANGLES = np.array(
    [
        [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        [[0.0, -1.0, -1.0], [0.0, 0.0, 1.0], [0.0, -1.0, 1.0]],
    ]
)


def find_contact_event(advance, face, centre, radius, **kwargs):
    """Explicit directional kinematic velocity for scalar-coordinate test fixtures."""

    def point_velocity(q, qd, point):
        triangles = face(q)
        closest, bary = contact_event._point_triangles(point, triangles)
        index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
        eps = 1e-7
        return (
            bary[index]
            @ (face(q + eps * qd)[index] - face(q - eps * qd)[index])
            / (2 * eps)
        )

    kwargs.setdefault("face_point_velocity_world", point_velocity)
    return _find_contact_event(advance, face, centre, radius, **kwargs)


def test_missing_face_velocity_does_not_accept_candidate() -> None:
    result = _find_contact_event(
        _advance(), _face, np.array([1.0, 0.0, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is None
    assert result.coverage_status == "unresolved_no_accepted_candidate"


def test_near_zero_physical_approach_is_unresolved() -> None:
    result = _find_contact_event(
        _advance(),
        _face,
        np.array([1.0, 0.0, 0.0]),
        0.5,
        duration_s=1.0,
        face_point_velocity_world=lambda q, v, point: np.array([1e-9, 0.0, 0.0]),
    )
    assert result.event is None


def test_rk4_prefix_derivative_cannot_substitute_for_event_velocity() -> None:
    from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step

    class Oscillator:
        def acceleration(self, q, v, tau):
            return -(q - 1.0)

    def advance(t):
        return zoh_rk4_step(
            Oscillator(), np.array([1.0]), np.array([1.0]), np.zeros(1), t
        )

    triangle = np.array([[[0.0, -4.0, -4.0], [0.0, 0.0, 4.0], [0.0, 4.0, -4.0]]])
    result = find_contact_event(
        advance,
        lambda q: triangle + np.array([q[0], 0.0, 0.0]),
        np.zeros(3),
        1.9375,
        duration_s=1.6,
        time_tolerance_s=1e-10,
        face_point_velocity_world=lambda q, qd, point: np.array([qd[0], 0.0, 0.0]),
    )
    assert result.event is None
    assert result.coverage_status == "unresolved_no_accepted_candidate"


def _advance(rate: float = 1.0):
    def advance(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([rate * time_s]), np.array([rate])

    return advance


def _face(q: np.ndarray) -> np.ndarray:
    return _TRIANGLES + np.array([q[0], 0.0, 0.0])


def test_finds_converged_approaching_root_and_returns_integrated_state() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([1.0, 0.0, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is not None
    assert result.coverage_status == "sampled_bracket_candidate"
    assert result.event.time_s == pytest.approx(0.5, abs=1e-8)
    np.testing.assert_allclose(result.event.q, [0.5], atol=1e-8)
    np.testing.assert_allclose(result.event.qd, [1.0], atol=1e-12)
    np.testing.assert_allclose(result.event.contact_point_m, [0.5, 0.0, 0.0], atol=1e-8)
    assert result.event.approach_speed_mps == pytest.approx(1.0, abs=1e-6)


def test_folded_seam_normal_switch_does_not_create_a_distant_contact() -> None:
    angle = np.pi / 6.0
    triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, np.cos(angle), np.sin(angle)]],
        ]
    )
    centre = np.array([0.25, -0.58, 0.98])
    radius = 0.05
    result = find_contact_event(
        _advance(),
        lambda q: triangles - np.array([0.0, 0.02 * q[0], 0.0]),
        centre,
        radius,
        duration_s=1.0,
        time_tolerance_s=1e-6,
    )
    assert result.event is None
    assert result.coverage_status == "unresolved_no_accepted_candidate"


def test_fast_contact_returns_geometrically_converged_sphere_clearance() -> None:
    centre = np.array([0.9, 0.0, 0.0])
    radius = 0.5
    result = find_contact_event(
        _advance(1000.0),
        _face,
        centre,
        radius,
        duration_s=0.002,
        time_tolerance_s=1e-4,
    )
    assert result.event is not None
    assert abs(np.linalg.norm(centre - result.event.contact_point_m) - radius) <= 1e-6


def test_miss_penalty_is_continuous_across_folded_seam_normal_switch() -> None:
    angle = np.pi / 6.0
    triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, np.cos(angle), np.sin(angle)]],
        ]
    )
    switch_y = -0.98 * np.tan(angle)
    penalties = []
    for offset in (-1e-6, 1e-6):
        result = find_contact_event(
            _advance(0.0),
            lambda _q: triangles,
            np.array([0.25, switch_y + offset, 0.98]),
            0.05,
            duration_s=1.0,
        )
        assert result.event is None
        penalties.append(result.closest_approach_penalty_m)
    assert abs(penalties[1] - penalties[0]) < 1e-4


def test_committed_driver_face_retains_a_front_side_contact() -> None:
    from src.shared.python.motion_matching.club_face_target import (
        face_triangles_in_frame,
    )

    spec_path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    )
    triangles = face_triangles_in_frame(
        json.loads(spec_path.read_text(encoding="utf-8"))
    )
    target = triangles[300]
    normal = np.cross(target[1] - target[0], target[2] - target[0])
    normal /= np.linalg.norm(normal)
    radius = 0.02135
    centre = target.mean(axis=0) + normal * (radius + 0.01)
    result = find_contact_event(
        _advance(),
        lambda q: triangles + q[0] * 0.02 * normal,
        centre,
        radius,
        duration_s=1.0,
        time_tolerance_s=1e-7,
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(0.5, abs=1e-6)
    assert result.event.approach_speed_mps == pytest.approx(0.02, rel=1e-5)
    assert float(result.event.contact_normal @ normal) > 0.99
    assert float((centre - result.event.contact_point_m) @ normal) > 0.0
    assert abs(np.linalg.norm(centre - result.event.contact_point_m) - radius) <= 1e-6


def test_boundary_topology_reuse_preserves_penalty_for_rigid_pose() -> None:
    topology = contact_event._build_boundary_topology(_TRIANGLES)
    point = np.array([0.2, 0.3, 0.4])
    radius = 0.1
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.3, -0.2, 0.6])
    moved = _TRIANGLES @ rotation.T + translation
    moved_point = point @ rotation.T + translation
    topology.validate_rigid_pose(moved)

    expected = contact_event._boundary_proximity_penalty(moved, moved_point, radius)
    actual = contact_event._boundary_proximity_penalty(
        moved, moved_point, radius, topology=topology
    )
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_boundary_topology_is_translation_invariant_at_resolvable_offsets() -> None:
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    second = first + np.array([0.0, 1e-8, 0.0])
    mesh = np.stack((first, second))
    translated = mesh + np.array([1e7, -1e7, 1e7])

    base_topology = contact_event._build_boundary_topology(mesh)
    translated_topology = contact_event._build_boundary_topology(translated)
    assert (
        len(base_topology.boundary_edges)
        == len(translated_topology.boundary_edges)
        == 6
    )


def test_boundary_topology_rejects_offsets_beyond_mesh_resolution() -> None:
    mesh = _TRIANGLES + np.array([1e12, -1e12, 1e12])
    with pytest.raises(ValueError, match="resolution"):
        contact_event._build_boundary_topology(mesh)


@pytest.mark.parametrize(
    "mesh",
    [
        np.stack((_TRIANGLES[0], _TRIANGLES[0])),
        np.stack((_TRIANGLES[0], _TRIANGLES[0][::-1])),
        np.stack(
            (
                _TRIANGLES[0],
                _TRIANGLES[1],
                np.array([[0.0, -1.0, -1.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.0]]),
            )
        ),
        np.stack((_TRIANGLES[0], _TRIANGLES[1][::-1])),
    ],
)
def test_invalid_face_topology_fails_closed(mesh: np.ndarray) -> None:
    with pytest.raises(ValueError, match="topology|winding"):
        contact_event._build_boundary_topology(mesh)


@pytest.mark.parametrize("change", ["deform", "reorder"])
def test_contact_search_fails_closed_if_face_topology_changes(change: str) -> None:
    def face(q: np.ndarray) -> np.ndarray:
        triangles = _TRIANGLES.copy()
        if change == "deform" and q[0] > 0.0:
            triangles[1, 2, 2] += q[0]
        elif change == "reorder" and q[0] > 0.0:
            triangles = triangles[::-1]
        return triangles

    with pytest.raises(ValueError, match="rigid|topology"):
        find_contact_event(
            _advance(), face, np.array([1.0, 0.0, 0.0]), 0.5, duration_s=1.0
        )


def test_root_converges_on_nonlinear_state_advance() -> None:
    def advance(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([time_s**2]), np.array([2.0 * time_s])

    result = find_contact_event(
        advance,
        _face,
        np.array([1.0, 0.0, 0.0]),
        0.5,
        duration_s=1.0,
        time_tolerance_s=1e-10,
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(np.sqrt(0.5), abs=1e-9)
    assert result.event.q[0] == pytest.approx(0.5, abs=1e-9)
    assert result.event.qd[0] == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_unconverged_bracket_is_not_returned_as_contact() -> None:
    with pytest.raises(ValueError, match="converg"):
        find_contact_event(
            _advance(),
            _face,
            np.array([0.9, 0.0, 0.0]),
            0.5,
            duration_s=1.0,
            time_tolerance_s=1e-9,
            max_iterations=1,
        )


def test_rejects_a_sphere_that_passes_outside_the_finite_face() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([1.0, 2.0, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is None
    assert np.isfinite(result.closest_approach_penalty_m)
    assert result.closest_approach_penalty_m > 0.0


def test_rejects_separating_motion() -> None:
    result = find_contact_event(
        _advance(-1.0), _face, np.array([1.0, 0.0, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is None
    assert np.isfinite(result.closest_approach_penalty_m)


def test_rejects_back_face_approach() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([-1.0, 0.0, 0.0]), 0.5, duration_s=2.0
    )
    assert result.event is None


def test_back_face_crossing_does_not_hide_later_front_face_contact() -> None:
    def advance(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([np.cos(np.pi * time_s)]), np.array(
            [-np.pi * np.sin(np.pi * time_s)]
        )

    result = find_contact_event(
        advance, _face, np.zeros(3), 0.1, duration_s=2.0, time_tolerance_s=1e-9
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(2.0 - np.arccos(-0.1) / np.pi, abs=1e-8)
    assert result.event.approach_speed_mps > 0.0


def test_rejects_contact_on_the_outer_mesh_boundary() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([1.0, 1.5, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is None


def test_boundary_vertex_remains_boundary_when_nearest_face_has_internal_edges() -> (
    None
):
    # The first face touches the outer corner only at its vertex; both its
    # incident edges are shared. Mesh ordering must not turn that corner into
    # an interior point when nearest triangles tie.
    vertex = [0.0, 0.0, 0.0]
    a, b = [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]
    c = [0.0, 0.0, 1.0]
    p, q = [0.0, 0.6, 0.3], [0.0, 0.3, 0.6]
    mesh = np.array(
        [
            [vertex, p, q],
            [vertex, a, p],
            [vertex, q, c],
            [a, b, p],
            [b, q, p],
            [b, c, q],
        ]
    )
    topology = contact_event._build_boundary_topology(mesh)
    assert contact_event._boundary_edge_contact(
        mesh, 0, np.array([1.0, 0.0, 0.0]), topology=topology
    )
    # p is a genuinely interior vertex, shared by four faces.
    assert not contact_event._boundary_edge_contact(
        mesh, 0, np.array([0.0, 1.0, 0.0]), topology=topology
    )
    result = find_contact_event(
        _advance(),
        lambda state: mesh + np.array([state[0], 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.5,
        duration_s=1.0,
    )
    assert result.event is None


def test_accepts_contact_across_an_interior_triangle_seam() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([1.0, -0.5, 0.0]), 0.5, duration_s=1.0
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(0.5, abs=1e-8)


def test_no_root_penalty_is_smooth_and_finite_across_a_triangle_seam() -> None:
    ball = np.array([1.0, 0.0, 0.0])
    offsets = np.linspace(-1e-5, 1e-5, 5)
    penalties = []
    for offset in offsets:
        moved = _TRIANGLES.copy()
        moved[..., 0] += 0.8
        moved[..., 1] += offset
        result = find_contact_event(
            _advance(0.0),
            lambda _q, tris=moved: tris,
            ball + np.array([0.0, 0.0, 0.0]),
            0.1,
            duration_s=1.0,
        )
        assert result.event is None
        penalties.append(result.closest_approach_penalty_m)
    assert np.isfinite(penalties).all()
    assert np.max(np.abs(np.diff(penalties))) < 1e-4


def test_penalty_is_continuous_when_the_root_is_just_acquired() -> None:
    ball = np.array([1.0, 0.0, 0.0])
    before = find_contact_event(_advance(), _face, ball, 0.5, duration_s=0.5 - 1e-6)
    after = find_contact_event(_advance(), _face, ball, 0.5, duration_s=0.5 + 1e-6)
    assert before.event is None and after.event is not None
    assert (
        abs(before.closest_approach_penalty_m - after.closest_approach_penalty_m) < 1e-5
    )


def test_miss_penalty_prefers_front_face_near_contact_to_penetration_backface_and_edge() -> (
    None
):
    radius = 0.5

    def static(_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.zeros(1)

    def cost(centre: np.ndarray) -> float:
        return find_contact_event(
            static, _face, centre, radius, duration_s=1.0
        ).closest_approach_penalty_m

    near_front = cost(np.array([radius + 0.05, 0.0, 0.0]))
    penetrating = cost(np.array([radius - 0.1, 0.0, 0.0]))
    backface = cost(np.array([-radius - 0.05, 0.0, 0.0]))
    outside_boundary = cost(np.array([radius, 1.2, 0.0]))
    assert np.isfinite([near_front, penetrating, backface, outside_boundary]).all()
    assert near_front < penetrating < backface
    assert near_front < outside_boundary


def test_exact_at_arm_front_face_contact_is_accepted_only_when_approaching() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([0.5, 0.0, 0.0]), 0.5, duration_s=0.1
    )
    assert result.event is not None
    assert result.event.time_s == 0.0
    np.testing.assert_array_equal(result.event.q, [0.0])
    np.testing.assert_allclose(result.event.qd, [1.0])
    assert result.event.approach_speed_mps == pytest.approx(1.0)

    separating = find_contact_event(
        _advance(-1.0), _face, np.array([0.5, 0.0, 0.0]), 0.5, duration_s=0.1
    )
    assert separating.event is None


def test_exact_at_arm_contact_rejects_initial_separation_before_later_reversal() -> (
    None
):
    def separate_then_reverse(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-time_s + 100.0 * time_s**2]), np.array(
            [-1.0 + 200.0 * time_s]
        )

    result = find_contact_event(
        separate_then_reverse,
        _face,
        np.array([0.5, 0.0, 0.0]),
        0.5,
        duration_s=0.1,
    )
    assert result.event is None


def test_separating_exact_arm_can_be_followed_by_a_later_approaching_root() -> None:
    root_time = 0.0097
    curvature = 1.0 / root_time

    def separate_then_approach(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-time_s + curvature * time_s**2]), np.array(
            [-1.0 + 2.0 * curvature * time_s]
        )

    result = find_contact_event(
        separate_then_approach,
        _face,
        np.array([0.5, 0.0, 0.0]),
        0.5,
        duration_s=0.02,
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(root_time, abs=1e-8)
    np.testing.assert_allclose(result.event.q, [0.0], atol=1e-8)
    np.testing.assert_allclose(result.event.qd, [1.0], atol=1e-8)


def test_zero_gap_at_both_endpoints_finds_approaching_return_at_end() -> None:
    duration = 0.01

    def separate_then_touch(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        q = -time_s * (duration - time_s)
        qd = -duration + 2.0 * time_s
        return np.array([q]), np.array([qd])

    result = find_contact_event(
        separate_then_touch,
        _face,
        np.array([0.5, 0.0, 0.0]),
        0.5,
        duration_s=duration,
    )
    assert result.event is not None
    assert result.event.time_s == pytest.approx(duration, abs=1e-12)
    np.testing.assert_allclose(result.event.q, [0.0], atol=1e-14)
    np.testing.assert_allclose(result.event.qd, [duration], atol=1e-14)


def test_endpoint_zero_gap_with_tangent_velocity_is_not_an_approaching_root() -> None:
    def tangent_at_end(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-((1.0 - time_s) ** 2)]), np.array([2.0 * (1.0 - time_s)])

    result = find_contact_event(
        tangent_at_end, _face, np.array([1.0, 0.0, 0.0]), 1.0, duration_s=1.0
    )
    assert result.event is None
    assert np.isfinite(result.closest_approach_penalty_m)
    assert result.coverage_status == "unresolved_no_accepted_candidate"


def test_narrow_unsampled_crossing_is_reported_as_unresolved() -> None:
    centre_time, width, duration = 0.0703125, 0.003, 0.125

    def advance(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        z = (time_s - centre_time) / width
        envelope = np.exp(-(z * z))
        return (
            np.array([0.6 - 0.2 * envelope]),
            np.array([0.4 * z / width * envelope]),
        )

    def moving_face(q: np.ndarray) -> np.ndarray:
        x = q[0]
        return np.array(
            [
                [[x, 0.0, 0.0], [x, 0.0, 1.0], [x, 1.0, 0.0]],
                [[x, 0.0, 1.0], [x, 1.0, 1.0], [x, 1.0, 0.0]],
            ]
        )

    result = find_contact_event(
        advance,
        moving_face,
        np.array([0.0, 0.5, 0.5]),
        0.5,
        duration_s=duration,
    )
    assert advance(centre_time)[0][0] - 0.5 < 0.0
    assert result.event is None
    assert result.coverage_status == "unresolved_no_accepted_candidate"


def test_nonfinite_geometry_cost_fails_closed() -> None:
    huge = _TRIANGLES * 1e200
    with pytest.raises(ValueError, match="finite"):
        find_contact_event(
            _advance(0.0),
            lambda _q: huge,
            np.zeros(3),
            0.5,
            duration_s=1.0,
        )


def test_miss_cost_is_continuous_with_matching_slopes_across_face_plane() -> None:
    def static(_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.zeros(1)

    def cost(x: float) -> float:
        return find_contact_event(
            static,
            _face,
            np.array([x, 0.0, 0.0]),
            0.5,
            duration_s=1.0,
        ).closest_approach_penalty_m

    h = 1e-12
    left, centre, right = cost(-h), cost(0.0), cost(h)
    assert max(abs(centre - left), abs(right - centre)) < 1e-3
    left_slope = (centre - left) / h
    right_slope = (right - centre) / h
    assert np.isfinite([left_slope, right_slope]).all()
    assert abs(left_slope - right_slope) < 0.02 * max(
        abs(left_slope), abs(right_slope), 1.0
    )


def test_miss_cost_has_finite_matching_slopes_across_internal_triangle_seam() -> None:
    def static(_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.zeros(1)

    def cost(y: float) -> float:
        return find_contact_event(
            static,
            _face,
            np.array([0.7, y, 0.0]),
            0.5,
            duration_s=1.0,
        ).closest_approach_penalty_m

    h = 1e-6
    left, centre, right = cost(-0.5 - h), cost(-0.5), cost(-0.5 + h)
    slopes = np.array([(centre - left) / h, (right - centre) / h])
    assert np.isfinite([left, centre, right, *slopes]).all()
    assert abs(slopes[0] - slopes[1]) < 0.1 * max(abs(slopes[0]), abs(slopes[1]), 1.0)


def test_root_search_raises_if_iteration_limit_prevents_convergence() -> None:
    with pytest.raises(ValueError, match="did not converge"):
        find_contact_event(
            _advance(),
            _face,
            np.array([0.95, 0.0, 0.0]),
            0.5,
            duration_s=1.0,
            time_tolerance_s=1e-12,
            max_iterations=1,
        )
