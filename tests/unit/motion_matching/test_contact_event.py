"""Bracketed sphere contact against finite, moving triangle patches."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_event import find_contact_event

pytestmark = pytest.mark.unit

_TRIANGLES = np.array(
    [
        [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        [[0.0, -1.0, -1.0], [0.0, 0.0, 1.0], [0.0, -1.0, 1.0]],
    ]
)


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
    assert result.event.time_s == pytest.approx(0.5, abs=1e-8)
    np.testing.assert_allclose(result.event.q, [0.5], atol=1e-8)
    np.testing.assert_allclose(result.event.qd, [1.0], atol=1e-12)
    np.testing.assert_allclose(result.event.contact_point_m, [0.5, 0.0, 0.0], atol=1e-8)
    assert result.event.approach_speed_mps == pytest.approx(1.0, abs=1e-6)


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


def test_rejects_contact_on_the_outer_mesh_boundary() -> None:
    result = find_contact_event(
        _advance(), _face, np.array([1.0, 1.5, 0.0]), 0.5, duration_s=1.0
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


def test_root_search_raises_if_iteration_limit_prevents_convergence() -> None:
    with pytest.raises(ValueError, match="did not converge"):
        find_contact_event(
            _advance(),
            _face,
            np.array([1.0, 0.0, 0.0]),
            0.5,
            duration_s=1.0,
            time_tolerance_s=1e-12,
            max_iterations=1,
        )
