"""Value-asserting tests for the real ``CollisionChecker`` (issue #6997).

The pre-existing ``test_planning_collision.py`` only exercises the checker
through a mock engine that hands back ``Sphere`` geometry. These tests use a
tiny *real* ``CollisionCapable`` stub and assert concrete numeric outcomes of
``_aabb_overlap``, ``check_collision`` (body-body and body-environment),
``compute_distance`` (sign + zero-at-contact), ``check_path_collision``, the
environment-primitive mutators, and pair enable/disable.

All geometry here is real (``Sphere``), so the narrow-phase
``compute_primitive_distance`` actually runs -- nothing is mocked.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.robotics.planning.collision.collision_checker import (
    CollisionChecker,
    CollisionCheckerConfig,
)
from src.robotics.planning.collision.collision_types import (
    CollisionPair,
    CollisionQuery,
    CollisionQueryType,
)
from src.robotics.planning.collision.geometric_primitives import Sphere

pytestmark = pytest.mark.unit


class StubEngine:
    """Minimal real ``CollisionCapable`` implementation.

    Two unit-radius spheres whose centres are driven directly by the first two
    DOF of ``q`` (metres). No mocking: ``get_body_collision_geometry`` returns
    genuine ``Sphere`` primitives.
    """

    def __init__(self) -> None:
        self._q = np.zeros(2)
        self._v = np.zeros(2)
        self._radius = 1.0
        # Far apart by default so the default checker reports no collision.
        self._positions = {
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([10.0, 0.0, 0.0]),
        }

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), self._v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()
        self._v = np.asarray(v, dtype=float).copy()
        # Body "a" slides along x by q[0]; body "b" by q[1] from its base.
        self._positions["a"] = np.array([self._q[0], 0.0, 0.0])
        self._positions["b"] = np.array([10.0 + self._q[1], 0.0, 0.0])

    def get_body_names(self) -> list[str]:
        return list(self._positions.keys())

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        pos = self._positions.get(body_name)
        return None if pos is None else pos.copy()

    def get_body_collision_geometry(self, body_name: str) -> Sphere | None:
        if body_name not in self._positions:
            return None
        return Sphere(center=self._positions[body_name].copy(), radius=self._radius)


@pytest.fixture
def checker() -> CollisionChecker:
    # Broad phase on, zero margin so contact is exact for assertions.
    cfg = CollisionCheckerConfig(default_margin=0.0)
    return CollisionChecker(StubEngine(), cfg)


# --------------------------------------------------------------------------- #
# Construction / protocol enforcement
# --------------------------------------------------------------------------- #


def test_rejects_non_collision_capable_engine() -> None:
    with pytest.raises(TypeError, match="CollisionCapable"):
        CollisionChecker(object())


def test_default_pairs_are_all_unordered_body_pairs(checker: CollisionChecker) -> None:
    pairs = checker.get_collision_pairs()
    assert len(pairs) == 1
    assert pairs[0] == CollisionPair("a", "b")


# --------------------------------------------------------------------------- #
# _aabb_overlap: overlap / touch / disjoint
# --------------------------------------------------------------------------- #


class TestAabbOverlap:
    def test_overlapping(self, checker: CollisionChecker) -> None:
        a = Sphere(center=np.zeros(3), radius=1.0)
        b = Sphere(center=np.array([1.0, 0.0, 0.0]), radius=1.0)
        assert checker._aabb_overlap(a, b, margin=0.0) is True

    def test_touching_counts_as_overlap(self, checker: CollisionChecker) -> None:
        # AABBs share a face exactly at x=1 (>= comparison => overlap).
        a = Sphere(center=np.zeros(3), radius=1.0)
        b = Sphere(center=np.array([2.0, 0.0, 0.0]), radius=1.0)
        assert checker._aabb_overlap(a, b, margin=0.0) is True

    def test_disjoint(self, checker: CollisionChecker) -> None:
        a = Sphere(center=np.zeros(3), radius=1.0)
        b = Sphere(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
        assert checker._aabb_overlap(a, b, margin=0.0) is False

    def test_margin_bridges_a_gap(self, checker: CollisionChecker) -> None:
        a = Sphere(center=np.zeros(3), radius=1.0)
        b = Sphere(center=np.array([2.5, 0.0, 0.0]), radius=1.0)
        # Gap between AABB faces is 0.5; margin 0.6 closes it.
        assert checker._aabb_overlap(a, b, margin=0.0) is False
        assert checker._aabb_overlap(a, b, margin=0.6) is True


# --------------------------------------------------------------------------- #
# check_collision: body-body
# --------------------------------------------------------------------------- #


class TestCheckCollisionBodyBody:
    def test_no_collision_when_far_apart(self, checker: CollisionChecker) -> None:
        result = checker.check_collision(np.zeros(2))
        assert result.in_collision is False
        assert result.collision_pairs == []
        assert result.num_contacts == 0

    def test_collision_when_spheres_overlap(self, checker: CollisionChecker) -> None:
        # Move "a" to x=8.5: centres 8.5 vs 10.0 => gap 1.5 < r_a+r_b=2 => overlap.
        result = checker.check_collision(np.array([8.5, 0.0]))
        assert result.in_collision is True
        assert CollisionPair("a", "b") in result.collision_pairs
        assert result.num_contacts == 1

    def test_state_restored_after_check(self, checker: CollisionChecker) -> None:
        checker.check_collision(np.array([8.5, 0.0]))
        q_after, v_after = checker._engine.get_state()
        assert np.allclose(q_after, np.zeros(2))
        assert np.allclose(v_after, np.zeros(2))

    def test_non_finite_configuration_rejected(self, checker: CollisionChecker) -> None:
        with pytest.raises(ValueError, match="finite"):
            checker.check_collision(np.array([np.inf, 0.0]))


# --------------------------------------------------------------------------- #
# check_collision: body-environment
# --------------------------------------------------------------------------- #


class TestCheckCollisionEnvironment:
    def test_environment_primitive_triggers_collision(
        self, checker: CollisionChecker
    ) -> None:
        # Sphere obstacle overlapping body "a" (centre 0, r=1): obstacle centre
        # 1.5 away with r=1 => surface gap 1.5 - 1 - 1 = -0.5 (penetrating).
        obstacle = Sphere(center=np.array([1.5, 0.0, 0.0]), radius=1.0)
        checker.add_environment_primitive("obstacle", obstacle)
        result = checker.check_collision(np.zeros(2))
        assert result.in_collision is True
        assert CollisionPair("a", "obstacle") in result.collision_pairs

    def test_clearing_environment_removes_collision(
        self, checker: CollisionChecker
    ) -> None:
        obstacle = Sphere(center=np.array([1.5, 0.0, 0.0]), radius=1.0)
        checker.add_environment_primitive("obstacle", obstacle)
        assert checker.check_collision(np.zeros(2)).in_collision is True
        checker.clear_environment()
        assert checker.check_collision(np.zeros(2)).in_collision is False


# --------------------------------------------------------------------------- #
# compute_distance: sign + zero-at-contact
# --------------------------------------------------------------------------- #


class TestComputeDistance:
    def test_positive_distance_when_separated(self, checker: CollisionChecker) -> None:
        # a@0, b@10, r=1 each => surface gap = 10 - 1 - 1 = 8.
        result = checker.compute_distance(np.zeros(2))
        assert result.distance == pytest.approx(8.0)
        assert result.in_collision is False
        assert result.closest_pair == CollisionPair("a", "b")

    def test_zero_distance_at_contact(self, checker: CollisionChecker) -> None:
        # Centres exactly 2 apart (r+r) => touching => distance 0.
        result = checker.compute_distance(np.array([8.0, 0.0]))
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_negative_distance_when_penetrating(
        self, checker: CollisionChecker
    ) -> None:
        # Centres 1.5 apart => penetration of 0.5 => signed distance -0.5.
        result = checker.compute_distance(np.array([8.5, 0.0]))
        assert result.distance == pytest.approx(-0.5)
        assert result.in_collision is True
        assert result.penetration_depth == pytest.approx(0.5)

    def test_state_restored_after_distance(self, checker: CollisionChecker) -> None:
        checker.compute_distance(np.array([8.5, 0.0]))
        q_after, _ = checker._engine.get_state()
        assert np.allclose(q_after, np.zeros(2))


# --------------------------------------------------------------------------- #
# check_path_collision
# --------------------------------------------------------------------------- #


class TestCheckPathCollision:
    def test_clear_path_is_collision_free(self, checker: CollisionChecker) -> None:
        # Both endpoints leave the spheres far apart.
        free, t = checker.check_path_collision(
            np.zeros(2), np.array([1.0, 0.0]), num_samples=10
        )
        assert free is True
        assert t is None

    def test_path_flags_interpolated_waypoint(self, checker: CollisionChecker) -> None:
        # Endpoints are collision-free but the midpoint drives "a" into "b".
        free, t = checker.check_path_collision(
            np.array([0.0, 0.0]), np.array([17.0, 0.0]), num_samples=11
        )
        assert free is False
        assert t is not None
        assert 0.0 < t < 1.0

    def test_too_few_samples_rejected(self, checker: CollisionChecker) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            checker.check_path_collision(np.zeros(2), np.zeros(2), num_samples=1)


# --------------------------------------------------------------------------- #
# environment primitive add/remove/clear
# --------------------------------------------------------------------------- #


class TestEnvironmentPrimitiveMutators:
    def test_add_then_remove(self, checker: CollisionChecker) -> None:
        prim = Sphere(center=np.zeros(3), radius=0.5)
        checker.add_environment_primitive("obs", prim)
        assert checker.remove_environment_primitive("obs") is True

    def test_remove_missing_returns_false(self, checker: CollisionChecker) -> None:
        assert checker.remove_environment_primitive("nope") is False

    def test_empty_name_rejected(self, checker: CollisionChecker) -> None:
        with pytest.raises(ValueError, match="name cannot be empty"):
            checker.add_environment_primitive("", Sphere())

    def test_clear_empties_environment(self, checker: CollisionChecker) -> None:
        checker.add_environment_primitive("o1", Sphere(radius=0.5))
        checker.add_environment_primitive("o2", Sphere(radius=0.5))
        checker.clear_environment()
        assert checker.remove_environment_primitive("o1") is False


# --------------------------------------------------------------------------- #
# enable / disable collision pair
# --------------------------------------------------------------------------- #


class TestEnableDisablePair:
    def test_disable_removes_pair_from_checks(self, checker: CollisionChecker) -> None:
        checker.disable_collision_pair("a", "b")
        assert CollisionPair("a", "b") not in checker.get_collision_pairs()
        # With the only pair disabled, an otherwise-colliding config is clear.
        result = checker.check_collision(np.array([8.5, 0.0]))
        assert result.in_collision is False

    def test_enable_restores_pair(self, checker: CollisionChecker) -> None:
        checker.disable_collision_pair("a", "b")
        checker.enable_collision_pair("a", "b")
        assert CollisionPair("a", "b") in checker.get_collision_pairs()
        result = checker.check_collision(np.array([8.5, 0.0]))
        assert result.in_collision is True

    def test_query_exclude_pairs_skips_check(self, checker: CollisionChecker) -> None:
        query = CollisionQuery(
            query_type=CollisionQueryType.BOOLEAN,
            exclude_pairs=[CollisionPair("a", "b")],
        )
        result = checker.check_collision(np.array([8.5, 0.0]), query=query)
        assert result.in_collision is False
