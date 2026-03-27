"""Tests for collision detection module.

Tests cover:
- Geometric primitives (Sphere, Box, Capsule, Cylinder, ConvexHull)
- Distance computation between primitives
- Collision checking
- CollisionChecker configuration-space collision detection
"""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.planning.collision import (
    Box,
    Capsule,
    CollisionChecker,
    CollisionCheckerConfig,
    CollisionPair,
    CollisionQuery,
    CollisionQueryType,
    CollisionResult,
    ConvexHull,
    Cylinder,
    DistanceResult,
    Sphere,
    check_primitive_collision,
    compute_primitive_distance,
)

# =============================================================================
# Mock Engine for Testing
# =============================================================================


class MockCollisionEngine:
    """Mock engine implementing CollisionCapable protocol."""

    def __init__(self, n_dof: int = 7) -> None:
        """Initialize mock engine."""
        self._n_dof = n_dof
        self._q = np.zeros(n_dof)
        self._v = np.zeros(n_dof)
        self._bodies: dict[str, dict] = {
            "link1": {"position": np.array([0.0, 0.0, 0.5]), "radius": 0.1},
            "link2": {"position": np.array([0.0, 0.0, 1.0]), "radius": 0.1},
            "link3": {"position": np.array([0.0, 0.0, 1.5]), "radius": 0.1},
        }

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current state."""
        return self._q.copy(), self._v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set state."""
        self._q = q.copy()
        self._v = v.copy()
        # Update body positions based on q (simplified kinematics)
        self._bodies["link1"]["position"] = np.array([q[0] * 0.1, q[1] * 0.1, 0.5])
        self._bodies["link2"]["position"] = np.array([q[0] * 0.1, q[1] * 0.1, 1.0])
        self._bodies["link3"]["position"] = np.array([q[2] * 0.1, q[3] * 0.1, 1.5])

    def get_body_names(self) -> list[str]:
        """Get body names."""
        return list(self._bodies.keys())

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        """Get body position."""
        if body_name in self._bodies:
            return self._bodies[body_name]["position"].copy()
        return None

    def get_body_collision_geometry(self, body_name: str) -> Sphere | None:
        """Get collision geometry as sphere."""
        if body_name in self._bodies:
            return Sphere(
                center=self._bodies[body_name]["position"],
                radius=self._bodies[body_name]["radius"],
            )
        return None


# =============================================================================
# Sphere Tests
# =============================================================================


class TestSphere:
    """Tests for Sphere primitive."""

    def test_create_sphere(self) -> None:
        """Test sphere creation with valid parameters."""
        sphere = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=0.5)
        assert sphere.radius == 0.5
        assert np.allclose(sphere.center, [1.0, 2.0, 3.0])

    def test_sphere_default_values(self) -> None:
        """Test sphere with default values."""
        sphere = Sphere()
        assert sphere.radius == 1.0
        assert np.allclose(sphere.center, [0.0, 0.0, 0.0])

    def test_sphere_invalid_radius(self) -> None:
        """Test that negative radius raises error."""
        with pytest.raises(ValueError, match="radius must be positive"):
            Sphere(radius=-1.0)

    def test_sphere_invalid_center(self) -> None:
        """Test that invalid center raises error."""
        with pytest.raises(ValueError, match="center must be shape"):
            Sphere(center=np.array([1.0, 2.0]))

    def test_sphere_aabb(self) -> None:
        """Test AABB computation."""
        sphere = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=0.5)
        min_corner, max_corner = sphere.get_aabb()
        assert np.allclose(min_corner, [0.5, 1.5, 2.5])
        assert np.allclose(max_corner, [1.5, 2.5, 3.5])

    def test_sphere_contains_point_inside(self) -> None:
        """Test point inside sphere."""
        sphere = Sphere(center=np.zeros(3), radius=1.0)
        assert sphere.contains_point(np.array([0.5, 0.0, 0.0]))
        assert sphere.contains_point(np.zeros(3))

    def test_sphere_contains_point_outside(self) -> None:
        """Test point outside sphere."""
        sphere = Sphere(center=np.zeros(3), radius=1.0)
        assert not sphere.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_sphere_support(self) -> None:
        """Test support function."""
        sphere = Sphere(center=np.array([1.0, 0.0, 0.0]), radius=0.5)
        direction = np.array([1.0, 0.0, 0.0])
        support = sphere.compute_support(direction)
        assert np.allclose(support, [1.5, 0.0, 0.0])


# =============================================================================
# Box Tests
# =============================================================================


class TestBox:
    """Tests for Box primitive."""

    def test_create_box(self) -> None:
        """Test box creation."""
        box = Box(
            center=np.array([1.0, 0.0, 0.0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        assert np.allclose(box.center, [1.0, 0.0, 0.0])
        assert np.allclose(box.half_extents, [0.5, 0.5, 0.5])

    def test_box_invalid_extents(self) -> None:
        """Test that non-positive extents raise error."""
        with pytest.raises(ValueError, match="half_extents must be positive"):
            Box(half_extents=np.array([0.5, -0.1, 0.5]))

    def test_box_aabb_axis_aligned(self) -> None:
        """Test AABB for axis-aligned box."""
        box = Box(
            center=np.array([1.0, 2.0, 3.0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        min_corner, max_corner = box.get_aabb()
        assert np.allclose(min_corner, [0.5, 1.5, 2.5])
        assert np.allclose(max_corner, [1.5, 2.5, 3.5])

    def test_box_contains_point_inside(self) -> None:
        """Test point inside box."""
        box = Box(center=np.zeros(3), half_extents=np.ones(3))
        assert box.contains_point(np.array([0.5, 0.5, 0.5]))
        assert box.contains_point(np.zeros(3))

    def test_box_contains_point_outside(self) -> None:
        """Test point outside box."""
        box = Box(center=np.zeros(3), half_extents=np.ones(3))
        assert not box.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_box_support(self) -> None:
        """Test support function."""
        box = Box(center=np.zeros(3), half_extents=np.array([1.0, 1.0, 1.0]))
        support = box.compute_support(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(support, [1.0, 1.0, 1.0])


# =============================================================================
# Capsule Tests
# =============================================================================


class TestCapsule:
    """Tests for Capsule primitive."""

    def test_create_capsule(self) -> None:
        """Test capsule creation."""
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.1,
        )
        assert capsule.radius == 0.1
        assert np.allclose(capsule.length, 1.0)

    def test_capsule_invalid_radius(self) -> None:
        """Test that non-positive radius raises error."""
        with pytest.raises(ValueError, match="radius must be positive"):
            Capsule(radius=0.0)

    def test_capsule_properties(self) -> None:
        """Test capsule properties."""
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 2.0]),
            radius=0.5,
        )
        assert np.allclose(capsule.length, 2.0)
        assert np.allclose(capsule.center, [0.0, 0.0, 1.0])
        assert np.allclose(capsule.axis, [0.0, 0.0, 1.0])

    def test_capsule_aabb(self) -> None:
        """Test AABB computation."""
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.1,
        )
        min_corner, max_corner = capsule.get_aabb()
        assert np.allclose(min_corner, [-0.1, -0.1, -0.1])
        assert np.allclose(max_corner, [0.1, 0.1, 1.1])

    def test_capsule_contains_point(self) -> None:
        """Test point containment."""
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.1,
        )
        assert capsule.contains_point(np.array([0.0, 0.0, 0.5]))
        assert capsule.contains_point(np.array([0.05, 0.0, 0.5]))
        assert not capsule.contains_point(np.array([0.2, 0.0, 0.5]))


# =============================================================================
# Cylinder Tests
# =============================================================================


class TestCylinder:
    """Tests for Cylinder primitive."""

    def test_create_cylinder(self) -> None:
        """Test cylinder creation."""
        cylinder = Cylinder(
            center=np.array([0.0, 0.0, 0.5]),
            radius=0.5,
            height=1.0,
            axis=np.array([0.0, 0.0, 1.0]),
        )
        assert cylinder.radius == 0.5
        assert cylinder.height == 1.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"radius": 0.0}, "radius must be positive"),
            ({"height": -1.0}, "height must be positive"),
        ],
        ids=["zero-radius", "negative-height"],
    )
    def test_cylinder_invalid_dimensions(self, kwargs: dict, match: str) -> None:
        """Test invalid dimensions."""
        with pytest.raises(ValueError, match=match):
            Cylinder(**kwargs)

    @pytest.mark.parametrize(
        "point, expected",
        [
            (np.array([0.0, 0.0, 0.5]), True),
            (np.array([0.4, 0.0, 0.5]), True),
            (np.array([0.6, 0.0, 0.5]), False),
            (np.array([0.0, 0.0, 1.5]), False),
        ],
        ids=["center", "near-edge", "outside-radial", "outside-axial"],
    )
    def test_cylinder_contains_point(self, point: np.ndarray, expected: bool) -> None:
        """Test point containment."""
        cylinder = Cylinder(
            center=np.array([0.0, 0.0, 0.5]),
            radius=0.5,
            height=1.0,
        )
        assert cylinder.contains_point(point) == expected


# =============================================================================
# ConvexHull Tests
# =============================================================================


class TestConvexHull:
    """Tests for ConvexHull primitive."""

    def test_create_convex_hull(self) -> None:
        """Test convex hull creation."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        hull = ConvexHull(vertices=vertices)
        assert hull.vertices.shape == (4, 3)

    def test_convex_hull_minimum_vertices(self) -> None:
        """Test that fewer than 4 vertices raises error."""
        with pytest.raises(ValueError, match="at least 4 vertices"):
            ConvexHull(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    def test_convex_hull_aabb(self) -> None:
        """Test AABB computation."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        hull = ConvexHull(vertices=vertices)
        min_corner, max_corner = hull.get_aabb()
        assert np.allclose(min_corner, [0, 0, 0])
        assert np.allclose(max_corner, [1, 1, 1])

    def test_convex_hull_support(self) -> None:
        """Test support function."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        hull = ConvexHull(vertices=vertices)
        support = hull.compute_support(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(support, [1, 0, 0])


# =============================================================================
# Distance Computation Tests
# =============================================================================


class TestPrimitiveDistance:
    """Tests for distance computation between primitives."""

    @pytest.mark.parametrize(
        "center_b, expected_distance",
        [
            (np.array([3.0, 0.0, 0.0]), 1.0),
            (np.array([2.0, 0.0, 0.0]), 0.0),
            (np.array([1.0, 0.0, 0.0]), -1.0),
        ],
        ids=["separated", "touching", "overlapping"],
    )
    def test_sphere_sphere_distance(
        self, center_b: np.ndarray, expected_distance: float
    ) -> None:
        """Test distance between sphere pairs at varying separation."""
        sphere_a = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        sphere_b = Sphere(center=center_b, radius=1.0)

        distance, _, _ = compute_primitive_distance(sphere_a, sphere_b)

        assert distance == pytest.approx(expected_distance, abs=1e-6)

    def test_sphere_capsule_distance(self) -> None:
        """Test distance between sphere and capsule."""
        sphere = Sphere(center=np.array([2.0, 0.0, 0.5]), radius=0.5)
        capsule = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.5,
        )

        distance, _, _ = compute_primitive_distance(sphere, capsule)

        # Distance should be 2.0 - 0.5 - 0.5 = 1.0
        assert distance == pytest.approx(1.0, abs=1e-6)

    def test_capsule_capsule_parallel(self) -> None:
        """Test distance between parallel capsules."""
        cap_a = Capsule(
            point_a=np.array([0.0, 0.0, 0.0]),
            point_b=np.array([0.0, 0.0, 1.0]),
            radius=0.1,
        )
        cap_b = Capsule(
            point_a=np.array([1.0, 0.0, 0.0]),
            point_b=np.array([1.0, 0.0, 1.0]),
            radius=0.1,
        )

        distance, _, _ = compute_primitive_distance(cap_a, cap_b)

        # Distance between axes is 1.0, subtract radii
        assert distance == pytest.approx(0.8, abs=1e-6)


class TestPrimitiveCollision:
    """Tests for collision checking between primitives."""

    def test_collision_with_margin(self) -> None:
        """Test collision detection with margin."""
        sphere_a = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        sphere_b = Sphere(center=np.array([2.5, 0.0, 0.0]), radius=1.0)

        # Without margin: not in collision
        assert not check_primitive_collision(sphere_a, sphere_b, margin=0.0)

        # With margin of 0.5: in collision (distance = 0.5)
        assert check_primitive_collision(sphere_a, sphere_b, margin=0.5)

    def test_collision_negative_margin_error(self) -> None:
        """Test that negative margin raises error."""
        sphere_a = Sphere()
        sphere_b = Sphere()

        with pytest.raises(ValueError, match="margin must be non-negative"):
            check_primitive_collision(sphere_a, sphere_b, margin=-0.1)


# =============================================================================
# CollisionPair Tests
# =============================================================================


class TestCollisionPair:
    """Tests for CollisionPair dataclass."""

    def test_create_collision_pair(self) -> None:
        """Test creating collision pair."""
        pair = CollisionPair(body_a="link1", body_b="link2")
        assert pair.body_a == "link1"
        assert pair.body_b == "link2"
        assert pair.enabled

    def test_collision_pair_empty_body_error(self) -> None:
        """Test that empty body name raises error."""
        with pytest.raises(ValueError, match="body_a cannot be empty"):
            CollisionPair(body_a="", body_b="link2")

    def test_collision_pair_equality(self) -> None:
        """Test that pairs are equal regardless of order."""
        pair1 = CollisionPair(body_a="link1", body_b="link2")
        pair2 = CollisionPair(body_a="link2", body_b="link1")
        assert pair1 == pair2

    def test_collision_pair_hash(self) -> None:
        """Test that hash is order-independent."""
        pair1 = CollisionPair(body_a="link1", body_b="link2")
        pair2 = CollisionPair(body_a="link2", body_b="link1")
        assert hash(pair1) == hash(pair2)


# =============================================================================
# CollisionResult Tests
# =============================================================================


class TestCollisionResult:
    """Tests for CollisionResult dataclass."""

    def test_create_collision_result(self) -> None:
        """Test creating collision result."""
        result = CollisionResult(
            in_collision=True,
            collision_pairs=[CollisionPair("link1", "link2")],
            num_contacts=1,
        )
        assert result.in_collision
        assert len(result.collision_pairs) == 1

    @pytest.mark.parametrize(
        "in_collision, pairs, match",
        [
            (True, [], "in_collision=True but no collision"),
            (False, [CollisionPair("a", "b")], "in_collision=False but collision"),
        ],
        ids=["true-no-pairs", "false-with-pairs"],
    )
    def test_collision_result_inconsistent_state(
        self, in_collision: bool, pairs: list, match: str
    ) -> None:
        """Test that inconsistent state raises error."""
        with pytest.raises(ValueError, match=match):
            CollisionResult(in_collision=in_collision, collision_pairs=pairs)


# =============================================================================
# DistanceResult Tests
# =============================================================================


class TestDistanceResult:
    """Tests for DistanceResult dataclass."""

    @pytest.mark.parametrize(
        "distance, in_collision, penetration_depth",
        [
            (0.5, False, 0.0),
            (-0.1, True, 0.1),
            (0.0, False, 0.0),
        ],
        ids=["separated", "penetrating", "touching"],
    )
    def test_distance_result_properties(
        self,
        distance: float,
        in_collision: bool,
        penetration_depth: float,
    ) -> None:
        """Test DistanceResult computed properties."""
        result = DistanceResult(distance=distance)
        assert result.distance == distance
        assert result.in_collision == in_collision
        assert result.penetration_depth == pytest.approx(penetration_depth)


# =============================================================================
# CollisionQuery Tests
# =============================================================================


class TestCollisionQuery:
    """Tests for CollisionQuery configuration."""

    def test_default_query(self) -> None:
        """Test default query configuration."""
        query = CollisionQuery()
        assert query.query_type == CollisionQueryType.BOOLEAN
        assert query.early_exit

    def test_query_invalid_max_distance(self) -> None:
        """Test that non-positive max_distance raises error."""
        with pytest.raises(ValueError, match="max_distance must be positive"):
            CollisionQuery(max_distance=0.0)

    def test_query_should_check_pair(self) -> None:
        """Test pair filtering logic."""
        query = CollisionQuery(
            exclude_pairs=[CollisionPair("link1", "link2")],
        )

        # Excluded pair should not be checked
        assert not query.should_check_pair(CollisionPair("link1", "link2"))

        # Other pairs should be checked
        assert query.should_check_pair(CollisionPair("link2", "link3"))

    def test_query_include_list(self) -> None:
        """Test include list filtering."""
        query = CollisionQuery(
            include_pairs=[CollisionPair("link1", "link2")],
        )

        assert query.should_check_pair(CollisionPair("link1", "link2"))
        assert not query.should_check_pair(CollisionPair("link2", "link3"))


# =============================================================================
# CollisionChecker Tests
# =============================================================================


class TestCollisionChecker:
    """Tests for CollisionChecker class."""

    @pytest.fixture
    def engine(self) -> MockCollisionEngine:
        """Create mock engine."""
        return MockCollisionEngine()

    @pytest.fixture
    def checker(self, engine: MockCollisionEngine) -> CollisionChecker:
        """Create collision checker."""
        return CollisionChecker(engine)

    def test_create_checker(self, engine: MockCollisionEngine) -> None:
        """Test creating collision checker."""
        checker = CollisionChecker(engine)
        assert checker is not None

    def test_checker_requires_collision_capable(self) -> None:
        """Test that non-CollisionCapable engine raises error."""

        class NotCollisionCapable:
            pass

        with pytest.raises(TypeError, match="CollisionCapable"):
            CollisionChecker(NotCollisionCapable())  # type: ignore

    def test_check_collision_no_collision(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test collision check with no collision."""
        q = np.zeros(7)
        result = checker.check_collision(q)
        assert not result.in_collision

    def test_check_collision_with_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test collision with environment obstacle."""
        # Add obstacle at link position
        obstacle = Sphere(center=np.array([0.0, 0.0, 0.5]), radius=0.15)
        checker.add_environment_primitive("obstacle", obstacle)

        q = np.zeros(7)
        result = checker.check_collision(q)

        # Should detect collision with link1
        assert result.in_collision

    def test_check_collision_invalid_config(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test that infinite configuration raises error."""
        q = np.array([np.inf, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="must be finite"):
            checker.check_collision(q)

    def test_compute_distance(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test distance computation."""
        q = np.zeros(7)
        result = checker.compute_distance(q)

        # Should return some distance (bodies are separated)
        assert (
            result.distance > 0
            or result.distance == float("inf")
            or result.closest_pair is None
        )

    def test_compute_distance_with_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test distance to environment obstacle."""
        # Add obstacle near link
        obstacle = Sphere(center=np.array([0.5, 0.0, 0.5]), radius=0.1)
        checker.add_environment_primitive("obstacle", obstacle)

        q = np.zeros(7)
        result = checker.compute_distance(q)

        # Distance should be positive (separated)
        assert result.distance > 0

    def test_check_path_collision(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test path collision checking."""
        q_start = np.zeros(7)
        q_end = np.ones(7)

        is_free, collision_t = checker.check_path_collision(q_start, q_end)

        # Path should be collision-free (no obstacles)
        assert is_free
        assert collision_t is None

    def test_check_path_with_obstacle(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test path collision with obstacle."""
        # Add obstacle in path
        obstacle = Sphere(center=np.array([0.05, 0.05, 0.5]), radius=0.1)
        checker.add_environment_primitive("obstacle", obstacle)

        q_start = np.zeros(7)
        q_end = np.ones(7)

        is_free, collision_t = checker.check_path_collision(q_start, q_end)

        # Path should have collision
        assert not is_free
        assert collision_t is not None
        assert 0.0 <= collision_t <= 1.0

    def test_add_remove_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test adding and removing environment primitives."""
        obstacle = Sphere(center=np.zeros(3), radius=1.0)

        checker.add_environment_primitive("test", obstacle)
        assert checker.remove_environment_primitive("test")
        assert not checker.remove_environment_primitive("nonexistent")

    def test_clear_environment(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test clearing all environment primitives."""
        checker.add_environment_primitive("obs1", Sphere())
        checker.add_environment_primitive("obs2", Sphere())

        checker.clear_environment()

        # Should not find any environment collisions now
        q = np.zeros(7)
        result = checker.check_collision(q)
        # Only self-collision pairs checked now
        assert isinstance(result, CollisionResult)

    def test_disable_enable_collision_pair(
        self,
        checker: CollisionChecker,
    ) -> None:
        """Test disabling and enabling collision pairs."""
        checker.disable_collision_pair("link1", "link2")
        pairs = checker.get_collision_pairs()
        assert CollisionPair("link1", "link2") not in pairs

        checker.enable_collision_pair("link1", "link2")
        pairs = checker.get_collision_pairs()
        assert CollisionPair("link1", "link2") in pairs


class TestCollisionCheckerConfig:
    """Tests for CollisionCheckerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = CollisionCheckerConfig()
        assert config.default_margin == 0.01
        assert config.use_broad_phase
        assert config.max_contacts == 100

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CollisionCheckerConfig(
            default_margin=0.05,
            use_broad_phase=False,
            max_contacts=50,
        )
        assert config.default_margin == 0.05
        assert not config.use_broad_phase

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"default_margin": -0.1}, "default_margin must be non-negative"),
            ({"max_contacts": 0}, "max_contacts must be positive"),
        ],
        ids=["negative-margin", "zero-max-contacts"],
    )
    def test_invalid_config(self, kwargs: dict, match: str) -> None:
        """Test invalid configuration values."""
        with pytest.raises(ValueError, match=match):
            CollisionCheckerConfig(**kwargs)
