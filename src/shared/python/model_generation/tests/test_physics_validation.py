"""Tests for Physics Validation module."""

from dataclasses import dataclass

import pytest
from model_generation.core.physics_validation import (
    PhysicsValidator,
)


@dataclass
class MockOrigin:
    """Mock origin for testing."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class MockInertia:
    """Mock inertia for testing."""

    mass: float = 1.0
    ixx: float = 1.0
    iyy: float = 1.0
    izz: float = 1.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


@dataclass
class MockInertial:
    """Mock inertial for testing."""

    mass: float = 1.0
    origin: MockOrigin = None

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = MockOrigin()


@dataclass
class MockGeometry:
    """Mock geometry for testing."""

    radius: float = 0.1


@dataclass
class MockCollision:
    """Mock collision for testing."""

    geometry: MockGeometry = None
    origin: MockOrigin = None

    def __post_init__(self) -> None:
        if self.geometry is None:
            self.geometry = MockGeometry()
        if self.origin is None:
            self.origin = MockOrigin()


@dataclass
class MockLink:
    """Mock link for testing."""

    name: str
    inertial: MockInertial = None
    collision: MockCollision = None

    def __post_init__(self) -> None:
        if self.inertial is None:
            self.inertial = MockInertial()


class TestInertiaValidation:
    """Tests for inertia tensor validation."""

    @pytest.fixture
    def validator(self) -> PhysicsValidator:
        return PhysicsValidator()

    def test_valid_inertia(self, validator: PhysicsValidator) -> None:
        """Test validation of valid inertia tensor."""
        inertia = MockInertia(mass=1.0, ixx=1.0, iyy=1.0, izz=1.0)
        result = validator.validate_inertia_tensor(inertia)

        assert result.is_valid
        assert result.is_symmetric
        assert result.is_positive_definite
        assert result.satisfies_triangle_inequality

    def test_negative_diagonal(self, validator: PhysicsValidator) -> None:
        """Test detection of negative diagonal elements."""
        inertia = MockInertia(mass=1.0, ixx=-1.0, iyy=1.0, izz=1.0)
        result = validator.validate_inertia_tensor(inertia)

        assert not result.is_valid
        assert not result.is_positive_definite

    def test_triangle_inequality_violation(self, validator: PhysicsValidator) -> None:
        """Test detection of triangle inequality violation."""
        # Izz > Ixx + Iyy violates triangle inequality
        inertia = MockInertia(mass=1.0, ixx=0.1, iyy=0.1, izz=1.0)
        result = validator.validate_inertia_tensor(inertia)

        assert not result.satisfies_triangle_inequality
        assert len(result.warnings) > 0

    def test_eigenvalue_computation(self, validator: PhysicsValidator) -> None:
        """Test eigenvalue computation."""
        inertia = MockInertia(mass=1.0, ixx=1.0, iyy=2.0, izz=3.0)
        result = validator.validate_inertia_tensor(inertia)

        assert result.eigenvalues is not None
        assert len(result.eigenvalues) == 3
        # Eigenvalues should be approximately the diagonal elements for diagonal matrix
        assert sorted(result.eigenvalues) == pytest.approx([1.0, 2.0, 3.0], rel=0.01)

    def test_condition_number_warning(self, validator: PhysicsValidator) -> None:
        """Test warning for high condition number."""
        # Very different diagonal elements
        inertia = MockInertia(mass=1.0, ixx=1e-6, iyy=1.0, izz=1.0)
        result = validator.validate_inertia_tensor(inertia)

        assert result.condition_number is not None
        assert result.condition_number > 1e5
        # Should have warning about numerical stability
        assert any("condition number" in w.lower() for w in result.warnings)


class TestStaticStability:
    """Tests for static stability analysis."""

    @pytest.fixture
    def validator(self) -> PhysicsValidator:
        return PhysicsValidator()

    def test_empty_links(self, validator: PhysicsValidator) -> None:
        """Test stability with no links."""
        result = validator.check_static_stability([])

        assert not result.is_stable

    def test_single_link_stability(self, validator: PhysicsValidator) -> None:
        """Test stability with single link."""
        link = MockLink(name="base", inertial=MockInertial(mass=1.0))
        result = validator.check_static_stability([link])

        assert result.center_of_mass == (0.0, 0.0, 0.0)

    def test_offset_com(self, validator: PhysicsValidator) -> None:
        """Test COM calculation with offset links."""
        link1 = MockLink(
            name="link1",
            inertial=MockInertial(
                mass=1.0,
                origin=MockOrigin(x=1.0, y=0.0, z=0.0),
            ),
        )
        link2 = MockLink(
            name="link2",
            inertial=MockInertial(
                mass=1.0,
                origin=MockOrigin(x=-1.0, y=0.0, z=0.0),
            ),
        )

        result = validator.check_static_stability([link1, link2])

        # COM should be at origin (average of +1 and -1)
        assert result.center_of_mass == pytest.approx((0.0, 0.0, 0.0), abs=0.01)


class TestCollisionCheck:
    """Tests for collision geometry validation."""

    @pytest.fixture
    def validator(self) -> PhysicsValidator:
        return PhysicsValidator()

    def test_no_collision(self, validator: PhysicsValidator) -> None:
        """Test links without collision geometry."""
        link = MockLink(name="link1")
        link.collision = None
        result = validator.check_collision_geometry([link])

        assert not result.has_self_intersection

    def test_separated_collisions(self, validator: PhysicsValidator) -> None:
        """Test well-separated collision geometries."""
        link1 = MockLink(
            name="link1",
            collision=MockCollision(
                geometry=MockGeometry(radius=0.1),
                origin=MockOrigin(x=0.0),
            ),
        )
        link2 = MockLink(
            name="link2",
            collision=MockCollision(
                geometry=MockGeometry(radius=0.1),
                origin=MockOrigin(x=1.0),  # 1m away
            ),
        )

        result = validator.check_collision_geometry([link1, link2])

        assert not result.has_self_intersection
        assert result.min_separation > 0

    def test_intersecting_collisions(self, validator: PhysicsValidator) -> None:
        """Test overlapping collision geometries."""
        link1 = MockLink(
            name="link1",
            collision=MockCollision(
                geometry=MockGeometry(radius=0.5),
                origin=MockOrigin(x=0.0),
            ),
        )
        link2 = MockLink(
            name="link2",
            collision=MockCollision(
                geometry=MockGeometry(radius=0.5),
                origin=MockOrigin(x=0.5),  # Overlaps
            ),
        )

        result = validator.check_collision_geometry([link1, link2])

        assert result.has_self_intersection
        assert ("link1", "link2") in result.penetration_pairs


class TestCompletePhysicsValidation:
    """Tests for complete physics validation."""

    @pytest.fixture
    def validator(self) -> PhysicsValidator:
        return PhysicsValidator()

    def test_valid_model(self, validator: PhysicsValidator) -> None:
        """Test validation of valid model."""
        link = MockLink(
            name="base",
            inertial=MockInertial(mass=1.0),
            collision=MockCollision(),
        )

        result = validator.validate_physics([link])

        assert result.is_valid
        assert result.total_mass == 1.0

    def test_skip_stability_check(self, validator: PhysicsValidator) -> None:
        """Test skipping stability check."""
        link = MockLink(name="base")

        result = validator.validate_physics(
            [link],
            check_stability=False,
        )

        assert result.stability is None

    def test_skip_collision_check(self, validator: PhysicsValidator) -> None:
        """Test skipping collision check."""
        link = MockLink(name="base")

        result = validator.validate_physics(
            [link],
            check_collisions=False,
        )

        assert result.collision is None
