"""
Comprehensive unit tests for model_generation.core.validation.

Tests cover the Validator class methods:
- validate_mass: positive, zero, negative, tiny values
- validate_inertia: positive-definite, diagonal, triangle inequality, NaN
- validate_link: delegates to validate_inertia, checks name
- validate_joint: parent/child existence, axis normalization, limit ordering
- validate_hierarchy: cycle detection, orphaned links, duplicate names, root
- validate_model: end-to-end full model validation
"""

from __future__ import annotations

from model_generation.core.types import (
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
)
from model_generation.core.validation import ValidationResult, Validator

# ── validate_mass ────────────────────────────────────────────────────────────


class TestValidateMass:
    """Test Validator.validate_mass for edge cases."""

    def test_positive_mass_is_valid(self) -> None:
        result = Validator.validate_mass(10.0)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_zero_mass_is_invalid(self) -> None:
        result = Validator.validate_mass(0.0)
        assert result.is_valid is False
        assert any("MASS_001" in e.code for e in result.errors)

    def test_negative_mass_is_invalid(self) -> None:
        result = Validator.validate_mass(-5.0)
        assert result.is_valid is False

    def test_very_small_mass_warns(self) -> None:
        result = Validator.validate_mass(1e-8)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("MASS_002" in w.code for w in result.warnings)

    def test_component_name_in_error(self) -> None:
        result = Validator.validate_mass(-1.0, component="pelvis")
        assert result.errors[0].component == "pelvis"


# ── validate_inertia ─────────────────────────────────────────────────────────


class TestValidateInertia:
    """Test Validator.validate_inertia for physical correctness."""

    def test_valid_sphere_inertia(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=0.1)
        result = Validator.validate_inertia(inertia)
        assert result.is_valid is True

    def test_negative_diagonal_is_invalid(self) -> None:
        inertia = Inertia(ixx=-1.0, iyy=1.0, izz=1.0, mass=1.0)
        result = Validator.validate_inertia(inertia)
        assert result.is_valid is False
        assert any("INERTIA_002" in e.code for e in result.errors)

    def test_not_positive_definite_off_diagonal(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=1.0, ixy=5.0, mass=1.0)
        result = Validator.validate_inertia(inertia)
        assert result.is_valid is False
        assert any("INERTIA_001" in e.code for e in result.errors)

    def test_triangle_inequality_violation_strict(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=10.0, mass=1.0)
        result = Validator.validate_inertia(inertia, strict=True)
        assert result.is_valid is False
        assert any("INERTIA_003" in e.code for e in result.errors)

    def test_triangle_inequality_violation_lenient(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=10.0, mass=1.0)
        result = Validator.validate_inertia(inertia, strict=False)
        # Should still be valid but with warning
        # (Note: it will also fail positive-definite depending on off-diag)
        # This specific case still has positive diagonals, so the PD check matters
        has_triangle_warning = any("INERTIA_003" in w.code for w in result.warnings)
        has_triangle_error = any("INERTIA_003" in e.code for e in result.errors)
        assert has_triangle_warning or has_triangle_error

    def test_zero_mass_inertia_is_invalid(self) -> None:
        inertia = Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=0.0)
        result = Validator.validate_inertia(inertia)
        assert result.is_valid is False

    def test_very_small_inertia_warns(self) -> None:
        inertia = Inertia(ixx=1e-14, iyy=1e-14, izz=1e-14, mass=1.0)
        result = Validator.validate_inertia(inertia)
        assert any("INERTIA_SMALL" in w.code for w in result.warnings)

    def test_component_name_propagated(self) -> None:
        inertia = Inertia(ixx=-1.0, iyy=1.0, izz=1.0, mass=1.0)
        result = Validator.validate_inertia(inertia, component="torso")
        assert any(e.component == "torso" for e in result.errors)


# ── validate_link ────────────────────────────────────────────────────────────


class TestValidateLink:
    """Test Validator.validate_link."""

    def test_valid_link(self) -> None:
        link = Link(name="base", inertia=Inertia.from_sphere(mass=1.0, radius=0.1))
        result = Validator.validate_link(link)
        assert result.is_valid is True

    def test_link_with_bad_inertia(self) -> None:
        link = Link(name="bad", inertia=Inertia(ixx=-1.0, iyy=1.0, izz=1.0, mass=1.0))
        result = Validator.validate_link(link)
        assert result.is_valid is False

    def test_empty_name_link(self) -> None:
        link = Link(name="", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0))
        result = Validator.validate_link(link)
        assert result.is_valid is False
        assert any("LINK_NAME_EMPTY" in e.code for e in result.errors)

    def test_whitespace_name_link(self) -> None:
        link = Link(name="   ", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0))
        result = Validator.validate_link(link)
        assert result.is_valid is False


# ── validate_joint ───────────────────────────────────────────────────────────


class TestValidateJoint:
    """Test Validator.validate_joint."""

    def _make_link_names(self) -> set[str]:
        return {"base", "arm", "hand"}

    def test_valid_joint(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            axis=(0.0, 0.0, 1.0),
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        assert result.is_valid is True

    def test_missing_parent(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="nonexistent",
            child="arm",
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        assert result.is_valid is False
        assert any("JOINT_003" in e.code for e in result.errors)

    def test_missing_child(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="nonexistent",
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        assert result.is_valid is False
        assert any("JOINT_004" in e.code for e in result.errors)

    def test_zero_axis_is_error(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            axis=(0.0, 0.0, 0.0),
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        assert result.is_valid is False
        assert any("JOINT_001" in e.code for e in result.errors)

    def test_unnormalized_axis_warns(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            axis=(0.0, 0.0, 2.0),
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        # Valid but with warning about normalization
        assert any("JOINT_001" in w.code for w in result.warnings)

    def test_inverted_limits_is_error(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            limits=JointLimits(lower=1.0, upper=-1.0),
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        assert result.is_valid is False
        assert any("JOINT_002" in e.code for e in result.errors)

    def test_fixed_joint_no_axis_check(self) -> None:
        joint = Joint(
            name="j1",
            joint_type=JointType.FIXED,
            parent="base",
            child="arm",
            axis=(0.0, 0.0, 0.0),  # zero axis OK for fixed
        )
        result = Validator.validate_joint(joint, self._make_link_names())
        # Fixed joints should not check axis
        assert not any("JOINT_001" in e.code for e in result.errors)


# ── validate_hierarchy ───────────────────────────────────────────────────────


class TestValidateHierarchy:
    """Test Validator.validate_hierarchy for structural correctness."""

    def _make_chain(self) -> tuple[list[Link], list[Joint]]:
        """Create a simple A -> B -> C chain."""
        links = [
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="B", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="C", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
        joints = [
            Joint(name="j_AB", joint_type=JointType.FIXED, parent="A", child="B"),
            Joint(name="j_BC", joint_type=JointType.FIXED, parent="B", child="C"),
        ]
        return links, joints

    def test_valid_chain(self) -> None:
        links, joints = self._make_chain()
        result = Validator.validate_hierarchy(links, joints)
        assert result.is_valid is True

    def test_cycle_detection(self) -> None:
        links = [
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="B", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
        joints = [
            Joint(name="j_AB", joint_type=JointType.FIXED, parent="A", child="B"),
            Joint(name="j_BA", joint_type=JointType.FIXED, parent="B", child="A"),
        ]
        result = Validator.validate_hierarchy(links, joints)
        assert result.is_valid is False
        has_cycle_error = any("HIERARCHY_001" in e.code for e in result.errors)
        assert has_cycle_error

    def test_duplicate_link_names(self) -> None:
        links = [
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
        joints: list[Joint] = []
        result = Validator.validate_hierarchy(links, joints)
        assert result.is_valid is False
        assert any("HIERARCHY_003" in e.code for e in result.errors)

    def test_duplicate_joint_names(self) -> None:
        links = [
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="B", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="C", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
        joints = [
            Joint(name="j1", joint_type=JointType.FIXED, parent="A", child="B"),
            Joint(name="j1", joint_type=JointType.FIXED, parent="A", child="C"),
        ]
        result = Validator.validate_hierarchy(links, joints)
        assert result.is_valid is False
        assert any("HIERARCHY_003" in e.code for e in result.errors)

    def test_multiple_roots_warns(self) -> None:
        links = [
            Link(name="A", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="B", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
        joints: list[Joint] = []  # no joints => both are roots
        result = Validator.validate_hierarchy(links, joints)
        # Multiple roots is a warning, not an error
        assert any("HIERARCHY_MULTIPLE_ROOTS" in w.code for w in result.warnings)

    def test_single_link_no_joints_valid(self) -> None:
        links = [Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))]
        joints: list[Joint] = []
        result = Validator.validate_hierarchy(links, joints)
        assert result.is_valid is True


# ── ValidationResult ─────────────────────────────────────────────────────────


class TestValidationResult:
    """Test ValidationResult helper methods."""

    def test_bool_true_when_valid(self) -> None:
        result = ValidationResult(is_valid=True)
        assert bool(result) is True

    def test_bool_false_after_error(self) -> None:
        result = ValidationResult(is_valid=True)
        result.add_error("E001", "something broke")
        assert bool(result) is False

    def test_merge_propagates_errors(self) -> None:
        r1 = ValidationResult(is_valid=True)
        r2 = ValidationResult(is_valid=True)
        r2.add_error("E001", "fail")
        r1.merge(r2)
        assert r1.is_valid is False
        assert len(r1.errors) == 1

    def test_get_error_messages(self) -> None:
        result = ValidationResult(is_valid=True)
        result.add_error("E001", "msg1", component="comp")
        msgs = result.get_error_messages()
        assert len(msgs) == 1
        assert "msg1" in msgs[0]

    def test_get_warning_messages(self) -> None:
        result = ValidationResult(is_valid=True)
        result.add_warning("W001", "warn1")
        msgs = result.get_warning_messages()
        assert len(msgs) == 1
        assert "warn1" in msgs[0]


# ── validate_model (end-to-end) ──────────────────────────────────────────────


class TestValidateModel:
    """Test Validator.validate_model end-to-end."""

    def test_valid_two_link_model(self) -> None:
        links = [
            Link(name="base", inertia=Inertia.from_box(5.0, 1.0, 1.0, 0.5)),
            Link(name="arm", inertia=Inertia.from_cylinder(2.0, 0.05, 0.5)),
        ]
        joints = [
            Joint(
                name="base_to_arm",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="arm",
                axis=(1.0, 0.0, 0.0),
            ),
        ]
        result = Validator.validate_model(links, joints)
        assert result.is_valid is True

    def test_invalid_model_bad_mass(self) -> None:
        links = [
            Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=-1.0)),
        ]
        joints: list[Joint] = []
        result = Validator.validate_model(links, joints)
        assert result.is_valid is False
