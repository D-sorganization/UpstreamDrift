"""Tests for model_generation.core.validation."""

from __future__ import annotations

import pytest
from model_generation.core.types import (
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
)
from model_generation.core.validation import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    Validator,
)


def test_validation_error_str_with_component() -> None:
    e = ValidationError(code="C", message="msg", component="comp")
    assert str(e) == "[comp] C: msg"


def test_validation_error_str_no_component() -> None:
    assert "C: msg" in str(ValidationError(code="C", message="msg"))


def test_validation_warning_str() -> None:
    w = ValidationWarning(code="C", message="hi", component="c")
    assert "Warning" in str(w)


def test_result_add_error_marks_invalid() -> None:
    r = ValidationResult(is_valid=True)
    r.add_error("C", "msg")
    assert not r.is_valid
    assert bool(r) is False
    assert r.get_error_messages() == ["C: msg"]


def test_result_add_warning_keeps_valid() -> None:
    r = ValidationResult(is_valid=True)
    r.add_warning("C", "msg")
    assert r.is_valid
    assert r.get_warning_messages()


def test_result_add_error_none_code_raises() -> None:
    r = ValidationResult(is_valid=True)
    with pytest.raises(ValueError):
        r.add_error(None, "msg")  # type: ignore[arg-type]


def test_result_merge() -> None:
    a = ValidationResult(is_valid=True)
    b = ValidationResult(is_valid=True)
    b.add_error("X", "err")
    b.add_warning("Y", "warn")
    a.merge(b)
    assert not a.is_valid
    assert len(a.errors) == 1
    assert len(a.warnings) == 1


def test_result_merge_none_raises() -> None:
    a = ValidationResult(is_valid=True)
    with pytest.raises(ValueError):
        a.merge(None)  # type: ignore[arg-type]


def test_validate_mass_positive() -> None:
    r = Validator.validate_mass(1.0)
    assert r.is_valid


def test_validate_mass_zero_fails() -> None:
    r = Validator.validate_mass(0.0)
    assert not r.is_valid


def test_validate_mass_too_small_warns() -> None:
    r = Validator.validate_mass(1e-9)
    assert r.is_valid
    assert any(w.code == Validator.MASS_TOO_SMALL for w in r.warnings)


def test_validate_inertia_good() -> None:
    r = Validator.validate_inertia(Inertia(1, 1, 1, mass=1.0))
    assert r.is_valid


def test_validate_inertia_negative_diag() -> None:
    r = Validator.validate_inertia(Inertia(-1, 1, 1, mass=1.0))
    assert not r.is_valid
    assert any(e.code == Validator.INERTIA_DIAGONAL_NEGATIVE for e in r.errors)


def test_validate_inertia_not_positive_definite() -> None:
    bad = Inertia(ixx=1, iyy=1, izz=1, ixy=2.0, mass=1.0)
    r = Validator.validate_inertia(bad)
    assert not r.is_valid
    assert any(e.code == Validator.INERTIA_NOT_POSITIVE_DEFINITE for e in r.errors)


def test_validate_inertia_triangle_inequality_strict() -> None:
    bad = Inertia(ixx=1, iyy=1, izz=100, mass=1.0)
    r = Validator.validate_inertia(bad, strict=True)
    assert any(e.code == Validator.INERTIA_TRIANGLE_INEQUALITY for e in r.errors)


def test_validate_inertia_triangle_inequality_nonstrict_warn() -> None:
    bad = Inertia(ixx=1, iyy=1, izz=100, mass=1.0)
    r = Validator.validate_inertia(bad, strict=False)
    assert any(w.code == Validator.INERTIA_TRIANGLE_INEQUALITY for w in r.warnings)


def test_validate_inertia_small_warns() -> None:
    r = Validator.validate_inertia(Inertia(1e-13, 1e-13, 1e-13, mass=1.0))
    assert any(w.code == "INERTIA_SMALL" for w in r.warnings)


def test_validate_link_ok() -> None:
    link = Link(name="link", inertia=Inertia(1, 1, 1, mass=1.0))
    r = Validator.validate_link(link)
    assert r.is_valid


def test_validate_link_empty_name() -> None:
    link = Link(name="", inertia=Inertia(1, 1, 1, mass=1.0))
    r = Validator.validate_link(link)
    assert not r.is_valid
    assert any(e.code == "LINK_NAME_EMPTY" for e in r.errors)


def test_validate_joint_missing_parent_and_child() -> None:
    j = Joint(name="j", joint_type=JointType.REVOLUTE, parent="x", child="y")
    r = Validator.validate_joint(j, link_names={"a"})
    assert not r.is_valid
    codes = {e.code for e in r.errors}
    assert Validator.JOINT_MISSING_PARENT in codes
    assert Validator.JOINT_MISSING_CHILD in codes


def test_validate_joint_axis_not_normalized_warns() -> None:
    j = Joint(
        name="j", joint_type=JointType.REVOLUTE, parent="a", child="b", axis=(0, 0, 2)
    )
    r = Validator.validate_joint(j, link_names={"a", "b"})
    # not normalized but not zero: warning
    assert any(w.code == Validator.JOINT_INVALID_AXIS for w in r.warnings)


def test_validate_joint_axis_zero_errors() -> None:
    j = Joint(
        name="j", joint_type=JointType.REVOLUTE, parent="a", child="b", axis=(0, 0, 0)
    )
    r = Validator.validate_joint(j, link_names={"a", "b"})
    assert any(e.code == Validator.JOINT_INVALID_AXIS for e in r.errors)


def test_validate_joint_inverted_limits() -> None:
    j = Joint(
        name="j",
        joint_type=JointType.REVOLUTE,
        parent="a",
        child="b",
        limits=JointLimits(lower=1.0, upper=-1.0),
    )
    r = Validator.validate_joint(j, link_names={"a", "b"})
    assert any(e.code == Validator.JOINT_INVALID_LIMITS for e in r.errors)


def test_validate_hierarchy_duplicate_link_names() -> None:
    links = [Link(name="x"), Link(name="x")]
    r = Validator.validate_hierarchy(links, [])
    assert not r.is_valid
    assert any(e.code == Validator.HIERARCHY_DUPLICATE for e in r.errors)


def test_validate_hierarchy_circular() -> None:
    links = [Link(name="a"), Link(name="b")]
    joints = [
        Joint(name="j1", joint_type=JointType.FIXED, parent="a", child="b"),
        Joint(name="j2", joint_type=JointType.FIXED, parent="b", child="a"),
    ]
    r = Validator.validate_hierarchy(links, joints)
    # No root (all are children) -> circular dependency error
    assert any(e.code == Validator.HIERARCHY_CIRCULAR for e in r.errors)


def test_validate_hierarchy_multiple_roots_warn() -> None:
    links = [Link(name="a"), Link(name="b"), Link(name="c")]
    joints = [
        Joint(name="j", joint_type=JointType.FIXED, parent="a", child="c"),
    ]
    r = Validator.validate_hierarchy(links, joints)
    assert any(w.code == "HIERARCHY_MULTIPLE_ROOTS" for w in r.warnings)


def test_validate_model_complete_ok() -> None:
    links = [
        Link(name="a", inertia=Inertia(1, 1, 1, mass=1.0)),
        Link(name="b", inertia=Inertia(1, 1, 1, mass=1.0)),
    ]
    joints = [Joint(name="j", joint_type=JointType.FIXED, parent="a", child="b")]
    r = Validator.validate_model(links, joints)
    assert r.is_valid


def test_validate_model_propagates_link_errors() -> None:
    links = [Link(name="bad", inertia=Inertia(-1, 1, 1, mass=1.0))]
    r = Validator.validate_model(links, [])
    assert not r.is_valid
