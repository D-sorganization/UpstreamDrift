"""Tests for the humanoid URDF contract validator (issue #2802)."""

from __future__ import annotations

from pathlib import Path

import biomechanics.humanoid_urdf_contracts as urdf_contracts
import pytest
from biomechanics.humanoid_urdf_contracts import (
    ContractViolation,
    ValidationReport,
    validate_humanoid_urdf,
)

pytestmark = pytest.mark.unit


def _build_urdf(
    *,
    joints: list[tuple[str, str, str]] | None = None,
    links: list[tuple[str, float, tuple[float, float, float]]] | None = None,
) -> str:
    """Assemble a minimal URDF XML string from joints and links.

    ``joints`` items are ``(name, parent, child)``.
    ``links`` items are ``(name, mass, (ixx, iyy, izz))``.
    """
    joints = joints or []
    links = links or []
    link_xml = []
    for name, mass, (ixx, iyy, izz) in links:
        link_xml.append(
            f'<link name="{name}">'
            f"<inertial>"
            f'<mass value="{mass}"/>'
            f'<inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}"'
            f' ixy="0" ixz="0" iyz="0"/>'
            f"</inertial>"
            f"</link>"
        )
    joint_xml = [
        f'<joint name="{n}" type="revolute">'
        f'<parent link="{p}"/><child link="{c}"/>'
        f"</joint>"
        for n, p, c in joints
    ]
    return (
        '<?xml version="1.0"?>'
        '<robot name="t">' + "".join(link_xml) + "".join(joint_xml) + "</robot>"
    )


def _valid_joint_set() -> list[tuple[str, str, str]]:
    stems = ("shoulder", "elbow", "hip", "knee", "ankle")
    return [
        (f"{side}_{s}", "pelvis", f"{side}_{s}_link")
        for s in stems
        for side in ("left", "right")
    ]


def _valid_link_set() -> list[tuple[str, float, tuple[float, float, float]]]:
    # Symmetric bilateral pairs plus a central pelvis.
    out: list[tuple[str, float, tuple[float, float, float]]] = [
        ("pelvis", 8.0, (0.03, 0.02, 0.03)),
    ]
    for stem, mass in (
        ("shoulder", 2.0),
        ("elbow", 1.5),
        ("hip", 6.0),
        ("knee", 3.0),
        ("ankle", 1.0),
    ):
        for side in ("left", "right"):
            out.append((f"{side}_{stem}_link", mass, (0.01, 0.01, 0.01)))
    return out


def test_valid_humanoid_passes() -> None:
    urdf = _build_urdf(joints=_valid_joint_set(), links=_valid_link_set())
    report = validate_humanoid_urdf(urdf)
    assert isinstance(report, ValidationReport)
    assert report.ok, report.describe()
    assert report.violations == []


def test_missing_left_hip_is_reported() -> None:
    joints = [j for j in _valid_joint_set() if j[0] != "left_hip"]
    urdf = _build_urdf(joints=joints, links=_valid_link_set())
    report = validate_humanoid_urdf(urdf)
    assert not report.ok
    assert any(
        v.category == "missing_joint" and "left_hip" in v.message
        for v in report.violations
    )


def test_negative_mass_is_reported() -> None:
    links = _valid_link_set()
    links[0] = ("pelvis", -1.0, (0.03, 0.02, 0.03))
    urdf = _build_urdf(joints=_valid_joint_set(), links=links)
    report = validate_humanoid_urdf(urdf)
    assert not report.ok
    assert any(v.category == "invalid_mass" for v in report.violations)


def test_negative_inertia_is_reported() -> None:
    links = _valid_link_set()
    links[0] = ("pelvis", 8.0, (-0.03, 0.02, 0.03))
    urdf = _build_urdf(joints=_valid_joint_set(), links=links)
    report = validate_humanoid_urdf(urdf)
    assert not report.ok
    assert any(v.category == "invalid_inertia" for v in report.violations)


def test_bilateral_mass_asymmetry_is_reported() -> None:
    links = _valid_link_set()
    # Inflate right_hip_link mass to break symmetry
    links = [
        (
            ("right_hip_link", 20.0, (0.01, 0.01, 0.01))
            if n == "right_hip_link"
            else (n, m, i)
        )
        for (n, m, i) in links
    ]
    urdf = _build_urdf(joints=_valid_joint_set(), links=links)
    report = validate_humanoid_urdf(urdf)
    assert not report.ok
    assert any(v.category == "asymmetric_limbs" for v in report.violations)


def test_triangle_inequality_violation_is_reported() -> None:
    links = _valid_link_set()
    links[0] = ("pelvis", 8.0, (0.001, 0.001, 1.0))
    urdf = _build_urdf(joints=_valid_joint_set(), links=links)
    report = validate_humanoid_urdf(urdf)
    assert not report.ok
    assert any(
        v.category == "invalid_inertia" and "triangle" in v.message
        for v in report.violations
    )


def test_describe_mentions_ok_when_valid() -> None:
    urdf = _build_urdf(joints=_valid_joint_set(), links=_valid_link_set())
    assert "OK" in validate_humanoid_urdf(urdf).describe()


def test_describe_lists_each_violation_when_invalid() -> None:
    report = ValidationReport(
        ok=False,
        violations=[
            ContractViolation("missing_joint", "left_hip absent"),
            ContractViolation("invalid_mass", "pelvis mass=-1"),
        ],
    )
    text = report.describe()
    assert "FAIL" in text
    assert "left_hip absent" in text
    assert "pelvis mass=-1" in text


def test_invalid_xml_raises() -> None:
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        validate_humanoid_urdf("<<not xml>>")


def test_existing_path_permission_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem errors from ET.parse(path) must surface for real paths."""
    monkeypatch.setattr(urdf_contracts.Path, "exists", lambda _self: True)
    monkeypatch.setattr(
        urdf_contracts.ET,
        "parse",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(PermissionError, match="denied"):
        validate_humanoid_urdf(Path("robot.urdf"))
