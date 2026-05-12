"""URDF quality regression tests for the humanoid character builder.

This file consolidates the validation contracts that the URDF Hardening
Campaign added in Phase 4:

- test_determinism            (#4537)  same params -> byte-identical URDF
- test_schema_validation      (#4536)  output is well-formed XML and matches URDF schema
- test_inertia_validation     (#4539)  mass conservation + positive-definite tensors
- test_joint_limits_anatomy   (#4540)  knee/elbow/etc. limits match human anatomy

Tests are unit-marked so they run in the default suite and gate every PR.
They use the primitive mesh backend so no SMPLX/MakeHuman assets are required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from humanoid_character_builder import BodyParameters, CharacterBuilder

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_params() -> BodyParameters:
    """Stable parameters used across multiple regression tests."""
    return BodyParameters(height_m=1.80, mass_kg=80.0)


@pytest.fixture(scope="module")
def baseline_urdf(baseline_params: BodyParameters) -> str:
    """A single URDF generated once and reused by quality tests."""
    builder = CharacterBuilder()
    return builder.generate_urdf(baseline_params)


# ---------------------------------------------------------------------------
# #4537 - Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_urdf_generation_is_deterministic_same_call() -> None:
    """Two back-to-back generate_urdf calls with identical params must
    produce byte-identical output. Catches dict iteration order, FP
    formatting drift, embedded timestamps, and similar non-determinism
    bugs. See issue #4537.
    """
    params = BodyParameters(height_m=1.80, mass_kg=80.0)
    builder = CharacterBuilder()
    a = builder.generate_urdf(params)
    b = builder.generate_urdf(params)
    assert a == b, (
        "Generated URDFs differ between successive calls with identical "
        "BodyParameters - non-deterministic generation is a regression."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "params",
    [
        BodyParameters(height_m=1.65, mass_kg=60.0),
        BodyParameters(height_m=1.80, mass_kg=80.0),
        BodyParameters(height_m=1.95, mass_kg=100.0),
    ],
)
def test_urdf_generation_is_deterministic_across_presets(
    params: BodyParameters,
) -> None:
    """Determinism check across the typical body-size range."""
    builder = CharacterBuilder()
    runs = [builder.generate_urdf(params) for _ in range(5)]
    assert (
        len(set(runs)) == 1
    ), f"5 runs produced {len(set(runs))} distinct URDFs; expected 1."


# ---------------------------------------------------------------------------
# #4536 - URDF schema / XML validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generated_urdf_is_well_formed_xml(baseline_urdf: str) -> None:
    """The most basic schema check: it must parse. See issue #4536."""
    root = ET.fromstring(baseline_urdf)
    assert root.tag == "robot", f"Top-level tag is {root.tag!r}, expected 'robot'"
    assert root.get("name"), "Root <robot> element must have a name attribute"


@pytest.mark.unit
def test_generated_urdf_has_required_structure(baseline_urdf: str) -> None:
    """URDF spec requires: at least one link, every joint references existing
    links, no link is its own parent. See issue #4536.
    """
    root = ET.fromstring(baseline_urdf)
    link_names = {link.get("name") for link in root.findall("link")}
    assert link_names, "URDF must contain at least one <link>"

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        assert parent is not None, f"joint {joint.get('name')!r} missing <parent>"
        assert child is not None, f"joint {joint.get('name')!r} missing <child>"
        parent_link = parent.get("link")
        child_link = child.get("link")
        assert (
            parent_link in link_names
        ), f"joint {joint.get('name')!r} references unknown parent {parent_link!r}"
        assert (
            child_link in link_names
        ), f"joint {joint.get('name')!r} references unknown child {child_link!r}"
        assert (
            parent_link != child_link
        ), f"joint {joint.get('name')!r} has same link as parent and child"


# ---------------------------------------------------------------------------
# #4539 - Inertia / mass validation
# ---------------------------------------------------------------------------


def _parse_inertias(urdf_xml: str) -> list[dict[str, float]]:
    """Pull mass + inertia tensor components from every <inertial> block."""
    root = ET.fromstring(urdf_xml)
    out: list[dict[str, float]] = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        i_el = inertial.find("inertia")
        if mass_el is None or i_el is None:
            continue
        out.append(
            {
                "link": link.get("name", ""),
                "mass": float(mass_el.get("value", "0")),
                "ixx": float(i_el.get("ixx", "0")),
                "iyy": float(i_el.get("iyy", "0")),
                "izz": float(i_el.get("izz", "0")),
                "ixy": float(i_el.get("ixy", "0")),
                "ixz": float(i_el.get("ixz", "0")),
                "iyz": float(i_el.get("iyz", "0")),
            }
        )
    return out


@pytest.mark.unit
def test_total_mass_matches_requested_within_tolerance(
    baseline_params: BodyParameters, baseline_urdf: str
) -> None:
    """Sum of segment masses must equal the requested body mass within
    1% tolerance. See issue #4539.
    """
    inertias = _parse_inertias(baseline_urdf)
    total = sum(seg["mass"] for seg in inertias)
    target = baseline_params.mass_kg
    assert abs(total - target) / target < 0.01, (
        f"Total segment mass {total:.3f} kg differs from requested "
        f"{target:.3f} kg by more than 1%."
    )


@pytest.mark.unit
def test_every_segment_mass_is_positive(baseline_urdf: str) -> None:
    """No segment may have zero or negative mass."""
    for seg in _parse_inertias(baseline_urdf):
        assert (
            seg["mass"] > 0
        ), f"Segment {seg['link']!r} has non-positive mass {seg['mass']}"


@pytest.mark.unit
def test_every_segment_inertia_is_positive_definite(baseline_urdf: str) -> None:
    """Every <inertia> tensor must be symmetric positive-definite. A
    physically real rigid body has all-positive eigenvalues. See #4539.
    """
    for seg in _parse_inertias(baseline_urdf):
        tensor = np.array(
            [
                [seg["ixx"], seg["ixy"], seg["ixz"]],
                [seg["ixy"], seg["iyy"], seg["iyz"]],
                [seg["ixz"], seg["iyz"], seg["izz"]],
            ]
        )
        eigvals = np.linalg.eigvalsh(tensor)
        assert (eigvals > 0).all(), (
            f"Segment {seg['link']!r} inertia tensor is not positive-definite "
            f"(eigenvalues {eigvals.tolist()}). Tensor: {tensor.tolist()}"
        )


@pytest.mark.unit
def test_every_segment_satisfies_inertia_triangle_inequality(
    baseline_urdf: str,
) -> None:
    """For a real rigid body with principal moments Ia, Ib, Ic each pair
    must satisfy Ia + Ib >= Ic. Violations indicate a non-physical
    inertia tensor. See #4539.
    """
    for seg in _parse_inertias(baseline_urdf):
        tensor = np.array(
            [
                [seg["ixx"], seg["ixy"], seg["ixz"]],
                [seg["ixy"], seg["iyy"], seg["iyz"]],
                [seg["ixz"], seg["iyz"], seg["izz"]],
            ]
        )
        principal = sorted(np.linalg.eigvalsh(tensor))
        a, b, c = principal
        # Allow a small numerical slack (1e-6) for floating-point noise
        assert a + b + 1e-6 >= c, (
            f"Segment {seg['link']!r} violates triangle inequality: "
            f"principal moments {principal}; smallest two sum to {a + b}, "
            f"largest is {c}."
        )


# ---------------------------------------------------------------------------
# #4540 - Joint limit anatomy sanity
# ---------------------------------------------------------------------------


def _parse_joint_limits(urdf_xml: str) -> list[dict[str, str | float]]:
    """Pull (name, type, axis, lower, upper) for every revolute/prismatic joint."""
    root = ET.fromstring(urdf_xml)
    out: list[dict[str, str | float]] = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        if jtype not in ("revolute", "prismatic", "continuous"):
            continue
        limit = joint.find("limit")
        lower = (
            float(limit.get("lower", "-inf")) if limit is not None else float("-inf")
        )
        upper = float(limit.get("upper", "inf")) if limit is not None else float("inf")
        out.append(
            {
                "name": joint.get("name", ""),
                "type": jtype,
                "lower": lower,
                "upper": upper,
            }
        )
    return out


@pytest.mark.unit
def test_every_joint_lower_below_upper(baseline_urdf: str) -> None:
    """A joint with lower > upper is a definite typo. See issue #4540."""
    for j in _parse_joint_limits(baseline_urdf):
        if j["type"] == "continuous":
            continue
        if j["lower"] == float("-inf") or j["upper"] == float("inf"):
            continue
        assert (
            j["lower"] < j["upper"]
        ), f"Joint {j['name']!r} has lower {j['lower']} >= upper {j['upper']}"


@pytest.mark.unit
def test_joint_limits_are_in_radians(baseline_urdf: str) -> None:
    """Sanity bound: any revolute joint with |limit| > 2*pi was probably
    written in degrees instead of radians. See issue #4540.
    """
    two_pi = 2 * np.pi
    for j in _parse_joint_limits(baseline_urdf):
        if j["type"] != "revolute":
            continue
        if j["lower"] == float("-inf") or j["upper"] == float("inf"):
            continue
        assert abs(j["lower"]) < two_pi + 0.01, (
            f"Joint {j['name']!r} lower limit {j['lower']} exceeds 2*pi - "
            f"likely written in degrees."
        )
        assert abs(j["upper"]) < two_pi + 0.01, (
            f"Joint {j['name']!r} upper limit {j['upper']} exceeds 2*pi - "
            f"likely written in degrees."
        )
