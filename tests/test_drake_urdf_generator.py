"""Tests for the Drake humanoid URDF generator (issue #4108, DRAKE-1).

Two tiers:

* Pure-Python tests that exercise the YAML loader, render-to-string, and
  joint-name plumbing. These run anywhere (no pydrake required) and are
  marked ``unit``.
* Drake-integration tests that load the generated URDF into a real
  ``MultibodyPlant`` and assert ``num_velocities() == EXPECTED_NUM_VELOCITIES``.
  These are gated on ``pydrake`` being importable; if not, the test is
  skipped (per CLAUDE.md, never silently no-op'd).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path

import pytest
from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    CANONICAL_URDF,
    EXPECTED_NUM_REVOLUTE_DOF,
    EXPECTED_NUM_VELOCITIES,
    SHARED_DIMENSIONS_YAML,
    build_humanoid_urdf,
    load_humanoid_dimensions,
    render_urdf_string,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_joint_names_from_yaml() -> set[str]:
    """Compute the URDF joint names that should appear after composition.

    Universal joints split into ``<name>_1`` / ``<name>_2``; gimbals into
    ``<name>_1`` / ``<name>_2`` / ``<name>_3``. Revolute / fixed / floating
    joints keep their names.
    """
    dims = load_humanoid_dimensions()
    expected: set[str] = set()
    for seg in dims.segments:
        j = seg.joint
        if j.type == "universal":
            expected |= {f"{j.name}_1", f"{j.name}_2"}
        elif j.type == "gimbal":
            expected |= {f"{j.name}_1", f"{j.name}_2", f"{j.name}_3"}
        else:
            expected.add(j.name)
    return expected


# ---------------------------------------------------------------------------
# Pure-Python tests (no pydrake)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_yaml_exists_and_parses() -> None:
    """The shared YAML is on disk and parses without error."""
    assert SHARED_DIMENSIONS_YAML.exists(), (
        f"Expected {SHARED_DIMENSIONS_YAML}; #4093 owns this file."
    )
    dims = load_humanoid_dimensions()
    assert dims.schema_version >= 1
    assert len(dims.segments) >= 12, f"Expected >=12 segments, got {len(dims.segments)}"
    assert dims.pelvis_to_shoulders_m > 0
    assert dims.hand_spacing_m > 0


@pytest.mark.unit
def test_render_urdf_string_well_formed_xml() -> None:
    """The rendered URDF parses as XML."""
    dims = load_humanoid_dimensions()
    urdf = render_urdf_string(dims)
    root = ET.fromstring(urdf)
    assert root.tag == "robot"
    assert root.get("name") == "golf_humanoid"


@pytest.mark.unit
def test_revolute_joint_count_matches_expected_dof() -> None:
    """Counting revolute joints in the rendered URDF matches the spec."""
    dims = load_humanoid_dimensions()
    urdf = render_urdf_string(dims)
    root = ET.fromstring(urdf)
    revolute = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    floating = [j for j in root.findall("joint") if j.get("type") == "floating"]
    assert len(floating) == 1, "Exactly one floating root joint expected"
    assert len(revolute) == EXPECTED_NUM_REVOLUTE_DOF, (
        f"Expected {EXPECTED_NUM_REVOLUTE_DOF} revolute joints, got "
        f"{len(revolute)}: "
        f"{sorted(j.get('name') for j in revolute)}"
    )


@pytest.mark.unit
def test_every_topology_joint_present() -> None:
    """Every joint name in the YAML topology appears in the URDF."""
    dims = load_humanoid_dimensions()
    urdf = render_urdf_string(dims)
    root = ET.fromstring(urdf)
    urdf_joint_names = {j.get("name") for j in root.findall("joint")}
    expected = _expected_joint_names_from_yaml()
    missing = expected - urdf_joint_names
    assert not missing, f"Missing joints from URDF: {sorted(missing)}"


@pytest.mark.unit
def test_universal_joints_use_two_axes() -> None:
    """Universal-joint composition emits exactly two revolute sub-joints."""
    dims = load_humanoid_dimensions()
    urdf = render_urdf_string(dims)
    root = ET.fromstring(urdf)
    joint_names = {j.get("name") for j in root.findall("joint")}
    for seg in dims.segments:
        if seg.joint.type == "universal":
            for idx in (1, 2):
                assert f"{seg.joint.name}_{idx}" in joint_names
            assert f"{seg.joint.name}_3" not in joint_names


@pytest.mark.unit
def test_gimbal_joints_use_three_axes() -> None:
    """Gimbal-joint composition emits exactly three revolute sub-joints."""
    dims = load_humanoid_dimensions()
    urdf = render_urdf_string(dims)
    root = ET.fromstring(urdf)
    joint_names = {j.get("name") for j in root.findall("joint")}
    for seg in dims.segments:
        if seg.joint.type == "gimbal":
            for idx in (1, 2, 3):
                assert f"{seg.joint.name}_{idx}" in joint_names
            assert f"{seg.joint.name}_4" not in joint_names


@pytest.mark.unit
def test_club_welded_to_right_hand() -> None:
    """The club is welded to right_hand only (URDF closed-loop compromise)."""
    dims = load_humanoid_dimensions()
    club = next(s for s in dims.segments if s.name == "club_shaft")
    assert club.parent == "right_hand"
    assert club.joint.type == "fixed"


@pytest.mark.unit
def test_build_humanoid_urdf_writes_file(tmp_path: Path) -> None:
    """``build_humanoid_urdf`` writes a non-empty file at the requested path."""
    out = tmp_path / "golfer.urdf"
    written = build_humanoid_urdf(out_path=out)
    assert written == out.resolve()
    assert out.exists()
    assert out.stat().st_size > 0
    # Sanity: contents include the floating root joint.
    text = out.read_text(encoding="utf-8")
    assert "pelvis_floating" in text


@pytest.mark.unit
def test_canonical_urdf_path_is_under_drake_engine() -> None:
    """Pin the canonical on-disk URDF path so CI gate (#4129) is stable."""
    parts = CANONICAL_URDF.parts
    assert "drake" in parts
    assert parts[-1] == "golfer.urdf"
    assert "generated" in parts


# ---------------------------------------------------------------------------
# Drake integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drake_modules():
    """Import pydrake or skip the drake-integration tests."""
    try:
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
    except ImportError as exc:
        pytest.skip(f"pydrake not installed: {exc}")
    return {"Parser": Parser, "MultibodyPlant": MultibodyPlant}


@pytest.fixture(scope="module")
def generated_urdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("drake_urdf") / "golfer.urdf"
    return build_humanoid_urdf(out_path=out)


@pytest.mark.integration
@pytest.mark.requires_drake
def test_urdf_parses_cleanly_in_drake(drake_modules, generated_urdf: Path) -> None:
    """pydrake's Parser ingests the generated URDF without raising."""
    plant = drake_modules["MultibodyPlant"](time_step=1e-3)
    parser = drake_modules["Parser"](plant)
    models = parser.AddModels(str(generated_urdf))
    plant.Finalize()
    assert len(models) == 1


@pytest.mark.integration
@pytest.mark.requires_drake
def test_drake_num_velocities_matches_spec(drake_modules, generated_urdf: Path) -> None:
    """After Finalize, ``num_velocities()`` matches the expected DOF total."""
    plant = drake_modules["MultibodyPlant"](time_step=1e-3)
    parser = drake_modules["Parser"](plant)
    parser.AddModels(str(generated_urdf))
    plant.Finalize()
    assert plant.num_velocities() == EXPECTED_NUM_VELOCITIES


@pytest.mark.integration
@pytest.mark.requires_drake
def test_drake_topology_joint_names_present(
    drake_modules, generated_urdf: Path
) -> None:
    """Every joint name from the YAML topology is registered in the plant."""
    from pydrake.multibody.tree import JointIndex

    plant = drake_modules["MultibodyPlant"](time_step=1e-3)
    parser = drake_modules["Parser"](plant)
    parser.AddModels(str(generated_urdf))
    plant.Finalize()

    plant_joint_names: set[str] = set()
    for i in range(plant.num_joints()):
        plant_joint_names.add(plant.get_joint(JointIndex(i)).name())

    expected = _expected_joint_names_from_yaml()
    # Drake adds an internal floating-base joint name per the model;
    # tolerate that by checking subset of expected revolute names.
    missing = {n for n in expected if "_floating" not in n} - plant_joint_names
    assert not missing, f"Missing joints in plant: {sorted(missing)}"
