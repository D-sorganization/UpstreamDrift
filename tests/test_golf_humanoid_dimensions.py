"""Tests for the shared anthropometric YAMLs (issue #4093).

Validates `shared/models/golf_humanoid_dimensions.yaml`,
`shared/models/golf_humanoid_inertia.yaml`, and
`shared/models/golf_humanoid_topology.yaml` — the cross-engine body-model
source-of-truth referenced by `src/engines/CROSS_ENGINE_PARITY_SPEC.md` §2.6.

These tests do NOT require MATLAB or Simscape to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "shared" / "models"

DIMENSIONS_PATH = MODELS_DIR / "golf_humanoid_dimensions.yaml"
INERTIA_PATH = MODELS_DIR / "golf_humanoid_inertia.yaml"
TOPOLOGY_PATH = MODELS_DIR / "golf_humanoid_topology.yaml"

REQUIRED_DIMENSION_KEYS = (
    "UpperTorsoLength",
    "LowerTorsoLength",
    "HubtoSLength",
    "LeftShoulderWidth",
    "RightShoulderWidth",
    "LeftUpperArmLength",
    "RightUpperArmLength",
    "LowerArmLength",
    "LeftWristStandoffLength",
    "RightWristStandoffLength",
    "ShaftLength",
    "NeckLength",
)

REQUIRED_INERTIA_SEGMENTS = (
    "UpperTorso",
    "LowerTorso",
    "LeftShoulderToHub",
    "RightShoulderToHub",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftForearmUpper",
    "LeftForearmLower",
    "RightForearmUpper",
    "RightForearmLower",
    "LeftHand",
    "RightHand",
    "Club",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dimensions() -> dict:
    with DIMENSIONS_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def inertia() -> dict:
    with INERTIA_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def topology() -> dict:
    with TOPOLOGY_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# File-existence + parse checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_yaml_files_exist() -> None:
    for path in (DIMENSIONS_PATH, INERTIA_PATH, TOPOLOGY_PATH):
        assert path.is_file(), f"missing required YAML: {path}"


@pytest.mark.unit
def test_yamls_parse_cleanly(dimensions: dict, inertia: dict, topology: dict) -> None:
    assert isinstance(dimensions, dict)
    assert isinstance(inertia, dict)
    assert isinstance(topology, dict)


@pytest.mark.unit
@pytest.mark.parametrize("doc_name", ["dimensions", "inertia", "topology"])
def test_schema_version_present(
    doc_name: str, dimensions: dict, inertia: dict, topology: dict
) -> None:
    docs = {"dimensions": dimensions, "inertia": inertia, "topology": topology}
    assert docs[doc_name].get("schema_version") == 1


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dimensions_units_system_is_si(dimensions: dict) -> None:
    assert dimensions.get("units_system") == "SI"


@pytest.mark.unit
@pytest.mark.parametrize("key", REQUIRED_DIMENSION_KEYS)
def test_required_dimension_keys_present(dimensions: dict, key: str) -> None:
    assert key in dimensions, f"missing required dimension entry: {key}"
    entry = dimensions[key]
    assert "value" in entry
    assert "units" in entry
    assert "source" in entry


@pytest.mark.unit
@pytest.mark.parametrize("key", REQUIRED_DIMENSION_KEYS)
def test_dimension_values_are_positive_metres(dimensions: dict, key: str) -> None:
    entry = dimensions[key]
    assert entry["units"] == "m", f"{key} must be in metres, got {entry['units']}"
    assert isinstance(entry["value"], (int, float))
    assert entry["value"] > 0, f"{key} must be > 0, got {entry['value']}"
    # Sanity: no segment longer than 2 metres.
    assert entry["value"] < 2.0, f"{key} unreasonably long: {entry['value']} m"


@pytest.mark.unit
def test_dimension_inch_conversions_round_trip(dimensions: dict) -> None:
    """For entries with raw_units == 'in', value ≈ raw_value × 0.0254."""
    inches_to_m = 0.0254
    for key, entry in dimensions.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("raw_units") == "in":
            expected = entry["raw_value"] * inches_to_m
            assert abs(entry["value"] - expected) < 1e-9, (
                f"{key}: stored value {entry['value']} does not match "
                f"raw_value {entry['raw_value']} in × 0.0254 = {expected}"
            )


@pytest.mark.unit
def test_dimensions_have_no_nan_or_inf(dimensions: dict) -> None:
    import math

    for key, entry in dimensions.items():
        if isinstance(entry, dict) and "value" in entry:
            v = entry["value"]
            assert math.isfinite(v), f"{key} value is not finite: {v}"


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inertia_units_system_is_si(inertia: dict) -> None:
    assert inertia.get("units_system") == "SI"


@pytest.mark.unit
def test_inertia_total_golfer_mass_is_realistic(inertia: dict) -> None:
    total = inertia["golfer"]["total_mass_kg"]
    assert 40.0 < total < 150.0, f"unrealistic golfer mass: {total} kg"


@pytest.mark.unit
@pytest.mark.parametrize("segment", REQUIRED_INERTIA_SEGMENTS)
def test_required_inertia_segments_present(inertia: dict, segment: str) -> None:
    assert segment in inertia, f"missing inertia entry: {segment}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "segment",
    [s for s in REQUIRED_INERTIA_SEGMENTS if s != "Club"],
)
def test_inertia_segment_has_required_fields(inertia: dict, segment: str) -> None:
    entry = inertia[segment]
    assert "mass_kg" in entry
    assert "com_offset_m" in entry
    assert "inertia_kgm2" in entry
    assert "source" in entry

    # mass_kg
    m = entry["mass_kg"]
    assert isinstance(m, (int, float))
    assert m > 0, f"{segment} mass must be > 0, got {m}"
    assert m < 50.0, f"{segment} mass unreasonable: {m} kg"

    # com_offset_m
    com = entry["com_offset_m"]
    assert isinstance(com, list) and len(com) == 3
    assert all(isinstance(c, (int, float)) for c in com)

    # inertia tensor
    inertia_tensor = entry["inertia_kgm2"]
    assert isinstance(inertia_tensor, list) and len(inertia_tensor) == 3
    assert all(len(row) == 3 for row in inertia_tensor)
    # Diagonal should be non-negative.
    for i in range(3):
        assert (
            inertia_tensor[i][i] >= 0.0
        ), f"{segment} I[{i}][{i}] must be >= 0, got {inertia_tensor[i][i]}"


@pytest.mark.unit
def test_club_entry_well_formed(inertia: dict) -> None:
    club = inertia["Club"]
    assert "mass_kg" in club
    assert club["mass_kg"] > 0.1
    assert "shaft" in club and "clubhead" in club
    assert club["shaft"]["length_m"] > 0.5
    assert club["clubhead"]["mass_kg"] > 0.0


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_topology_total_dof_matches_q_order(topology: dict) -> None:
    declared = topology["total_dof"]
    assert isinstance(declared, int) and declared > 0
    assert len(topology["q_order"]) == declared


@pytest.mark.unit
def test_topology_joint_dof_sum_matches_total(topology: dict) -> None:
    joint_dof_sum = sum(j["dof"] for j in topology["joints"])
    # 'welded' joints contribute 0 DOF and are still expected to be listed.
    assert joint_dof_sum == topology["total_dof"]


@pytest.mark.unit
def test_topology_bodies_and_joints_are_consistent(topology: dict) -> None:
    body_names = {b["name"] for b in topology["bodies"]}
    for joint in topology["joints"]:
        assert (
            joint["parent"] in body_names
        ), f"joint {joint['name']} references unknown parent {joint['parent']}"
        assert (
            joint["child"] in body_names
        ), f"joint {joint['name']} references unknown child {joint['child']}"


@pytest.mark.unit
def test_topology_joint_types_are_valid(topology: dict) -> None:
    valid_types = {"floating", "universal", "revolute", "gimbal", "welded"}
    for joint in topology["joints"]:
        assert (
            joint["type"] in valid_types
        ), f"unknown joint type {joint['type']!r} on {joint['name']}"


@pytest.mark.unit
def test_topology_q_order_references_known_joints(topology: dict) -> None:
    joint_names = {j["name"] for j in topology["joints"]}
    for entry in topology["q_order"]:
        assert entry["joint"] in joint_names
        assert entry["dof"] in {"tx", "ty", "tz", "rx", "ry", "rz"}


# ---------------------------------------------------------------------------
# Cross-document consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dimensions_and_inertia_agree_on_arm_lengths(
    dimensions: dict, inertia: dict
) -> None:
    """COM offset of upper arm should be ~half the upper arm length (in magnitude)."""
    upper_arm_length_m = dimensions["LeftUpperArmLength"]["value"]
    com_x = inertia["LeftUpperArm"]["com_offset_m"][0]
    half_length = upper_arm_length_m / 2.0
    # Allow ±10 % slack — the Simscape COM is exactly halfway for cylinders.
    assert (
        abs(abs(com_x) - half_length) <= 0.10 * half_length
    ), f"LeftUpperArm COM_x={com_x} does not match half-length {half_length}"


@pytest.mark.unit
def test_topology_segment_inertia_refs_resolve(inertia: dict, topology: dict) -> None:
    """Every inertia_ref on a body must resolve to a key in the inertia YAML."""
    inertia_keys = set(inertia.keys())
    for body in topology["bodies"]:
        ref = body.get("inertia_ref")
        if ref is None:
            continue
        assert (
            ref in inertia_keys
        ), f"body {body['name']} references unknown inertia segment {ref}"
