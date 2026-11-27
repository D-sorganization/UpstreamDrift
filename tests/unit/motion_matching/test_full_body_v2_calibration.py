"""A v2 spec carries the run's calibration from its v1 source (#15670)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from src.shared.python.motion_matching.full_body_migration import canonical_sha256
from src.shared.python.motion_matching.full_body_v2_calibration import (
    carry_calibration,
)

pytestmark = pytest.mark.unit


def _base() -> dict[str, Any]:
    return {
        "schema_version": "full-body-v1",
        "coordinate_order": ["hip_flexion_r", "lumbar_flexion"],
        "joints": [
            {"name": "hip_r", "parent_to_base": [[1.0, 0.0], [0.0, 1.0]]},
            {"name": "lumbar", "parent_to_base": [[1.0, 0.0], [0.0, 1.0]]},
        ],
        "bodies": [{"name": "tibia_r", "com_m": 0.20}, {"name": "torso", "m": 30.0}],
        "contact": {"ground": {"calibrated": False}},
        "subject": {"mass_kg": 78.0},
        "provenance": "base",
    }


def _calibrated(base: dict[str, Any]) -> dict[str, Any]:
    run = copy.deepcopy(base)
    run["joints"][0]["parent_to_base"] = [[0.9, 0.1], [-0.1, 0.9]]
    run["bodies"][0]["com_m"] = 0.19
    run["contact"] = {"ground": {"calibrated": True, "height_m": 2.6e-4}}
    run["subject"]["hip_zero_twist_deg"] = {"r": -5.7}
    run["provenance"] = "hip-calibrated, scaled"
    return run


def _v2(base: dict[str, Any]) -> dict[str, Any]:
    v2 = copy.deepcopy(base)
    v2["schema_version"] = "full-body-v2"
    v2["joints"][1] = {"name": "lumbar", "parent_to_base": [[0.0, 1.0], [1.0, 0.0]]}
    v2["bodies"][1] = {"name": "abdomen", "m": 12.0}
    v2["provenance"] = "v2 trunk"
    v2["source_spec_sha256"] = canonical_sha256(base)
    return v2


def test_calibrated_parts_replace_the_base_parts_in_v2() -> None:
    base = _base()
    run = _calibrated(base)
    out = carry_calibration(_v2(base), base, run)
    assert out["joints"][0] == run["joints"][0]
    assert out["bodies"][0] == run["bodies"][0]
    assert out["contact"] == run["contact"]
    assert out["subject"] == run["subject"]


def test_v2_trunk_and_provenance_are_kept() -> None:
    base = _base()
    v2 = _v2(base)
    out = carry_calibration(v2, base, _calibrated(base))
    assert out["joints"][1] == v2["joints"][1]
    assert out["bodies"][1] == v2["bodies"][1]
    assert out["provenance"] == "v2 trunk"
    assert out["schema_version"] == "full-body-v2"


def test_the_result_names_the_calibrated_spec_as_its_source() -> None:
    base = _base()
    run = _calibrated(base)
    out = carry_calibration(_v2(base), base, run)
    assert out["source_spec_sha256"] == canonical_sha256(run)


def test_inputs_are_not_mutated() -> None:
    base = _base()
    run = _calibrated(base)
    v2 = _v2(base)
    before = copy.deepcopy((v2, base, run))
    carry_calibration(v2, base, run)
    assert (v2, base, run) == before


def test_rejects_a_v2_spec_not_derived_from_the_base() -> None:
    base = _base()
    v2 = _v2(base)
    v2["source_spec_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_spec_sha256"):
        carry_calibration(v2, base, _calibrated(base))


def test_rejects_calibration_of_a_part_the_v2_migration_changed() -> None:
    base = _base()
    run = _calibrated(base)
    run["joints"][1]["parent_to_base"] = [[0.5, 0.5], [0.5, 0.5]]
    with pytest.raises(ValueError, match="lumbar"):
        carry_calibration(_v2(base), base, run)


def test_rejects_a_calibration_that_adds_or_drops_parts() -> None:
    base = _base()
    run = _calibrated(base)
    run["bodies"].append({"name": "extra"})
    with pytest.raises(ValueError, match="same joints and bodies"):
        carry_calibration(_v2(base), base, run)


def test_rejects_a_calibration_with_another_coordinate_order() -> None:
    base = _base()
    run = _calibrated(base)
    run["coordinate_order"] = ["lumbar_flexion", "hip_flexion_r"]
    with pytest.raises(ValueError, match="coordinate_order"):
        carry_calibration(_v2(base), base, run)
