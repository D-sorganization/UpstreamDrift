"""Tests for the cited spinal coupling-law data and loader (SPN-3, #12743).

Every literature number in ``src/config/spine_coupling_laws.json`` must trace
to a JSON block of ``docs/research/segmental_spine/data_extraction.md``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.spine.coupling import regional_fraction
from src.shared.python.spine.coupling_data import (
    available_laws,
    coupling_law,
    load_coupling_laws,
)
from src.shared.python.spine.levels import SpinalLevel, SpinalRegion

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
CONFIG = REPO / "src" / "config" / "spine_coupling_laws.json"
EXTRACTION = REPO / "docs" / "research" / "segmental_spine" / "data_extraction.md"
LUMBAR = [lv for lv in SpinalLevel if lv.region is SpinalRegion.LUMBAR]
THORACIC = [lv for lv in SpinalLevel if lv.region is SpinalRegion.THORACIC]
SHIPPED = {"bruno2015_regional", "fujii2007_lumbar", "uniform_reference"}


def _extraction_blocks() -> dict[tuple[str, str], dict]:
    text = EXTRACTION.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    parsed = [json.loads(b) for b in blocks]
    return {(b["source_id"], b["table_id"]): b for b in parsed}


def _numbers(node: object) -> list[float]:
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        return [n for v in node.values() for n in _numbers(v)]
    if isinstance(node, list):
        return [n for v in node for n in _numbers(v)]
    return []


def _raw() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _raw_law(law_id: str) -> dict:
    return next(e for e in _raw()["laws"] if e["law_id"] == law_id)


def test_extraction_blocks_parse() -> None:
    assert ("bruno2015", "B1_regional_split") in _extraction_blocks()


def test_shipped_and_pending_sets() -> None:
    assert set(available_laws()) == SHIPPED
    pending = set(available_laws(include_pending=True)) - SHIPPED
    assert pending == {"bruno2015_per_level", "pearcy1984_lumbar"}


def test_available_laws_sorted_tuple() -> None:
    names = available_laws()
    assert isinstance(names, tuple) and list(names) == sorted(names)


def test_metadata_has_citations_and_flags() -> None:
    entries = load_coupling_laws()
    assert set(entries) == SHIPPED | {"bruno2015_per_level", "pearcy1984_lumbar"}
    for law_id, entry in entries.items():
        assert entry["law_id"] == law_id
        assert entry["citation"]["source"].strip()
        assert entry["citation"]["table"].strip()
        assert entry["notes"].strip()
        assert isinstance(entry["verified_from_primary"], bool)
        assert entry["version"].strip()


def test_uniform_is_labelled_not_literature() -> None:
    entry = load_coupling_laws()["uniform_reference"]
    assert entry["citation"]["source"] == "repo-authored reference, not literature"
    assert entry["verified_from_primary"] is False


def test_bruno_regional_splits_to_1e12() -> None:
    law = coupling_law("bruno2015_regional")
    expected = {0: (0.51, 0.49), 1: (0.27, 0.73), 2: (0.17, 0.83)}
    for axis, (lum, tho) in expected.items():
        phi = np.zeros(3)
        phi[axis] = 1.0
        assert abs(regional_fraction(law, LUMBAR, axis, phi) - lum) < 1e-12
        assert abs(regional_fraction(law, THORACIC, axis, phi) - tho) < 1e-12


def test_bruno_regional_independent_and_levels() -> None:
    law = coupling_law("bruno2015_regional")
    assert law.independent_names() == (
        "flexion_extension",
        "lateral_bending",
        "axial_rotation",
    )
    assert law.levels == tuple(SpinalLevel)


def test_bruno_within_region_uniform_assumption_recorded() -> None:
    entry = _raw_law("bruno2015_regional")
    assert "uniform" in entry["assumptions"].lower()
    law = coupling_law("bruno2015_regional")
    gains = {lv: law.matrices[lv][2, 2] for lv in LUMBAR}
    assert len(set(round(g, 15) for g in gains.values())) == 1


@pytest.mark.parametrize("law_id", ["bruno2015_regional", "uniform_reference"])
def test_share_laws_row_sum_to_one(law_id: str) -> None:
    law = coupling_law(law_id)
    total = sum(law.matrices[lv] for lv in law.levels)
    np.testing.assert_allclose(total, np.eye(3), atol=1e-12)


def test_uniform_reference_equal_shares() -> None:
    law = coupling_law("uniform_reference")
    for lv in law.levels:
        np.testing.assert_allclose(law.matrices[lv], np.eye(3) / 17, atol=1e-15)


def test_fujii_matches_extraction_rows() -> None:
    entry = _raw_law("fujii2007_lumbar")
    ref = entry["source_ref"]
    block = _extraction_blocks()[(ref["source_id"], ref["table_id"])]
    assert block["verified_from_primary"] is True
    by_segment = {row["segment"]: row for row in block["rows"]}
    law = coupling_law("fujii2007_lumbar")
    rot = entry["distribution"]["reference_trunk_rotation_deg"]
    assert rot == 45.0
    assert len(entry["distribution"]["rows"]) == len(by_segment) == len(law.levels)
    for row in entry["distribution"]["rows"]:
        src = by_segment[row["segment"]]
        assert row["axial_rotation"] == src["axial_rotation"]
        assert row["coupled_flexion"] == src["coupled_flexion"]
        assert row["coupled_lateral_bend"] == src["coupled_lateral_bend"]
        level = SpinalLevel[row["level"]]
        expected = np.array(
            [
                [math.radians(src["coupled_flexion"][0])],
                [math.radians(src["coupled_lateral_bend"][0])],
                [math.radians(src["axial_rotation"][0])],
            ]
        ) / math.radians(rot)
        np.testing.assert_allclose(law.matrices[level], expected, atol=1e-15)


def test_fujii_levels_and_independent() -> None:
    law = coupling_law("fujii2007_lumbar")
    assert law.levels == (
        SpinalLevel.L5_S1,
        SpinalLevel.L4_L5,
        SpinalLevel.L3_L4,
        SpinalLevel.L2_L3,
        SpinalLevel.L1_L2,
        SpinalLevel.T12_L1,
    )
    assert law.independent_names() == ("trunk_axial_rotation",)
    out = law.expand(np.array([math.radians(45.0)]))
    assert out[SpinalLevel.L3_L4][1] == pytest.approx(math.radians(-3.4))


def test_every_literature_number_in_extraction() -> None:
    blocks = _extraction_blocks()
    for entry in _raw()["laws"]:
        ref = entry.get("source_ref")
        if ref is None:
            continue
        pool = set(_numbers(blocks[(ref["source_id"], ref["table_id"])]))
        # reference rotation 45 deg is the heading of Table F1
        pool.add(45.0)
        for number in _numbers(entry["distribution"]):
            assert number in pool, (entry["law_id"], number)


def test_regional_percent_equals_extraction_rows() -> None:
    entry = _raw_law("bruno2015_regional")
    ref = entry["source_ref"]
    block = _extraction_blocks()[(ref["source_id"], ref["table_id"])]
    rows = {r["axis"]: r for r in block["rows"] if "lumbar_pct" in r}
    for axis, split in entry["distribution"]["regional_percent"].items():
        assert split["lumbar"] == rows[axis]["lumbar_pct"]
        assert split["thoracic"] == rows[axis]["thoracic_pct"]
    assert block["verified_from_primary"] is True


def test_shipped_laws_reference_verified_or_reference_sources() -> None:
    for entry in _raw()["laws"]:
        if entry["status"] == "available" and entry["law_id"] != "uniform_reference":
            assert entry["verified_from_primary"] is True


@pytest.mark.parametrize("law_id", ["bruno2015_per_level", "pearcy1984_lumbar"])
def test_pending_laws_refuse_and_have_no_matrices(law_id: str) -> None:
    entry = _raw_law(law_id)
    assert entry["status"] == "pending"
    assert entry["pending_reason"].strip()
    assert "matrices" not in entry and "distribution" not in entry
    assert entry["verified_from_primary"] is False
    with pytest.raises(ValueError, match="pending"):
        coupling_law(law_id)


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        coupling_law("nope")


def test_bad_types_raise() -> None:
    with pytest.raises(TypeError):
        coupling_law(3)  # type: ignore[arg-type]


def test_custom_path_and_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"laws": [{"law_id": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_coupling_laws(bad)
    good = load_coupling_laws(CONFIG)
    assert set(good) == set(load_coupling_laws())


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["laws"].append(raw["laws"][0])
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_coupling_laws(path)


def test_returned_law_is_independent_copy() -> None:
    first = coupling_law("uniform_reference")
    second = coupling_law("uniform_reference")
    assert first is not second
    assert not first.matrices[SpinalLevel.L5_S1].flags.writeable
