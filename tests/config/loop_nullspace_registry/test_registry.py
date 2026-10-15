"""Tests for NSX-11 hypothesis, convention, and preset registry loader and citation gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.loop_nullspace.registry import (
    ConventionRecord,
    HypothesisRecord,
    HypothesisSet,
    LoopNullspaceRegistry,
    PresetRecord,
    SourceRecord,
    convention,
    hypothesis,
    load_registry,
    preset,
    require_verified_for_display,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "src" / "config" / "loop_nullspace_registry.json"


def _raw_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_load_registry_schema_and_unique_ids() -> None:
    """Loaded registry returns typed frozen records with unique IDs."""
    reg = load_registry()
    assert isinstance(reg, LoopNullspaceRegistry)
    assert reg.schema_version == "1.0.0"

    hyp_ids = [h.id for h in reg.hypotheses]
    assert len(hyp_ids) == len(set(hyp_ids))
    assert {
        "H-TORQUE-CAPACITY",
        "H-TORQUE-CAPACITY-BOX",
        "H-MIN-EFFORT",
        "H-EFFORT-CAPACITY-QUADRATIC",
        "H-SMOOTH-TORQUE-RATE",
        "H-TORQUE-RATE-SMOOTHNESS",
        "H-FRICTION-CONE",
        "H-GRIP-FRICTION-CONE",
        "H-GRIP-PRELOAD",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    }.issubset(set(hyp_ids))

    conv_ids = [c.id for c in reg.conventions]
    assert len(conv_ids) == len(set(conv_ids))
    assert {
        "C-FRAME-GRIP",
        "C-METRIC-CAPACITY",
        "C-METRIC-IDENTITY",
        "C-METRIC-MINV",
        "C-CLOSURE-CANONICAL-R8",
        "C-CONE-POLYHEDRAL-8",
    }.issubset(set(conv_ids))

    preset_names = [p.name for p in reg.presets]
    assert len(preset_names) == len(set(preset_names))
    assert {"capacity_min_effort", "envelope_only", "smooth_capacity"}.issubset(
        set(preset_names)
    )


def test_every_source_has_verification_status_and_assumed_flagged() -> None:
    """Every hypothesis and capacity source carries an explicit verification_status."""
    raw = _raw_registry()
    for hyp_raw in raw["hypotheses"]:
        assert hyp_raw["verification_status"] in {"VERIFIED", "UNVERIFIED", "REMOVED"}
        src = hyp_raw["source"]
        if isinstance(src, dict):
            assert src["verification_status"] in {"VERIFIED", "UNVERIFIED", "REMOVED"}

    for slot in raw["torque_capacity"]:
        assert slot["verification_status"] in {"VERIFIED", "UNVERIFIED", "REMOVED"}
        assert isinstance(slot["source"], dict)
        assert slot["source"]["verification_status"] in {
            "VERIFIED",
            "UNVERIFIED",
            "REMOVED",
        }
        if slot.get("status") == "ASSUMED" or slot["verification_status"] != "VERIFIED":
            assert slot["verification_status"] == "UNVERIFIED"
            assert isinstance(slot.get("warning"), str) and slot["warning"].strip()


def test_lookup_functions_and_preset_resolution() -> None:
    """Test hypothesis(), convention(), preset(), and require_verified_for_display()."""
    reg = load_registry()
    h_rec = hypothesis("H-TORQUE-CAPACITY", registry=reg)
    assert isinstance(h_rec, HypothesisRecord)
    assert isinstance(h_rec.source, SourceRecord)
    assert h_rec.verification_status == "VERIFIED"

    c_rec = convention("C-FRAME-GRIP", registry=reg)
    assert isinstance(c_rec, ConventionRecord)

    h_set = preset("capacity_min_effort", registry=reg)
    assert isinstance(h_set, HypothesisSet)
    assert isinstance(h_set.preset, PresetRecord)
    assert len(h_set.hypotheses) >= 2
    assert len(h_set.conventions) >= 2

    verified = require_verified_for_display(h_set.hypothesis_ids, registry=reg)
    assert len(verified) == len(h_set.hypothesis_ids)


def test_negative_duplicate_hypothesis_id(tmp_path: Path) -> None:
    """Duplicate hypothesis IDs are rejected by load_registry."""
    raw = copy.deepcopy(_raw_registry())
    raw["hypotheses"].append(copy.deepcopy(raw["hypotheses"][0]))
    bad_path = tmp_path / "dup_hyp.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate hypothesis id"):
        load_registry(bad_path)


def test_negative_dangling_preset_reference(tmp_path: Path) -> None:
    """Presets referencing unknown hypothesis or convention IDs are rejected."""
    raw = copy.deepcopy(_raw_registry())
    raw["presets"][0]["hypothesis_ids"] = ["H-NONEXISTENT-999"]
    raw["presets"][0]["hypotheses"] = ["H-NONEXISTENT-999"]
    bad_path = tmp_path / "dangling_preset.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="H-NONEXISTENT-999"):
        load_registry(bad_path)


def test_negative_missing_source_verification_status(tmp_path: Path) -> None:
    """Sources missing verification_status are rejected."""
    raw = copy.deepcopy(_raw_registry())
    raw["hypotheses"][0]["verification_status"] = "INVALID_STATUS"
    bad_path = tmp_path / "bad_status.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="verification_status"):
        load_registry(bad_path)


def test_negative_unverified_citation_in_statement_or_label(tmp_path: Path) -> None:
    """User-facing statements and preset labels citing UNVERIFIED sources are rejected."""
    raw = copy.deepcopy(_raw_registry())
    raw["hypotheses"][0]["statement"] = (
        "Assumes Thornton 2025 hand-share split across lead and trail hands."
    )
    bad_path = tmp_path / "unverified_stmt.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Thornton"):
        load_registry(bad_path)

    raw2 = copy.deepcopy(_raw_registry())
    raw2["presets"][0]["label"] = "Bosch 15-40 ms Dead-Zone Preset"
    bad_path2 = tmp_path / "unverified_label.json"
    bad_path2.write_text(json.dumps(raw2), encoding="utf-8")
    with pytest.raises(ValueError, match="Bosch|15-40 ms"):
        load_registry(bad_path2)


def test_negative_unflagged_assumed_capacity(tmp_path: Path) -> None:
    """ASSUMED capacity slots without UNVERIFIED status or warning are rejected."""
    raw = copy.deepcopy(_raw_registry())
    for slot in raw["torque_capacity"]:
        if slot.get("status") == "ASSUMED":
            slot["warning"] = ""
            break
    bad_path = tmp_path / "unflagged_assumed.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="warning|ASSUMED"):
        load_registry(bad_path)


def test_require_verified_for_display_rejects_unverified(tmp_path: Path) -> None:
    """require_verified_for_display raises ValueError on an UNVERIFIED hypothesis."""
    raw = copy.deepcopy(_raw_registry())
    raw["hypotheses"].append(
        {
            "id": "H-EXPERIMENTAL-UNVERIFIED",
            "statement": "Experimental unverified prior held for negative test.",
            "parameters": {},
            "source": {
                "citation": "Experimental internal note",
                "doi_or_url": None,
                "table_or_figure": None,
                "verification_status": "UNVERIFIED",
            },
            "verification_status": "UNVERIFIED",
        }
    )
    custom_path = tmp_path / "with_unverified.json"
    custom_path.write_text(json.dumps(raw), encoding="utf-8")
    reg = load_registry(custom_path)
    with pytest.raises(ValueError, match="H-EXPERIMENTAL-UNVERIFIED"):
        require_verified_for_display(["H-EXPERIMENTAL-UNVERIFIED"], registry=reg)
