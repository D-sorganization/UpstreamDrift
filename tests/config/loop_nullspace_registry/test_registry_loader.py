"""Typed loader and citation gate for the NSX registry (NSX-11, #12556)."""

from __future__ import annotations

import copy

import pytest

from src.shared.python.loop_nullspace import registry as reg

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def shipped() -> reg.Registry:
    return reg.load_registry()


@pytest.fixture
def raw() -> dict:
    return copy.deepcopy(reg.load_raw_registry())


def _bad(raw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        reg.parse_registry(raw)


def test_shipped_registry_loads_and_is_consistent(shipped: reg.Registry) -> None:
    assert shipped.schema_version == 1
    assert {p.id for p in shipped.presets} >= {"envelope_only", "capacity_min_effort"}
    assert shipped.hypothesis("H-TORQUE-CAPACITY").id == "H-TORQUE-CAPACITY"


def test_hypothesis_lookup_unknown_raises() -> None:
    with pytest.raises(KeyError, match="H-NOPE"):
        reg.hypothesis("H-NOPE")


def test_preset_returns_hypothesis_set() -> None:
    hs = reg.preset("capacity_min_effort")
    assert isinstance(hs, reg.HypothesisSet)
    assert "H-EFFORT-CAPACITY" in hs.hypothesis_ids()
    assert hs.label == "Capacity-Weighted Minimum Effort"
    assert all(c.id.startswith("C-") for c in hs.conventions)


def test_preset_unknown_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        reg.preset("nope")


def test_require_verified_for_display_accepts_shipped_hypotheses() -> None:
    ids = reg.preset("envelope_only").hypothesis_ids()
    assert reg.require_verified_for_display(ids) == tuple(ids)


def test_require_verified_for_display_rejects_unverified(raw: dict) -> None:
    # H-GROSS-POWER-COST is in no preset, so it may legally be UNVERIFIED.
    target = next(h for h in raw["hypotheses"] if h["id"] == "H-GROSS-POWER-COST")
    target["source"]["citation_id"] = "CIT-X"
    target["verification_status"] = "UNVERIFIED"
    raw["citations"].append(
        {"id": "CIT-X", "reference": "r", "verification_status": "UNVERIFIED"}
    )
    r = reg.parse_registry(raw)
    with pytest.raises(ValueError, match="UNVERIFIED"):
        r.require_verified_for_display(["H-GROSS-POWER-COST"])


def test_require_verified_for_display_unknown_and_empty() -> None:
    with pytest.raises(ValueError, match="H-NOPE"):
        reg.require_verified_for_display(["H-NOPE"])
    with pytest.raises(TypeError):
        reg.require_verified_for_display("H-TORQUE-CAPACITY")  # type: ignore[arg-type]


def test_duplicate_hypothesis_ids_rejected(raw: dict) -> None:
    raw["hypotheses"].append(copy.deepcopy(raw["hypotheses"][0]))
    _bad(raw, "duplicate")


def test_duplicate_id_across_kinds_rejected(raw: dict) -> None:
    raw["conventions"][0]["id"] = raw["hypotheses"][0]["id"]
    _bad(raw, "duplicate")


def test_schema_missing_key_rejected(raw: dict) -> None:
    del raw["hypotheses"][0]["statement"]
    _bad(raw, "statement")


def test_schema_version_rejected(raw: dict) -> None:
    raw["schema_version"] = 2
    _bad(raw, "schema_version")


def test_bad_status_rejected(raw: dict) -> None:
    raw["hypotheses"][0]["verification_status"] = "MAYBE"
    _bad(raw, "verification_status")


def test_preset_with_unknown_hypothesis_rejected(raw: dict) -> None:
    raw["presets"][0]["hypotheses"].append("H-GHOST")
    _bad(raw, "H-GHOST")


def test_preset_with_unknown_convention_rejected(raw: dict) -> None:
    raw["presets"][0]["conventions"].append("C-GHOST")
    _bad(raw, "C-GHOST")


def test_unknown_citation_reference_rejected(raw: dict) -> None:
    raw["hypotheses"][0]["source"]["citation_id"] = "CIT-GHOST"
    _bad(raw, "CIT-GHOST")


def test_citation_without_status_rejected(raw: dict) -> None:
    del raw["citations"][0]["verification_status"]
    _bad(raw, "verification_status")


def test_verified_hypothesis_needs_verified_citation(raw: dict) -> None:
    raw["hypotheses"][0]["verification_status"] = "VERIFIED"
    raw["hypotheses"][0]["source"]["citation_id"] = None
    _bad(raw, "VERIFIED")


def test_statement_citing_unverified_source_rejected(raw: dict) -> None:
    raw["citations"].append(
        {
            "id": "CIT-BAD-2001",
            "reference": "Bad A. (2001)",
            "verification_status": "UNVERIFIED",
        }
    )
    raw["hypotheses"][0]["statement"] += " See CIT-BAD-2001."
    _bad(raw, "UNVERIFIED")


def test_preset_label_citing_unverified_source_rejected(raw: dict) -> None:
    raw["citations"].append(
        {
            "id": "CIT-BAD-2001",
            "reference": "Bad A. (2001)",
            "verification_status": "REMOVED",
        }
    )
    raw["presets"][0]["label"] = "Per CIT-BAD-2001"
    _bad(raw, "REMOVED")


def test_hypothesis_with_unverified_citation_in_preset_rejected(raw: dict) -> None:
    raw["citations"].append(
        {"id": "CIT-BAD-2001", "reference": "Bad", "verification_status": "UNVERIFIED"}
    )
    first = raw["presets"][0]["hypotheses"][0]
    for h in raw["hypotheses"]:
        if h["id"] == first:
            h["source"]["citation_id"] = "CIT-BAD-2001"
    _bad(raw, "UNVERIFIED")


def _cap(**over: object) -> dict:
    base = {
        "joint_slot": "lead_shoulder",
        "direction": "flexion",
        "tau_max_nm": 80.0,
        "scaling_rule": "none",
        "population": "adult male",
        "source": {"citation_id": None, "note": "placeholder"},
        "verification_status": "UNVERIFIED",
        "assumed": True,
    }
    base.update(over)
    return base


def test_assumed_capacity_is_flagged(raw: dict) -> None:
    raw["torque_capacity"] = [_cap()]
    r = reg.parse_registry(raw)
    assert [c.joint_slot for c in r.assumed_capacities()] == ["lead_shoulder"]


def test_uncited_capacity_must_be_assumed(raw: dict) -> None:
    raw["torque_capacity"] = [_cap(assumed=False)]
    _bad(raw, "assumed")


def test_uncited_capacity_must_be_unverified(raw: dict) -> None:
    raw["torque_capacity"] = [_cap(verification_status="VERIFIED")]
    _bad(raw, "VERIFIED")


def test_cited_verified_capacity_is_not_assumed(raw: dict) -> None:
    raw["torque_capacity"] = [
        _cap(
            source={"citation_id": raw["citations"][0]["id"], "note": "n"},
            verification_status="VERIFIED",
            assumed=False,
        )
    ]
    assert reg.parse_registry(raw).assumed_capacities() == ()


def test_capacity_value_must_be_positive(raw: dict) -> None:
    raw["torque_capacity"] = [_cap(tau_max_nm=0.0)]
    _bad(raw, "tau_max_nm")


def test_parse_registry_requires_mapping() -> None:
    with pytest.raises(TypeError):
        reg.parse_registry([])  # type: ignore[arg-type]
