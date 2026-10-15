"""Seed-registry invariants for NSX-00 (#12549); the typed loader is NSX-11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "config"
    / "loop_nullspace_registry.json"
)
STATUSES = {"VERIFIED", "UNVERIFIED", "REMOVED", "NOT_APPLICABLE"}


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_schema_version_and_capacity_present(reg: dict) -> None:
    assert reg["schema_version"] == 1
    assert reg["torque_capacity"]


def test_ids_unique_and_prefixed(reg: dict) -> None:
    hyp = [h["id"] for h in reg["hypotheses"]]
    conv = [c["id"] for c in reg["conventions"]]
    assert len(set(hyp)) == len(hyp) and all(i.startswith("H-") for i in hyp)
    assert len(set(conv)) == len(conv) and all(i.startswith("C-") for i in conv)


def test_presets_reference_existing_ids(reg: dict) -> None:
    hyp = {h["id"] for h in reg["hypotheses"]}
    conv = {c["id"] for c in reg["conventions"]}
    for preset in reg["presets"]:
        assert set(preset["hypotheses"]) <= hyp, preset["id"]
        assert set(preset["conventions"]) <= conv, preset["id"]


def test_every_hypothesis_has_a_declared_status(reg: dict) -> None:
    for h in reg["hypotheses"]:
        assert h["verification_status"] in STATUSES, h["id"]
        assert h["statement"].strip(), h["id"]


def test_statements_never_rest_on_unverified_sources(reg: dict) -> None:
    status = {c["id"]: c["verification_status"] for c in reg["citations"]}
    for h in reg["hypotheses"]:
        cid = h["source"]["citation_id"]
        if cid is not None:
            assert status[cid] == "VERIFIED", h["id"]
        if h["verification_status"] in {"UNVERIFIED", "REMOVED"}:
            raise AssertionError(f"{h['id']} would present an unverified source")


SLOT_STATUS = {"VERIFIED", "UNVERIFIED"}
SPEC_DIR = (
    Path(__file__).resolve().parents[3] / "docs" / "development" / "full_body_models"
)


def test_torque_capacity_entries_obey_schema(reg: dict) -> None:
    cit = {c["id"]: c["verification_status"] for c in reg["citations"]}
    seen = set()
    for e in reg["torque_capacity"]:
        key = (e["joint_slot"], e["direction"])
        assert key not in seen, key
        seen.add(key)
        assert e["direction"] in {"negative", "positive"}, key
        assert e["tau_max_nm"] > 0, key
        assert e["verification_status"] in SLOT_STATUS, key
        assert e["scaling_rule"]["rule"] in {"none", "bw_height_normalized"}, key
        assert e["population"]["description"].strip(), key
        assert e["axis_mapping"] in {"hinge", "approximate-at-reference-pose"}, key
        assert e["axis_evidence"].strip(), key
        if e["assumed"]:
            assert e["verification_status"] == "UNVERIFIED", key
            assert e["rationale"].strip(), key
        else:
            assert e["verification_status"] == "VERIFIED", key
            assert cit[e["source"]["citation_id"]] == "VERIFIED", key
            assert e["source"]["location"].strip(), key


def test_every_joint_slot_of_both_specs_has_both_directions(reg: dict) -> None:
    keys = {(e["joint_slot"], e["direction"]) for e in reg["torque_capacity"]}
    for name in (
        "full_body_spec_anthro_driver.json",
        "full_body_spec_anthro_iron7.json",
    ):
        spec = json.loads((SPEC_DIR / name).read_text(encoding="utf-8"))
        for coord in spec["coordinate_order"]:
            if coord.startswith("Translation"):
                continue
            assert (coord, "negative") in keys, coord
            assert (coord, "positive") in keys, coord
