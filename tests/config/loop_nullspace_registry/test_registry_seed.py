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


def test_schema_version_and_empty_capacity(reg: dict) -> None:
    assert reg["schema_version"] == 1
    assert reg["torque_capacity"] == []


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
