"""Forearm supination torque capacity (NSX-04, #12553).

Gulecyuz 2017 (Biomed Res Int, doi 10.1155/2017/1654796, Table 4, full text
read) measured isometric supination torque with a torque-sensor dynamometer.
It measured SUPINATION ONLY, so only the supination direction of each forearm
coordinate may be VERIFIED; the pronation direction stays an assumed prior.
Which sign is supination comes from the coordinate sign map, never a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.coordinate_sign_map import sign_for

pytestmark = pytest.mark.unit

PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "config"
    / "loop_nullspace_registry.json"
)
CITATION = "CIT-GULECYUZ-2017"


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def _cap(reg: dict, slot: str, direction: str) -> dict:
    (hit,) = [
        e
        for e in reg["torque_capacity"]
        if e["joint_slot"] == slot and e["direction"] == direction
    ]
    return hit


def _supination_direction(slot: str) -> str:
    """The generalised-coordinate sign whose motion is supination."""
    meaning = sign_for(slot).meaning.lower()
    assert ("supination" in meaning) != ("pronation" in meaning), meaning
    positive_is_supination = "supination" in meaning
    return "positive" if positive_is_supination else "negative"


def _other(direction: str) -> str:
    return "negative" if direction == "positive" else "positive"


@pytest.mark.parametrize(
    ("slot", "value"),
    [
        ("LFInput", 10.68),  # left = non-dominant arm of a right-handed golfer
        ("RFInput", 11.33),  # right = dominant arm
    ],
)
def test_supination_direction_is_sourced_from_gulecyuz_table_4(
    reg: dict, slot: str, value: float
) -> None:
    e = _cap(reg, slot, _supination_direction(slot))
    assert e["assumed"] is False and e["verification_status"] == "VERIFIED"
    assert e["source"]["citation_id"] == CITATION
    assert "Table 4" in e["source"]["location"]
    assert e["tau_max_nm"] == pytest.approx(value)
    assert e["scaling_rule"]["rule"] == "none"
    assert e["population"]["n"] == 13 and e["population"]["sex"] == "male"


@pytest.mark.parametrize("slot", ["LFInput", "RFInput"])
def test_pronation_direction_stays_an_assumed_prior(reg: dict, slot: str) -> None:
    """The paper never measured pronation, so no value may be read across."""
    e = _cap(reg, slot, _other(_supination_direction(slot)))
    assert e["assumed"] is True and e["verification_status"] == "UNVERIFIED"
    assert e["source"]["citation_id"] is None


def test_citation_is_registered_verified_with_the_table_read(reg: dict) -> None:
    (c,) = [c for c in reg["citations"] if c["id"] == CITATION]
    assert c["verification_status"] == "VERIFIED"
    assert c["doi"] == "10.1155/2017/1654796"
    assert "Table 4" in c["claim_checked"]
