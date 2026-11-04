"""SHOW-A3 (#14552, D4): the committed v2 showpiece driver spec is derived data.

``full_body_spec_anthro_driver_v2.json`` is generated from the v1 driver spec by
``full_body_v2_showpiece``; a drift test pins the committed bytes to a fresh
derivation and a propagation test shows a v1 change (the A2 grip offset, for
example) reaches the v2 document.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.full_body_v2 import V2_TRUNK
from src.shared.python.motion_matching.full_body_v2_showpiece import (
    SHOWPIECE_V1,
    SHOWPIECE_V2,
    derive_showpiece_v2,
    main,
    showpiece_v2_bytes,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def v1() -> dict:
    return json.loads(Path(SHOWPIECE_V1).read_text(encoding="utf-8"))


def test_committed_v2_spec_equals_fresh_derivation_from_v1(v1) -> None:
    assert Path(SHOWPIECE_V2).read_bytes() == showpiece_v2_bytes(v1)


def test_check_mode_passes_on_main_and_fails_on_drift(tmp_path, v1) -> None:
    assert main(["--check"]) == 0
    stale = tmp_path / "v2.json"
    stale.write_bytes(b"{}")
    assert main(["--check", "--v2", str(stale)]) == 1


def test_v1_grip_change_propagates_into_v2(v1) -> None:
    moved = copy.deepcopy(v1)
    joint = next(j for j in moved["joints"] if j["child"].endswith("/RHandStandoff"))
    joint["parent_to_base"][0][3] += 0.01
    base = derive_showpiece_v2(v1)
    changed = derive_showpiece_v2(moved)
    name = joint["name"]
    got = {j["name"]: j for j in changed["joints"]}[name]["parent_to_base"][0][3]
    ref = {j["name"]: j for j in base["joints"]}[name]["parent_to_base"][0][3]
    assert got == pytest.approx(ref + 0.01)
    assert changed["source_spec_sha256"] != base["source_spec_sha256"]


def test_derived_spec_is_v2_with_the_declared_trunk_coordinates(v1) -> None:
    doc = derive_showpiece_v2(v1)
    assert doc["schema_version"] == "full-body-v2"
    assert set(V2_TRUNK.coordinates) <= set(doc["coordinate_order"])
    assert doc["trunk_coupling"]["enabled"] is False
