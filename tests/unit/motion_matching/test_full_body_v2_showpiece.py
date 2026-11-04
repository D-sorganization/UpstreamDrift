"""SHOW-A3 (#14552, D4): the committed v2 showpiece spec is derived data.

Drift test: committed bytes equal a fresh derivation from v1. Propagation test:
a v1 change (A2 grip translation or rotation) reaches the v2 document.
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


def test_check_mode_passes_on_main_and_fails_on_drift(tmp_path) -> None:
    assert main(["--check"]) == 0
    stale = tmp_path / "v2.json"
    stale.write_bytes(b"{}")
    assert main(["--check", "--v2", str(stale)]) == 1


@pytest.mark.parametrize("row, col", [(0, 3), (1, 1)])  # translation, rotation entry
def test_v1_grip_change_propagates_into_v2(v1, row, col) -> None:
    moved = copy.deepcopy(v1)
    hand = next(j for j in moved["joints"] if j["child"].endswith("/RHandStandoff"))
    hand["parent_to_base"][row][col] += 0.01

    def entry(doc: dict) -> float:
        by_name = {j["name"]: j for j in doc["joints"]}
        return by_name[hand["name"]]["parent_to_base"][row][col]

    base, changed = derive_showpiece_v2(v1), derive_showpiece_v2(moved)
    assert entry(changed) == pytest.approx(entry(base) + 0.01)
    assert changed["source_spec_sha256"] != base["source_spec_sha256"]


def test_derived_spec_is_v2_with_the_declared_trunk_coordinates(v1) -> None:
    doc = derive_showpiece_v2(v1)
    assert doc["schema_version"] == "full-body-v2"
    assert set(V2_TRUNK.coordinates) <= set(doc["coordinate_order"])
    assert doc["trunk_coupling"]["enabled"] is False
