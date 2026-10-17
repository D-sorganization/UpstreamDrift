"""The whole-row de Leva update keeps the committed geometry (#13037)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import apply_de_leva_whole_rows as module
from src.shared.python.body_segment_parameters.de_leva_1996 import table4_row
from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    de_leva_table_sha256,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
CLUBS = ("driver", "iron7")


@pytest.fixture(scope="module")
def native() -> dict:
    return json.loads((REPO_ROOT / module.NATIVE).read_text(encoding="utf-8"))


def _committed(club: str) -> dict:
    path = DOCS / f"full_body_spec_anthro_{club}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _superseded(document: dict) -> dict:
    """The committed document as it was before #13037, up to the rebuilt solids."""
    old = copy.deepcopy(document)
    old["de_leva_table_sha256"] = "0" * 64
    old["subject"]["trunk_scale"] = 1.15
    for body in old["bodies"]:
        for solid in body["solids"]:
            if solid["name"].endswith(module.REBUILT_SOLIDS):
                solid["com_m"] = [0.0, 0.0, 0.0]
    return old


def test_preserved_trunk_scale_keeps_the_trunk_length() -> None:
    old_length = table4_row("trunk_supr_midh", "male").length_m
    scale = module.preserved_trunk_scale(1.15)
    assert old_length == pytest.approx(0.5319)
    assert scale * DE_LEVA_MALE["trunk"].length_m == pytest.approx(1.15 * old_length)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="trunk_scale"):
            module.preserved_trunk_scale(bad)


@pytest.mark.parametrize("club", CLUBS)
def test_committed_documents_carry_the_current_table(club: str, native) -> None:
    document = _committed(club)
    assert document["de_leva_table_sha256"] == de_leva_table_sha256()
    assert document["subject"]["trunk_scale"] == module.preserved_trunk_scale(1.15)
    assert module.apply_whole_rows(copy.deepcopy(document), native) is None


@pytest.mark.parametrize("club", CLUBS)
def test_update_reproduces_the_committed_document(club: str, native) -> None:
    committed = _committed(club)
    document = _superseded(committed)
    change = module.apply_whole_rows(document, native)
    assert change is not None
    assert change["max_geometry_change_m"] <= module.GEOMETRY_TOL
    assert change["trunk_scale"] == {
        "before": 1.15,
        "after": module.preserved_trunk_scale(1.15),
    }
    document["provenance"] = committed["provenance"]
    assert document == committed


def test_update_refuses_to_move_the_geometry(native) -> None:
    document = _committed("driver")
    document["de_leva_table_sha256"] = "0" * 64  # rescaling twice moves the trunk
    with pytest.raises(ValueError, match="geometry"):
        module.apply_whole_rows(document, native)


@pytest.mark.parametrize("club", CLUBS)
def test_current_table_rebuilds_only_the_trunk_solids(club: str, native) -> None:
    """A document on the current table with caudal-end trunk CMs keeps its
    trunk scale and has exactly the head and trunk-part solids rebuilt."""
    committed = _committed(club)
    document = copy.deepcopy(committed)
    document["provenance"] = committed["provenance"].replace(module.COM_ORIGIN_NOTE, "")
    for body in document["bodies"]:
        for solid in body["solids"]:
            if solid["name"].endswith(module.REBUILT_SOLIDS):
                solid["com_m"] = [0.0, 0.0, 0.0]
    change = module.apply_whole_rows(document, native)
    assert change is not None
    assert change["trunk_scale"]["before"] == change["trunk_scale"]["after"]
    assert len(change["rebuilt_solids"]) == len(module.REBUILT_SOLIDS)
    assert document == committed
