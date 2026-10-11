"""Analysis, gap rules and report rendering over the committed receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.lifting.pack_audit import analysis as an
from src.shared.python.lifting.pack_audit.gaps import ISSUES, REPO, derive_gaps
from src.shared.python.lifting.pack_audit.report import condense, render_markdown
from src.shared.python.lifting.pack_audit.stills import render_still

pytestmark = pytest.mark.unit

DOCS = Path(__file__).resolve().parents[4] / "docs" / "development" / "lifting"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads((DOCS / "pack_parity_baseline.json").read_text("utf-8"))


def test_receipt_covers_every_lift_and_engine(receipt):
    assert set(receipt["results"]) == set(an.LIFTS)
    for lift in an.LIFTS:
        assert an.engines_of(receipt, lift) == list(an.ENGINES)


def test_smoke_loaded_and_stepped_everywhere(receipt):
    for lift in an.LIFTS:
        for res in receipt["results"][lift].values():
            assert res["smoke"]["loaded"] and res["smoke"]["stepped"]


def test_reference_fk_within_standard_tolerance(receipt):
    for row in an.reference_fk_rows(receipt):
        assert row["max_abs_m"] < receipt["tolerances"]["position_m"]


def test_same_q_fk_parity_within_tolerance_except_bar(receipt):
    for row in an.parity_rows(receipt):
        for key in ("segments_max_m", "hands_max_m", "feet_max_m"):
            assert row["worst"][key][0] < receipt["tolerances"]["position_m"]


def test_bench_mass_is_the_only_mass_disagreement(receipt):
    spread = {r["lift"]: r["mass_spread_kg"] for r in an.parity_rows(receipt)}
    assert spread["bench_press"] > 1.0
    assert all(v < 1e-3 for k, v in spread.items() if k != "bench_press")


def test_every_gap_has_an_issue_in_a_known_repo(receipt):
    gaps = derive_gaps(receipt)
    assert gaps
    for gap in gaps:
        assert gap["issues"], gap["key"]
        for ref in gap["issues"]:
            assert ref.split("#")[0] in REPO.values()
        assert gap["evidence"], gap["key"]


def test_issue_map_references_positive_numbers():
    for per_engine in ISSUES.values():
        for engine, nums in per_engine.items():
            assert engine in REPO
            assert all(isinstance(n, int) and n > 0 for n in nums)


def test_unattached_right_hand_is_reported_for_drake_and_pinocchio(receipt):
    gap = next(g for g in derive_gaps(receipt) if g["key"] == "grip_attachment")
    assert gap["engines"] == ["drake", "pinocchio"]


def test_committed_markdown_matches_the_receipt(receipt):
    assert render_markdown(receipt) == (DOCS / "PACK_PARITY_BASELINE.md").read_text(
        "utf-8"
    )


def test_condense_is_idempotent(receipt):
    assert condense(receipt) == receipt


def test_render_still_validates_and_writes(tmp_path):
    import numpy as np

    with pytest.raises(ValueError, match="pelvis"):
        render_still({}, "x", tmp_path / "a.png")
    pos = {"pelvis": np.zeros(3), "barbell_shaft": np.array([0, 0, 1.0])}
    assert render_still(pos, "t", tmp_path / "a.png").stat().st_size > 0
