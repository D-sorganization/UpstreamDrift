"""Phase-name gap rule and report column against the vendored standard (#11743)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.lifting.lift_spec import lift_phase_names
from src.shared.python.lifting.pack_audit import analysis as an
from src.shared.python.lifting.pack_audit.gaps import derive_gaps
from src.shared.python.lifting.pack_audit.report import render_markdown

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
DOCS = ROOT / "docs" / "development" / "lifting"
STD = ["setup", "first_pull", "knee_pass", "lockout"]


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads((DOCS / "pack_parity_baseline.json").read_text("utf-8"))


@pytest.fixture()
def receipt(committed: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(committed)


def _rename_to_standard(receipt: dict[str, Any], lift: str, engine: str) -> None:
    """Give one pack exactly the standard's phases (fields copied from its first)."""
    res = receipt["results"][lift][engine]
    template = res["phases"][0]
    res["phases"] = [{**template, "name": n} for n in lift_phase_names(lift)]


def _phase_names_gap(receipt: dict[str, Any]) -> dict[str, Any] | None:
    return next((g for g in derive_gaps(receipt) if g["key"] == "phase_names"), None)


# -- phase_name_mismatch ------------------------------------------------------


def test_identical_names_have_no_mismatch():
    assert an.phase_name_mismatch(list(STD), STD) is None


def test_renamed_phase_reports_missing_and_unexpected():
    names = ["setup", "break_floor", "knee_pass", "lockout"]
    reason = an.phase_name_mismatch(names, STD)
    assert reason is not None
    assert "missing: first_pull" in reason
    assert "unexpected: break_floor" in reason


def test_missing_phase_is_reported():
    reason = an.phase_name_mismatch(["setup", "knee_pass", "lockout"], STD)
    assert reason == "missing: first_pull"


def test_reordered_phases_are_reported():
    names = ["setup", "knee_pass", "first_pull", "lockout"]
    assert an.phase_name_mismatch(names, STD) == "order differs from the standard"


def test_duplicated_phase_is_reported():
    names = [*STD, "lockout"]
    assert an.phase_name_mismatch(names, STD) == "duplicate: lockout"


@pytest.mark.parametrize(
    ("names", "standard"),
    [("setup", STD), (STD, "setup"), (["setup", 3], STD), (STD, ["setup", None])],
)
def test_non_string_sequences_are_rejected(names: Any, standard: Any):
    with pytest.raises(TypeError):
        an.phase_name_mismatch(names, standard)


@pytest.mark.parametrize("standard", [[], ["setup", "setup"], ["setup", ""]])
def test_malformed_standard_is_rejected(standard: list[str]):
    with pytest.raises(ValueError):
        an.phase_name_mismatch(["setup"], standard)


# -- phase_rows / gap rule ----------------------------------------------------


def test_phase_rows_carry_a_reason_exactly_when_names_differ(committed):
    for row in an.phase_rows(committed):
        assert (row["names_reason"] is None) == row["names_match"]


def test_pack_with_standard_names_has_no_phase_names_gap(receipt):
    for lift in an.LIFTS:
        for engine in an.engines_of(receipt, lift):
            _rename_to_standard(receipt, lift, engine)
    assert _phase_names_gap(receipt) is None


def test_renamed_or_missing_phase_is_a_gap_with_a_reason(receipt):
    for lift in an.LIFTS:
        for engine in an.engines_of(receipt, lift):
            _rename_to_standard(receipt, lift, engine)
    dead = receipt["results"]["deadlift"]
    dead["drake"]["phases"] = dead["drake"]["phases"][:-1]  # drop a phase
    dead["mujoco"]["phases"][1]["name"] = "below_knee"  # rename a phase
    gap = _phase_names_gap(receipt)
    assert gap is not None
    assert gap["engines"] == ["drake", "mujoco"]
    assert gap["lift_story"] == "LIFT-3"
    assert gap["issues"], "every phase-name gap maps to a tracking issue"
    evidence = " | ".join(gap["evidence"])
    assert "drake/deadlift: missing: descent" in evidence
    assert "mujoco/deadlift: missing: first_pull; unexpected: below_knee" in evidence


def test_committed_receipt_reports_every_pack_off_standard(committed):
    gap = _phase_names_gap(committed)
    assert gap is not None
    assert gap["engines"] == sorted(an.ENGINES)
    assert len(gap["evidence"]) == sum(
        not r["names_match"] for r in an.phase_rows(committed)
    )


def test_report_has_a_names_column_built_on_names_match(committed):
    text = render_markdown(committed)
    assert "| Names Match Standard |" in text
    assert "Phase names differ from the standard" in text


# -- offline regeneration -----------------------------------------------------


def _load_script() -> Any:
    path = ROOT / "scripts" / "lifting" / "run_pack_parity_baseline.py"
    spec = importlib.util.spec_from_file_location("run_pack_parity_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_rerender_reproduces_the_committed_documents(tmp_path):
    script = _load_script()
    assert (
        script.main(
            [
                "--from-receipt",
                str(DOCS / "pack_parity_baseline.json"),
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    md = "PACK_PARITY_BASELINE.md"
    assert (tmp_path / md).read_text("utf-8") == (DOCS / md).read_text("utf-8")
    # The committed JSON is Prettier-formatted, so compare content, not bytes.
    js = "pack_parity_baseline.json"
    assert json.loads((tmp_path / js).read_text("utf-8")) == json.loads(
        (DOCS / js).read_text("utf-8")
    )


def test_offline_rerender_rejects_a_missing_receipt(tmp_path):
    script = _load_script()
    with pytest.raises(FileNotFoundError):
        script.main(
            ["--from-receipt", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path)]
        )
