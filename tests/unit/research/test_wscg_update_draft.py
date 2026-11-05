"""The WSCG update manuscript draft (Issue #14441).

The draft must stay a draft, state its methods, cite the pinned run for every
figure and table, and quote the checked-in additivity record rather than
hand-copied numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from src.sim_run.manifest import validate_run_manifest
from src.sim_run.manifest_emitter import sha256_json

pytestmark = pytest.mark.unit

_RESEARCH = Path(__file__).resolve().parents[3] / (
    "docs/research/proximal_distal_energy_transfer"
)
_DRAFT = _RESEARCH / "WSCG_ZTCF_UPDATE_DRAFT.md"
_CAPTION = re.compile(r"^\*\*(Figure|Table) \d+\.\*\*(.*)$", re.MULTILINE)
_RUN_ID = re.compile(r"run_id `([0-9a-f]{64})`")


def _record(name: str) -> dict:
    return json.loads((_RESEARCH / "data" / name).read_text(encoding="utf-8"))


def _pinned_run_id() -> str:
    manifest = validate_run_manifest(_record("wscg_model_selection.json")["manifest"])
    return sha256_json(manifest.model_dump(mode="json"))


def test_stays_a_draft_behind_the_publication_gates() -> None:
    status = " ".join(_DRAFT.read_text(encoding="utf-8").split("\n\n")[1].split())
    for phrase in ("Draft", "Dieter's approval", "Assessment Board gate"):
        assert phrase in status


def test_methods_state_model_matching_conventions_and_limitations() -> None:
    text = _DRAFT.read_text(encoding="utf-8")
    for heading in (
        "## Methods",
        "### Model",
        "### Matching Approach",
        "### Conventions",
        "### Limitations",
    ):
        assert f"\n{heading}\n" in text


def test_every_figure_and_table_cites_the_pinned_run() -> None:
    captions = _CAPTION.findall(_DRAFT.read_text(encoding="utf-8"))
    assert {kind for kind, _ in captions} == {"Figure", "Table"}
    for _kind, caption in captions:
        assert _RUN_ID.findall(caption) == [_pinned_run_id()]


def test_quotes_the_checked_in_records() -> None:
    text = _DRAFT.read_text(encoding="utf-8")
    selection = _record("wscg_model_selection.json")
    additivity = _record("wscg_ztcf_additivity.json")
    assert f"`{selection['selected']}`" in text
    assert selection["manifest"]["model"]["revision"] in text
    assert f"{additivity['max_additivity_residual']:.1e}" in text
    assert f"{additivity['max_velocity_residual']:.1e}" in text
    assert f"{100 * additivity['median_zvcf_share']:.0f}%" in text
    assert f"{additivity['samples']} samples" in text
