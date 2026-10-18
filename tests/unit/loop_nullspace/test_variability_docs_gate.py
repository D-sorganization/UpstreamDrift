"""NSX-101 documentation gate: variability claims stay blocked until evidence exists.

Every variability claim type needs a publication-gate row naming its required
evidence, a blocked or deferred status, the blocking item and the preregistered
experiment that would unblock it.  No row may read as validated while the
external-validation record is still deferred.  Software correctness only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "docs/research/loop_nullspace/publication_gate.md"
DV = ROOT / "docs/development/planning/DV-12554.md"
CATALOG = ROOT / "docs/development/planning/catalog.json"
HELP = ROOT / "docs/help/loop_nullspace_variability.md"
PLOTS_HELP = ROOT / "docs/help/grip_wrench_plots.md"
QMD = ROOT / "manuals/upstreamdrift/chapters/65-loop-null-space.qmd"

SECTION = "## Variability Sensitivity Claims"
REQUIRED_CLAIMS = {
    "VAR-01": "absolute face-angle dispersion",
    "VAR-02": "variance budget",
    "VAR-03": "minimum-variance allocation",
    "VAR-04": "grip preload",
    "VAR-05": "co-contraction",
    "VAR-06": "delayed feedback",
    "VAR-07": "real-capture",
    "VAR-08": "strategy ranking",
    "VAR-09": "strategy coverage",
    "VAR-10": "speed-accuracy",
}
EXPERIMENTS = {
    "EXP-V1": "Calibrated Face and Launch Measurement",
    "EXP-V2": "Instrumented Grip",
    "EXP-V3": "Perturbation-Derived Impedance",
    "EXP-V4": "EMG Co-Contraction Ratios",
    "EXP-V5": "Randomized Strategy Interventions",
}
EXPERIMENT_FIELDS = (
    "Hypothesis",
    "Measurement",
    "Sample and Power",
    "Pass and Fail",
    "Unblocks",
)
ALLOWED_STATUS = {"BLOCKED", "DEFERRED"}


def _rows() -> dict[str, list[str]]:
    text = GATE.read_text(encoding="utf-8")
    assert SECTION in text, f"missing section {SECTION!r}"
    body = text.split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows: dict[str, list[str]] = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.fullmatch(r"VAR-\d\d", cells[0]):
            rows[cells[0]] = cells
    return rows


def test_every_variability_claim_type_has_a_row() -> None:
    rows = _rows()
    assert set(REQUIRED_CLAIMS) <= set(rows), set(REQUIRED_CLAIMS) - set(rows)
    for rid, key in REQUIRED_CLAIMS.items():
        assert key in rows[rid][1].lower(), (rid, rows[rid][1])


def test_rows_have_evidence_status_blocker_and_experiment() -> None:
    for rid, cells in _rows().items():
        assert len(cells) == 6, (rid, len(cells))
        _, claim, evidence, status, blocker, unblocked = cells
        assert claim and evidence and blocker and unblocked, rid
        assert status in ALLOWED_STATUS, (rid, status)
        assert re.search(r"#\d{4,5}|DV-\d+", blocker), (rid, blocker)
        assert re.search(r"EXP-V\d|NSX-\d+", unblocked), (rid, unblocked)


def test_no_row_reads_validated_while_blocker_is_open() -> None:
    entry = next(
        e
        for e in json.loads(CATALOG.read_text(encoding="utf-8"))["entries"]
        if e["id"] == "DV-12554"
    )
    assert entry["status"] == "deferred"
    for rid, cells in _rows().items():
        for cell in cells[3:5]:
            assert not re.search(r"(?<!un)(?<!not )\bvalidated\b", cell, re.I), rid
        assert cells[3] != "VALIDATED", rid


def test_rows_point_only_at_defined_experiments() -> None:
    dv = DV.read_text(encoding="utf-8")
    for rid, cells in _rows().items():
        for exp in re.findall(r"EXP-V\d", cells[5]):
            assert exp in EXPERIMENTS, (rid, exp)
            assert f"### {exp}" in dv, (rid, exp)


@pytest.mark.parametrize("exp", sorted(EXPERIMENTS))
def test_preregistered_experiment_is_complete(exp: str) -> None:
    dv = DV.read_text(encoding="utf-8")
    match = re.search(rf"### {exp} (.+)\n", dv)
    assert match and match.group(1).strip() == EXPERIMENTS[exp]
    block = dv.split(f"### {exp} ", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
    for field in EXPERIMENT_FIELDS:
        assert f"**{field}:**" in block, (exp, field)
    assert "TBD" in block, f"{exp} must mark unknown numbers TBD"


def test_help_guide_is_linked_and_states_the_limits() -> None:
    text = HELP.read_text(encoding="utf-8")
    assert "loop_nullspace_variability.md" in PLOTS_HELP.read_text(encoding="utf-8")
    for term in (
        "FAMILY_DEPENDENT",
        "UNAVAILABLE",
        "UNVALIDATED",
        "not coaching advice",
        "S10",
        "withheld",
        "hypothesis chip",
    ):
        assert term in text, term


def test_manual_chapter_has_a_variability_section_with_non_claims() -> None:
    text = QMD.read_text(encoding="utf-8")
    assert "## Variability Sensitivity" in text
    assert "Non-Claims" in text.split("## Variability Sensitivity", 1)[1]
