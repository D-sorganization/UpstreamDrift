"""The report appendix split retains every source row and figure link."""

from __future__ import annotations

import json
import re
from collections import Counter
import hashlib
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis import build_final_report
from scripts.research.shot_pattern_analysis.report_tables import (
    resolved_selection_lines,
)

pytestmark = pytest.mark.unit


CLUBS = ("driver", "seven_iron", "pitching_wedge")
REPORT_DIR = Path("docs/research/shot_pattern_analysis/phase2")
ARCHIVE_MANIFEST = REPORT_DIR / "REPORT_PUBLICATION_SPLIT_ARCHIVE.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved_cells() -> list[dict]:
    fixture = Path(__file__).parent / "fixtures/shot_pattern_selected_synthesis.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["cells"]


def test_per_club_resolved_tables_preserve_all_original_rows_and_global_keys() -> None:
    cells = _resolved_cells()
    complete = "\n".join(resolved_selection_lines(cells))
    keyed_rows = [
        line
        for line in complete.splitlines()
        if line.startswith("|") and re.match(r"\| C\d{2} \|", line)
    ]
    assert len(keyed_rows) == 48  # 24 conditions in each of two keyed tables.

    split_rows: list[str] = []
    for club in CLUBS:
        rendered = "\n".join(resolved_selection_lines(cells, club_id=club))
        assert len(rendered.encode("utf-8")) < 51_200
        club_rows = [
            line
            for line in rendered.splitlines()
            if line.startswith("|") and re.match(r"\| C\d{2} \|", line)
        ]
        assert len(club_rows) == 16
        assert all(
            cell["cell"].startswith(club + "_")
            for cell in cells
            if f"`{cell['cell']}`" in rendered
        )
        split_rows.extend(club_rows)

    assert Counter(split_rows) == Counter(keyed_rows)
    for club, first_id in zip(CLUBS, (1, 9, 17), strict=True):
        rendered = "\n".join(resolved_selection_lines(cells, club_id=club))
        assert all(
            f"C{index:02d}" in rendered for index in range(first_id, first_id + 8)
        )


def test_club_appendices_partition_all_verified_figure_links() -> None:
    figure_links = [
        {
            "club": club,
            "kind": "density",
            "presentation": presentation,
            "source_protocol": "original_control/1",
            "source_kind": "individual_supported_endpoints",
            "relative_path": f"figures/{club}/{presentation}.png",
        }
        for club in CLUBS
        for presentation in ("academic", "social")
    ]
    interpretation = {
        "figure_links": figure_links,
        "resolved_selected_cells": _resolved_cells(),
    }

    appendices = build_final_report.render_club_appendices(interpretation)

    assert tuple(appendices) == CLUBS
    collected = []
    for club, markdown in appendices.items():
        assert len(markdown.encode("utf-8")) < 51_200
        expected_paths = {
            entry["relative_path"] for entry in figure_links if entry["club"] == club
        }
        found_paths = set(re.findall(r"\]\(([^)]+\.png)\)", markdown))
        assert found_paths == expected_paths
        assert all(
            entry["club"] == club or entry["relative_path"] not in markdown
            for entry in figure_links
        )
        collected.extend(found_paths)
    assert sorted(collected) == sorted(entry["relative_path"] for entry in figure_links)


def test_published_split_preserves_original_rows_links_and_archive_receipt() -> None:
    manifest = json.loads((REPO_ROOT / ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
    snapshots = {entry["original_path"]: entry for entry in manifest["snapshots"]}
    archived_report = (REPO_ROOT / snapshots["REPORT.md"]["archive_path"]).read_bytes()
    archived_receipt = (
        REPO_ROOT / snapshots["REPORT_PUBLICATION_RECEIPT.json"]["archive_path"]
    ).read_bytes()
    assert len(archived_report) == snapshots["REPORT.md"]["size_bytes"]
    assert (
        hashlib.sha256(archived_report).hexdigest() == snapshots["REPORT.md"]["sha256"]
    )
    assert (
        hashlib.sha256(archived_receipt).hexdigest()
        == snapshots["REPORT_PUBLICATION_RECEIPT.json"]["sha256"]
    )

    output_paths = [
        REPORT_DIR / "REPORT.md",
        *(REPORT_DIR / f"REPORT_{club.upper()}.md" for club in CLUBS),
    ]
    output_paths = [REPO_ROOT / path for path in output_paths]
    outputs = [path.read_text(encoding="utf-8") for path in output_paths]
    assert all(len(content.encode("utf-8")) < 51_200 for content in outputs)
    assert all(path.name in outputs[0] for path in output_paths[1:])

    original_lines = archived_report.decode("utf-8").splitlines()
    figures_start = original_lines.index("### Verified Academic and Social Figures")
    figures_end = original_lines.index(
        "### Separately Qualified Bounds Supplement", figures_start
    )
    original_figure_lines = [
        line
        for line in original_lines[figures_start:figures_end]
        if line.startswith("- [") and line.endswith(".png)")
    ]
    split_figure_lines = [
        line
        for appendix in outputs[1:]
        for line in appendix.splitlines()
        if line.startswith("- [") and line.endswith(".png)")
    ]
    assert Counter(split_figure_lines) == Counter(original_figure_lines)

    resolved_start = original_lines.index(
        "### Resolved Zero-Lean Curve and Variability Comparison"
    )
    resolved_end = original_lines.index(
        "### Primary Matrix Direction Counts", resolved_start
    )
    original_rows = [
        line
        for line in original_lines[resolved_start:resolved_end]
        if re.match(r"\| C\d{2} \|", line)
    ]
    split_rows = [
        line
        for appendix in outputs[1:]
        for line in appendix.splitlines()
        if re.match(r"\| C\d{2} \|", line)
    ]
    assert Counter(split_rows) == Counter(original_rows)
    assert {
        int(key) for row in split_rows for key in re.findall(r"\bC(\d{2})\b", row)
    } == set(range(1, 25))

    # Keep the historical scientific receipt untouched. The successor binds
    # only the layout migration and explicitly disclaims a fresh scientific build.
    historical_receipt = REPO_ROOT / REPORT_DIR / "REPORT_PUBLICATION_RECEIPT.json"
    assert historical_receipt.read_bytes() == archived_receipt
    layout_receipt = json.loads(
        (REPO_ROOT / REPORT_DIR / "REPORT_LAYOUT_SPLIT_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert "layout split only" in layout_receipt["purpose"]
    assert "no scientific report builder" in layout_receipt["purpose"]
    assert manifest["successor_receipt_path"] == "REPORT_LAYOUT_SPLIT_RECEIPT.json"
    assert manifest["successor_receipt_path"] != "REPORT_PUBLICATION_RECEIPT.json"
    assert (
        layout_receipt["archive_manifest_sha256"]
        == hashlib.sha256((REPO_ROOT / ARCHIVE_MANIFEST).read_bytes()).hexdigest()
    )
    assert (
        layout_receipt["source_archived_report"]["sha256"]
        == snapshots["REPORT.md"]["sha256"]
    )
    assert set(layout_receipt["output_sha256"]) == {path.name for path in output_paths}


def test_markdown_budget_guard_rejects_any_oversized_publication_file() -> None:
    with pytest.raises(ValueError, match="51200-byte budget.*REPORT_DRIVER.md"):
        build_final_report._assert_markdown_output_sizes(
            {"REPORT.md": "small", "REPORT_DRIVER.md": "x" * 51_200}
        )
