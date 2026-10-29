"""Governance and verification for Chapter 69: Interface Capability Ellipsoids (#14052)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_design_manual_governance import verify_repository

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = REPO_ROOT / "manuals" / "upstreamdrift"
CHAPTER_FILE = MANUAL_ROOT / "chapters" / "69-interface-capability-ellipsoids.qmd"
INDEX_FILE = MANUAL_ROOT / "index.qmd"
REGISTRY_FILE = MANUAL_ROOT / "calculation-registry.json"

pytestmark = pytest.mark.unit


def test_chapter_file_exists_and_included_in_index() -> None:
    """Chapter 69 must exist and be included in index.qmd."""
    assert CHAPTER_FILE.is_file(), f"Missing chapter file: {CHAPTER_FILE}"
    index_text = INDEX_FILE.read_text(encoding="utf-8")
    expected_include = "{{< include chapters/69-interface-capability-ellipsoids.qmd >}}"
    assert expected_include in index_text, f"index.qmd must include {expected_include}"


def test_chapter_defines_capability_equations_and_contracts() -> None:
    """Chapter 69 must document operational-space capability matrices, drift offset, and visual style."""
    content = CHAPTER_FILE.read_text(encoding="utf-8")

    # Core equations and symbols
    assert "W = J A J^{\\top}" in content or "W = J A J^\\top" in content
    assert "\\Lambda = W^{-1}" in content
    assert (
        "Q_a = J A S T^2 S^{\\top} A^{\\top} J^{\\top}" in content
        or "Q_a = J A S T^2 S^\\top A^\\top J^\\top" in content
    )
    assert "c_a = J A (-h) + \\dot{J}\\dot{q}" in content
    assert (
        "Q_f = (J S T^{-2} S^{\\top} J^{\\top})^{-1}" in content
        or "Q_f = (J S T^{-2} S^\\top J^\\top)^{-1}" in content
    )

    # Interface conventions
    for interface in ("lead_grip", "trail_grip", "grip_midpoint"):
        assert interface in content

    # Styling and Okabe-Ito conventions
    assert "#0072B2" in content  # Okabe-Ito Blue (lead)
    assert "#D55E00" in content  # Okabe-Ito Vermillion (trail)
    assert "translucent shell" in content.lower()
    assert "wireframe" in content.lower()


def test_calculation_registry_records_capability_ellipsoids_blocker() -> None:
    """Calculation registry must record the provisional capability ellipsoids blocker."""
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    blocker_ids = {blocker["id"]: blocker for blocker in registry["blockers"]}
    assert "UP-CAP-interface-capability-ellipsoids" in blocker_ids
    entry = blocker_ids["UP-CAP-interface-capability-ellipsoids"]
    assert "#14052" in entry["owner"]
    assert "#14038" in entry["owner"]


def test_design_manual_governance_passes() -> None:
    """Repository governance verification must pass cleanly with the new chapter."""
    summary = verify_repository(REPO_ROOT)
    assert summary.manual_id == "upstreamdrift"
    assert summary.calculation_count == 0
    assert summary.release_status == "blocked-inventory-required"
