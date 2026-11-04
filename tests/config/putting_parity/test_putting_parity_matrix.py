"""Putting feature parity matrix contract (#14456, epic #14452).

Every advanced putting feature is mapped to the shared core/API, PyQt6,
React, Tauri and the offline studio. Gaps are tracked, not hidden: each
partial or missing cell names its issue and shows up as a strict xfail
below, so landing the feature (and flipping the cell) is visible here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.putting_parity_loader import (
    DOC_PATH,
    REQUIRED_FEATURES,
    SURFACES,
    PuttingParityMatrix,
    render_markdown,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX = PuttingParityMatrix.load()


def test_every_required_feature_is_mapped() -> None:
    assert set(_MATRIX.feature_ids()) == set(REQUIRED_FEATURES)


def test_every_feature_has_all_five_surfaces() -> None:
    assert SURFACES == ("core_api", "pyqt6", "react", "tauri", "studio")
    for feature in _MATRIX.features:
        assert tuple(feature.cells) == SURFACES, feature.feature_id


def test_evidence_paths_exist() -> None:
    for feature, surface, cell in _MATRIX.iter_cells():
        for path in cell.evidence:
            assert (_REPO_ROOT / path).exists(), f"{feature}/{surface}: {path}"


def test_present_and_partial_cells_cite_evidence() -> None:
    for feature, surface, cell in _MATRIX.iter_cells():
        if cell.status != "missing":
            assert cell.evidence, f"{feature}/{surface} has no evidence"


def test_gaps_name_a_tracking_issue() -> None:
    for feature, surface, cell in _MATRIX.iter_cells():
        if cell.status != "present":
            assert cell.issue, f"{feature}/{surface} gap has no issue"


def test_tauri_inherits_react() -> None:
    # The Tauri shell only wraps the React build (ui/src-tauri adds no
    # putting code), so its cell can never claim more than React's.
    for feature in _MATRIX.features:
        react, tauri = feature.cells["react"], feature.cells["tauri"]
        assert tauri.status == react.status, feature.feature_id
        assert tauri.issue == react.issue, feature.feature_id


def test_committed_doc_matches_matrix() -> None:
    committed = DOC_PATH.read_text(encoding="utf-8")
    assert committed == render_markdown(_MATRIX), (
        "Stale doc. Regenerate: python -m src.config.putting_parity_loader --write"
    )


@pytest.mark.parametrize(
    ("feature", "surface"),
    [(f, s) for f, s, cell in _MATRIX.iter_cells() if cell.status != "present"],
)
@pytest.mark.xfail(strict=True, reason="tracked putting parity gap (#14452)")
def test_parity_gap(feature: str, surface: str) -> None:
    assert _MATRIX.cell(feature, surface).status == "present"
