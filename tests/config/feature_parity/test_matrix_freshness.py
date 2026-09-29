"""Freshness gate: the committed feature-parity matrix doc must match the
registry (issue #7445 / epic #7462).

Regenerates docs/development/feature_parity_matrix.md in memory from
src/config/feature_parity.json and compares it to the committed file.
"""

from __future__ import annotations

import pytest
from src.config.feature_parity_loader import FeatureParityRegistry

from scripts.generate_feature_parity_matrix import MATRIX_PATH, render_matrix

pytestmark = pytest.mark.parity


def test_committed_matrix_doc_exists() -> None:
    assert MATRIX_PATH.exists(), (
        f"Missing generated doc {MATRIX_PATH}. "
        "Run: python -m scripts.generate_feature_parity_matrix"
    )


def test_committed_matrix_matches_registry() -> None:
    registry = FeatureParityRegistry.load()
    rendered = render_matrix(registry)
    committed = MATRIX_PATH.read_text(encoding="utf-8")
    assert committed == rendered, (
        "docs/development/feature_parity_matrix.md is stale relative to "
        "src/config/feature_parity.json. "
        "Regenerate with: python -m scripts.generate_feature_parity_matrix"
    )


def test_render_matrix_is_deterministic() -> None:
    registry = FeatureParityRegistry.load()
    assert render_matrix(registry) == render_matrix(registry)
