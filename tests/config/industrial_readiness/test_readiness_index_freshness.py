"""Freshness gate: the committed readiness index must match the ledger (#9539).

Regenerates docs/operations/industrial-readiness-index.md in memory from
src/config/industrial_readiness.json and compares it to the committed file.
"""

from __future__ import annotations

import pytest
from src.config.industrial_readiness_loader import IndustrialReadinessLedger

from scripts.generate_industrial_readiness_index import INDEX_PATH, render_index

pytestmark = pytest.mark.unit


def test_committed_index_doc_exists() -> None:
    assert INDEX_PATH.exists(), (
        f"Missing generated doc {INDEX_PATH}. "
        "Run: python3 -m scripts.generate_industrial_readiness_index"
    )


def test_committed_index_matches_ledger() -> None:
    ledger = IndustrialReadinessLedger.load()
    rendered = render_index(ledger)
    committed = INDEX_PATH.read_text(encoding="utf-8")
    assert committed == rendered, (
        "docs/operations/industrial-readiness-index.md is stale relative to "
        "src/config/industrial_readiness.json. "
        "Regenerate with: python3 -m scripts.generate_industrial_readiness_index"
    )


def test_render_index_is_deterministic() -> None:
    ledger = IndustrialReadinessLedger.load()
    assert render_index(ledger) == render_index(ledger)
