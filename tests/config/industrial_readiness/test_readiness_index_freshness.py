"""Freshness gate: each committed readiness index must match its ledger.

Regenerates docs/operations/industrial-readiness-index.md (#9539) and
docs/operations/impact-zone-readiness-index.md (#9546) in memory from their
ledgers and compares them to the committed files.
"""

from __future__ import annotations

import pytest
from src.config.industrial_readiness_loader import IndustrialReadinessLedger

from scripts.generate_industrial_readiness_index import (
    INDEX_SPECS,
    IndexSpec,
    render_index,
)

pytestmark = pytest.mark.unit

_SPEC_IDS = [spec.ledger_path.stem for spec in INDEX_SPECS]


@pytest.mark.parametrize("spec", INDEX_SPECS, ids=_SPEC_IDS)
def test_committed_index_doc_exists(spec: IndexSpec) -> None:
    assert spec.index_path.exists(), (
        f"Missing generated doc {spec.index_path}. "
        "Run: python3 -m scripts.generate_industrial_readiness_index"
    )


@pytest.mark.parametrize("spec", INDEX_SPECS, ids=_SPEC_IDS)
def test_committed_index_matches_ledger(spec: IndexSpec) -> None:
    ledger = IndustrialReadinessLedger.load(spec.ledger_path)
    rendered = render_index(ledger, spec)
    committed = spec.index_path.read_text(encoding="utf-8")
    assert committed == rendered, (
        f"{spec.index_path.name} is stale relative to {spec.ledger_path.name}. "
        "Regenerate with: python3 -m scripts.generate_industrial_readiness_index"
    )


@pytest.mark.parametrize("spec", INDEX_SPECS, ids=_SPEC_IDS)
def test_render_index_is_deterministic(spec: IndexSpec) -> None:
    ledger = IndustrialReadinessLedger.load(spec.ledger_path)
    assert render_index(ledger, spec) == render_index(ledger, spec)
