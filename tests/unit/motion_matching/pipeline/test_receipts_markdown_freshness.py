"""Tests that docs/development/full_body_models/RECEIPTS.md is up to date with schema."""

from pathlib import Path

import pytest

from src.shared.python.motion_matching.pipeline.receipt_schema import (
    render_receipts_markdown,
)


@pytest.mark.unit
def test_receipts_markdown_freshness() -> None:
    """Verifies that RECEIPTS.md exists and is byte-identical to render_receipts_markdown()."""
    repo_root = Path(__file__).resolve().parents[4]
    doc_path = repo_root / "docs" / "development" / "full_body_models" / "RECEIPTS.md"

    assert doc_path.is_file(), (
        f"Expected documentation file at {doc_path} does not exist."
    )

    expected_content = render_receipts_markdown()
    actual_content = doc_path.read_text(encoding="utf-8")

    assert actual_content == expected_content, (
        "docs/development/full_body_models/RECEIPTS.md is out of date with "
        "src/shared/python/motion_matching/pipeline/receipt_schema.py. "
        "Regenerate with:\n"
        "python -m src.shared.python.motion_matching.pipeline.receipt_schema --markdown"
    )
