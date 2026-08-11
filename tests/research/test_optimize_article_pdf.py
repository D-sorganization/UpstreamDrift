"""Contracts for lossless article-PDF compaction."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from scripts.research.proximal_distal_energy.optimize_article_pdf import optimize_pdf

pytestmark = pytest.mark.unit


def test_optimizer_preserves_pages_uri_links_and_outline(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "Mechanism Ladder")
    first.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(70, 55, 210, 80),
            "uri": "https://example.org/evidence",
        }
    )
    document.new_page().insert_text((72, 72), "Scientific Boundary")
    document.set_toc([[1, "Mechanism Ladder", 1], [1, "Scientific Boundary", 2]])
    document.save(source)
    document.close()

    result = optimize_pdf(source, output, max_bytes=1_000_000)

    assert result["pages"] == 2
    assert result["uri_links"] == 1
    assert result["outline_entries"] == 2
    assert output.is_file()


def test_optimizer_rejects_nonpositive_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    with pytest.raises(ValueError, match="positive"):
        optimize_pdf(source, max_bytes=0)
