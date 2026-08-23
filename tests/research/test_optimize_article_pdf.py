"""Contracts for lossless article-PDF compaction."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from scripts.research.proximal_distal_energy import optimize_article_pdf

optimize_pdf = optimize_article_pdf.optimize_pdf

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

    result = optimize_pdf(source, output)

    assert result["pages"] == 2
    assert result["uri_links"] == 1
    assert result["outline_entries"] == 2
    assert result["fast_web_access"] == 1
    assert output.is_file()
    with fitz.open(output) as optimized:
        assert optimized.is_fast_webaccess


def test_optimizer_rejects_nonpositive_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    with pytest.raises(ValueError, match="positive"):
        optimize_pdf(source, max_bytes=0)


def test_optimizer_honors_an_explicit_release_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Evidence")
    document.save(source)
    document.close()

    with pytest.raises(RuntimeError, match="limit is 1 bytes"):
        optimize_pdf(source, max_bytes=1)


def test_optimizer_failure_preserves_destination_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Evidence")
    document.save(source)
    document.close()
    output.write_bytes(b"prior publication")

    class BrokenPikePdf:
        class ObjectStreamMode:
            generate = object()

        class Document:
            def __enter__(self) -> BrokenPikePdf.Document:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def save(self, path: Path, **_options: object) -> None:
                Path(path).write_bytes(b"partial publication")
                raise RuntimeError("planted linearization failure")

        @staticmethod
        def open(_path: Path) -> BrokenPikePdf.Document:
            return BrokenPikePdf.Document()

    monkeypatch.setattr(optimize_article_pdf, "_load_pikepdf", lambda: BrokenPikePdf)

    with pytest.raises(RuntimeError, match="planted linearization failure"):
        optimize_pdf(source, output)

    assert output.read_bytes() == b"prior publication"
    assert not output.with_suffix(".pdf.compacted.tmp").exists()
    assert not output.with_suffix(".pdf.optimized.tmp").exists()
