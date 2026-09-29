"""Losslessly compact the rendered article while preserving reviewer links."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_PDF = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "proximal_distal_energy_transfer.pdf"
)


def _load_fitz():
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for PDF optimization; install `pymupdf`."
        ) from error
    return fitz


def _load_pikepdf():
    try:
        import pikepdf
    except ImportError as error:
        raise RuntimeError(
            "pikepdf is required for PDF linearization; install `pikepdf`."
        ) from error
    return pikepdf


def _pdf_contract(path: Path) -> tuple[int, int, int]:
    fitz = _load_fitz()
    with fitz.open(path) as document:
        uri_count = sum(
            link.get("kind") == fitz.LINK_URI
            for page in document
            for link in page.get_links()
        )
        return document.page_count, uri_count, len(document.get_toc())


def optimize_pdf(
    input_path: Path,
    output_path: Path | None = None,
    *,
    max_bytes: int | None = None,
) -> dict[str, int]:
    """Compact and linearize a PDF without changing pages, links, or outline."""
    source = input_path.resolve()
    destination = (output_path or input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".optimized.tmp")
    compacted = destination.with_suffix(destination.suffix + ".compacted.tmp")
    temporary.unlink(missing_ok=True)
    compacted.unlink(missing_ok=True)

    fitz = _load_fitz()
    pikepdf = _load_pikepdf()
    before = _pdf_contract(source)
    try:
        with fitz.open(source) as document:
            document.save(
                compacted,
                garbage=4,
                clean=True,
                deflate=True,
                deflate_images=True,
                deflate_fonts=True,
                use_objstms=1,
                compression_effort=100,
            )
        with pikepdf.open(compacted) as document:
            document.save(
                temporary,
                linearize=True,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
        after = _pdf_contract(temporary)
        if after != before:
            raise RuntimeError(
                "PDF optimization changed the page/link/outline contract: "
                f"before={before}, after={after}"
            )
        optimized_bytes = temporary.stat().st_size
        if max_bytes is not None and optimized_bytes > max_bytes:
            raise RuntimeError(
                f"Optimized PDF is {optimized_bytes} bytes; limit is {max_bytes} bytes"
            )
        source_bytes = source.stat().st_size
        with fitz.open(temporary) as document:
            fast_web_access = int(bool(document.is_fast_webaccess))
        if not fast_web_access:
            raise RuntimeError("PDF linearization did not produce fast web access")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        compacted.unlink(missing_ok=True)
    return {
        "source_bytes": source_bytes,
        "optimized_bytes": optimized_bytes,
        "pages": after[0],
        "uri_links": after[1],
        "outline_entries": after[2],
        "fast_web_access": fast_web_access,
    }


def main() -> None:
    """Optimize the canonical article PDF in place by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="Optional release-specific ceiling; no fixed ceiling is applied by default.",
    )
    args = parser.parse_args()
    result = optimize_pdf(args.input, args.output, max_bytes=args.max_bytes)
    print(
        "Optimized PDF: "
        f"{result['source_bytes']} -> {result['optimized_bytes']} bytes; "
        f"{result['pages']} pages; {result['uri_links']} URI links; "
        f"{result['outline_entries']} outline entries; fast web access enabled"
    )


if __name__ == "__main__":
    main()
