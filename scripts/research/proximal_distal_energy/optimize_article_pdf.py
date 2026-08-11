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
DEFAULT_LIMIT_BYTES = 1_048_576


def _load_fitz():
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for PDF optimization; install `pymupdf`."
        ) from error
    return fitz


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
    max_bytes: int = DEFAULT_LIMIT_BYTES,
) -> dict[str, int]:
    """Compact a PDF and fail closed if pages, URI links, or outline change."""
    source = input_path.resolve()
    destination = (output_path or input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".optimized.tmp")
    if temporary.exists():
        temporary.unlink()

    fitz = _load_fitz()
    before = _pdf_contract(source)
    with fitz.open(source) as document:
        document.save(
            temporary,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=1,
            compression_effort=100,
        )
    after = _pdf_contract(temporary)
    if after != before:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "PDF optimization changed the page/link/outline contract: "
            f"before={before}, after={after}"
        )
    optimized_bytes = temporary.stat().st_size
    if optimized_bytes > max_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Optimized PDF is {optimized_bytes} bytes; limit is {max_bytes} bytes"
        )
    source_bytes = source.stat().st_size
    temporary.replace(destination)
    return {
        "source_bytes": source_bytes,
        "optimized_bytes": optimized_bytes,
        "pages": after[0],
        "uri_links": after[1],
        "outline_entries": after[2],
    }


def main() -> None:
    """Optimize the canonical article PDF in place by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    args = parser.parse_args()
    result = optimize_pdf(args.input, args.output, max_bytes=args.max_bytes)
    print(
        "Optimized PDF: "
        f"{result['source_bytes']} -> {result['optimized_bytes']} bytes; "
        f"{result['pages']} pages; {result['uri_links']} URI links; "
        f"{result['outline_entries']} outline entries"
    )


if __name__ == "__main__":
    main()
