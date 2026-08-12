"""Losslessly compact article figure PDFs before assembling the release."""

from __future__ import annotations

import argparse
from pathlib import Path

FIGURE_DIR = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "figures"
)


def optimize_figure(path: Path) -> tuple[int, int]:
    """Compact one PDF and preserve its page contract."""
    import fitz

    source_bytes = path.stat().st_size
    temporary = path.with_suffix(".pdf.optimized.tmp")
    temporary.unlink(missing_ok=True)
    with fitz.open(path) as document:
        page_count = document.page_count
        document.save(
            temporary,
            garbage=4,
            clean=True,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=1,
            compression_effort=100,
        )
    with fitz.open(temporary) as compacted:
        if compacted.page_count != page_count:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Figure page contract changed: {path}")
    optimized_bytes = temporary.stat().st_size
    if optimized_bytes < source_bytes:
        temporary.replace(path)
    else:
        temporary.unlink()
        optimized_bytes = source_bytes
    return source_bytes, optimized_bytes


def main() -> None:
    """Compact all vector figure PDFs and report the aggregate reduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    paths = sorted(args.figure_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF figures found in {args.figure_dir}")
    before = after = 0
    for path in paths:
        source_bytes, optimized_bytes = optimize_figure(path)
        before += source_bytes
        after += optimized_bytes
    print(f"Compacted {len(paths)} figure PDFs: {before} -> {after} bytes")


if __name__ == "__main__":
    main()
