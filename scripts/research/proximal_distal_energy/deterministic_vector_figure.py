"""Deterministic PDF/SVG persistence for governed research figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure


def _stem(path: Path) -> Path:
    if path.suffix.lower() in {".pdf", ".svg"}:
        return path.with_suffix("")
    return path


def _metadata(title: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    pdf: dict[str, Any] = {"CreationDate": None, "ModDate": None}
    svg: dict[str, Any] = {"Date": None}
    if title:
        pdf["Title"] = title
        svg["Title"] = title
    return pdf, svg


def _normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_vector_figure(
    figure: Figure,
    output: Path,
    *,
    salt: str,
    title: str | None = None,
    write_svg: bool = True,
    atomic_pdf: bool = False,
    bbox_inches: str | None = "tight",
) -> tuple[Path, Path | None]:
    """Write deterministic vector artifacts and return their final paths.

    Preconditions: ``salt`` must be non-empty and ``output`` must resolve to a
    file stem or PDF/SVG path. Postconditions: PDF metadata contains no clock
    time, SVG IDs are derived from the fixed salt, and any SVG text ends in one
    LF-terminated normalized line stream.
    """
    if not isinstance(salt, str) or not salt.strip():
        raise ValueError("salt must be a non-empty string")
    target = _stem(Path(output))
    target.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = target.with_suffix(".pdf")
    svg_path = target.with_suffix(".svg") if write_svg else None
    pdf_metadata, svg_metadata = _metadata(title)
    with mpl.rc_context({"svg.hashsalt": salt}):
        if atomic_pdf:
            temporary = pdf_path.with_name(f"{pdf_path.stem}.tmp.pdf")
            figure.savefig(
                temporary,
                bbox_inches=bbox_inches,
                metadata=pdf_metadata,
            )
            temporary.replace(pdf_path)
        else:
            figure.savefig(
                pdf_path,
                bbox_inches=bbox_inches,
                metadata=pdf_metadata,
            )
        if svg_path is not None:
            figure.savefig(
                svg_path,
                bbox_inches=bbox_inches,
                metadata=svg_metadata,
            )
            _normalize_svg(svg_path)
    return pdf_path, svg_path
