"""Publication-figure contracts for moving-base flexible-club evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.make_moving_base_flexible_figures import (
    FIGURE_STEMS,
    render_figures,
)

pytestmark = pytest.mark.scientific


def test_figure_set_renders_pdf_and_svg(tmp_path: Path) -> None:
    outputs = render_figures(tmp_path)
    assert len(outputs) == 2 * len(FIGURE_STEMS)
    for stem in FIGURE_STEMS:
        pdf = tmp_path / f"{stem}.pdf"
        svg = tmp_path / f"{stem}.svg"
        assert pdf.read_bytes().startswith(b"%PDF")
        text = svg.read_text(encoding="utf-8")
        assert "<svg" in text
        assert pdf.stat().st_size > 5_000
        assert svg.stat().st_size > 5_000
