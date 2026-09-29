"""Publication-asset contracts for the spatial full-body study."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = (
    REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "figures"
)


@pytest.mark.parametrize(
    "stem",
    [
        "fig_spatial_full_body_force_geometry",
        "fig_spatial_cross_formulation_inverse_dynamics",
        "fig_spatial_full_body_falsification",
    ],
)
def test_spatial_figure_family_has_nonempty_pdf_and_svg(stem: str) -> None:
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"

    assert pdf.stat().st_size > 10_000
    assert svg.stat().st_size > 10_000
    assert svg.read_text(encoding="utf-8").count("<svg") == 1


def test_spatial_figures_use_scientifically_bounded_language() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in FIGURE_DIR.glob("fig_spatial*.svg")
    )

    assert "prescribed action" in combined
    assert "Inconclusive" in combined
    assert "Unsupported" in combined
    assert "validated human" not in combined.lower()
