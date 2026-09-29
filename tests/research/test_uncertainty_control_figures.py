"""Publication-figure contracts for the uncertainty/control study."""

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
        "fig_uncertainty_intervals_and_prcc",
        "fig_identifiability_audit",
        "fig_control_pareto_train_holdout",
        "fig_control_strategy_tradeoffs",
    ],
)
def test_uncertainty_control_figure_family_has_pdf_and_svg(stem: str) -> None:
    for suffix in ("pdf", "svg"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        assert path.is_file()
        assert path.stat().st_size > 10_000
