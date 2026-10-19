"""Render the native-versus-projected bilateral-contact discrepancy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)
from scripts.research.proximal_distal_energy.make_articulated_native_constraint_discrepancy_figure_support import (
    load_discrepancy_series,
    render_discrepancy_panels,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_native_constraint_discrepancy.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_native_constraint_discrepancy.pdf"


def make_figure(output: Path = OUTPUT) -> Path:
    """Render force, closure, and trajectory-discrepancy observables."""

    series = load_discrepancy_series(DATA)
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), layout="constrained")
    render_discrepancy_panels(figure, axes, series)
    save_vector_figure(
        figure,
        output,
        salt="articulated-native-constraint-discrepancy-v1",
        write_svg=False,
        atomic_pdf=True,
    )
    plt.close(figure)
    return output


def main() -> None:
    print(make_figure())


if __name__ == "__main__":
    main()
