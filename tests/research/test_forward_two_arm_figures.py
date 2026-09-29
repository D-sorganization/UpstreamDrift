"""Publication-figure tests for the forward two-hand study."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.make_forward_two_arm_figures import (
    FIGURE_STEMS,
    render_figures,
)
from scripts.research.proximal_distal_energy.run_forward_two_arm_study import (
    run_study,
    write_study,
)

pytestmark = pytest.mark.scientific


def test_renderer_emits_paired_publication_figures(tmp_path: Path) -> None:
    record, arrays = run_study()
    data_dir = tmp_path / "data"
    figure_dir = tmp_path / "figures"
    write_study(data_dir, record=record, arrays=arrays)
    paths = render_figures(data_dir=data_dir, figure_dir=figure_dir)
    assert len(paths) == 2 * len(FIGURE_STEMS)
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".svg"):
            path = figure_dir / f"{stem}{suffix}"
            assert path in paths
            assert path.stat().st_size > 1_000
