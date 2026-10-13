"""Final filename contracts for fractional delivery conditions."""

import pytest

from src.tools.shot_pattern_analysis import academic_plots as academic
from src.tools.shot_pattern_analysis import matched_supplement_figures as figures

pytestmark = pytest.mark.unit


class RecordingFigure:
    def savefig(self, path, **kwargs):
        pass


def test_fractional_cells_keep_distinct_final_density_and_overhead_paths(tmp_path):
    outputs = []
    for label in ("driver_rotation1.5_combined2", "driver_rotation1.6_combined2"):
        stem = tmp_path / figures._slug(label)
        for kind in ("density", "overhead"):
            normalized = academic._validate_stem(stem.with_name(stem.name + "_" + kind))
            paths = academic._save_formats(RecordingFigure(), normalized, dpi=100)
            assert all(f"_{kind}." in path.name for path in paths.values())
            outputs.extend(paths.values())
    assert len(set(outputs)) == 12


def test_cell_name_only_reference_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        figures.validate_planned_figure_identities(
            [], [{"source_protocol": "test/1", "cell_name": "cell"}]
        )


def test_naming_helper_is_bound_to_report_sources():
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        _report_source_hashes,
    )

    assert "matched_figure_identity.py" in _report_source_hashes()
