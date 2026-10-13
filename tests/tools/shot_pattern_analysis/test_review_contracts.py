"""Contracts added by the #12409 review: style, preset flags and face inputs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.tools.shot_pattern_analysis import academic_plots, reporting
from src.tools.shot_pattern_analysis.core import PATTERN_NAMES
from src.tools.shot_pattern_analysis.figure_style import (
    DARK_PATTERN_COLORS,
    FIGURE_CREATOR,
    PUBLICATION_PATTERN_COLORS,
    reproducible_metadata,
)
from src.tools.shot_pattern_analysis.matched_config import (
    MatchedConfig,
    delivery_warnings,
    matched_preset,
    preset_modifications,
)
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics

pytestmark = pytest.mark.unit


def test_palettes_cover_every_pattern_and_are_shared():
    assert set(PUBLICATION_PATTERN_COLORS) == set(PATTERN_NAMES)
    assert set(DARK_PATTERN_COLORS) == set(PATTERN_NAMES)
    assert academic_plots._COLORS is PUBLICATION_PATTERN_COLORS
    assert reporting._COLORS is DARK_PATTERN_COLORS


@pytest.mark.parametrize("file_format,key", [("pdf", "CreationDate"), ("SVG", "Date")])
def test_vector_metadata_omits_the_timestamp(file_format, key):
    metadata = reproducible_metadata(file_format, {"Title": "Driver"})
    assert metadata == {"Title": "Driver", "Creator": FIGURE_CREATOR, key: None}


def test_raster_metadata_is_passed_through_and_inputs_are_validated():
    assert reproducible_metadata("png", {"Title": "x"}) == {"Title": "x"}
    with pytest.raises(ValueError, match="non-reproducible"):
        reproducible_metadata("pdf", {"CreationDate": "2026-10-12"})
    with pytest.raises(ValueError, match="non-empty"):
        reproducible_metadata("")


def test_pdf_render_is_byte_reproducible(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def render(name):
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], color=PUBLICATION_PATTERN_COLORS["Draw"])
        path = tmp_path / name
        figure.savefig(path, format="pdf", metadata=reproducible_metadata("pdf"))
        plt.close(figure)
        return path.read_bytes()

    assert render("a.pdf") == render("b.pdf")


@pytest.mark.parametrize("club_id", ["driver", "seven_iron", "pitching_wedge"])
def test_presets_are_unmodified(club_id):
    assert preset_modifications(matched_preset(club_id)) == ()


def test_any_changed_assumption_marks_the_preset_modified():
    driver = matched_preset("driver")
    changed = replace(driver, contact_law="maximum_dissipation_newton", cg_depth_m=0.02)
    assert preset_modifications(changed) == ("cg_depth_m", "contact_law")
    assert preset_modifications(replace(driver, club_id="custom", loft_deg=12)) == ()
    with pytest.raises(TypeError):
        preset_modifications({"club_id": "driver"})


def test_driver_forward_lean_is_flagged_only_for_the_driver():
    driver = matched_preset("driver")
    assert delivery_warnings(driver) == ()
    (warning,) = delivery_warnings(replace(driver, shaft_lean_deg=5.0))
    assert "5° forward shaft lean" in warning
    assert f"loft is {driver.loft_deg - 5:g}°" in warning
    wedge = replace(matched_preset("pitching_wedge"), shaft_lean_deg=5.0)
    assert delivery_warnings(wedge) == ()
    with pytest.raises(TypeError):
        delivery_warnings(None)


def test_matched_physics_rejects_a_nominal_face_it_would_ignore():
    with pytest.raises(ValueError, match="nominal face"):
        object.__new__(MatchedShotPhysics).simulate(
            face_deg=2.0, path_deg=0.0, config=MatchedConfig(), nominal_face_deg=1.0
        )
