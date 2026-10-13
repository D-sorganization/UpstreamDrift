"""Preserve selected physical-pattern and condition identities in figure links."""

import pytest

from scripts.research.shot_pattern_analysis import final_report_inputs as inputs

pytestmark = pytest.mark.unit


def _effect():
    return {
        "comparison_kind": "shaft_rotation_sd",
        "source_protocols": ("original_primary/1", "original_primary/1"),
        "source_cells": (
            "driver_lean0_curve1_rotation1_combined2",
            "driver_lean0_curve1_rotation2_combined2",
        ),
    }


def _entry(pattern):
    effect = _effect()
    return {
        "figure_kind": f"density_{pattern.lower()}_social_png",
        "presentation": "social",
        "physical_pattern": pattern,
        "source_kind": "six_population_common_index_comparison",
        "comparison_kind": effect["comparison_kind"],
        "first_protocol": effect["source_protocols"][0],
        "second_protocol": effect["source_protocols"][1],
        "first_cell": effect["source_cells"][0],
        "second_cell": effect["source_cells"][1],
        "relative_path": f"comparisons/{pattern}.png",
    }


def test_selected_pair_keeps_all_three_physical_patterns_and_condition_identity(
    tmp_path,
):
    entries = [_entry(pattern) for pattern in ("Straight", "Draw", "Fade")]
    unrelated = {
        **_entry("Draw"),
        "second_cell": "driver_lean5_curve1_rotation2_combined2",
    }
    links = inputs.supplement_figure_links(
        entries + [unrelated],
        [_effect()],
        supplement_dir=tmp_path / "supplement",
        report_root=tmp_path,
    )
    assert len(links) == 3
    assert {link["physical_pattern"] for link in links} == {"Straight", "Draw", "Fade"}
    assert all(link["source_cells"] == _effect()["source_cells"] for link in links)
    assert all(
        link["club"] == "driver" and link["presentation"] == "social" for link in links
    )


@pytest.mark.parametrize("change", ["relabel", "escape", "missing_pattern"])
def test_selected_asset_cannot_be_relabelled_or_escape_its_inventory(tmp_path, change):
    entry = _entry("Draw")
    if change == "relabel":
        entry["presentation"] = "academic"
    elif change == "escape":
        entry["relative_path"] = "../outside.png"
    else:
        entry["physical_pattern"] = None
    with pytest.raises(ValueError):
        inputs.supplement_figure_links(
            [entry],
            [_effect()],
            supplement_dir=tmp_path / "supplement",
            report_root=tmp_path,
        )
