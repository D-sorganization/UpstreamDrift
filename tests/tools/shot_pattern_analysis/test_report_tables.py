"""Generated article tables: formatting contracts, traceability and freshness."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import report_tables as tables
from src.tools.shot_pattern_analysis.headline_selection import (
    condition_parts,
    headline_cell_names,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "docs/research/shot_pattern_analysis/phase2"
DATASET = PHASE2 / "statistics_v2/headline_dataset.json"
GENERATED = ROOT / "docs/research/shot_pattern_analysis/generated"


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_committed_tables_are_fresh(dataset):
    assert tables.stale_tables(dataset, GENERATED) == []


def test_article_inputs_every_generated_table():
    article = (GENERATED.parent / "shot_pattern_analysis.tex").read_text(
        encoding="utf-8"
    )
    for name in tables.render_all(_tiny()):
        assert f"\\input{{generated/{name.removesuffix('.tex')}}}" in article, name


def _lf_sha256(path: Path) -> str:
    """Evidence hashes are of LF blobs; tolerate an autocrlf working copy."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_dataset_binds_the_committed_evidence(dataset):
    assert dataset["inputs_sha256"]
    for relative, expected in dataset["inputs_sha256"].items():
        assert _lf_sha256(PHASE2 / relative) == expected, relative


def test_dataset_covers_the_headline_conditions_at_publication_resamples(dataset):
    assert dataset["bootstrap_resamples"] == 2000
    assert [cell["cell"] for cell in dataset["cells"]] == list(headline_cell_names())
    for cell in dataset["cells"]:
        reproduction = cell["nominal_reproduction"]
        assert reproduction["shots_compared"] == 30_000
        assert reproduction["status_mismatches"] == 0
        assert reproduction["max_abs_endpoint_difference_m"] == 0.0
        for pattern in ("Draw", "Fade"):
            interval = cell["paired_benefit_vs_straight"][pattern]
            band = cell["assumption_band"][pattern]
            assert band["band_lower_95"] <= interval["lower_95"]
            assert band["band_upper_95"] >= interval["upper_95"]
    assert list(dataset["contrasts"]) == [f"E{i:02d}" for i in range(1, 13)] + [
        "F1",
        "F2",
    ]


@pytest.mark.parametrize(
    "value,expected",
    [(0.0123456, "$+0.012$"), (-0.0004, "$0.000$"), (-1.5, "$-1.500$")],
)
def test_signed_uses_math_minus_and_no_negative_zero(value, expected):
    assert tables.signed(value) == expected


def test_units_are_per_100_shots_and_metres_with_yards():
    assert tables.sg(0.002893) == "$+0.289$"
    assert tables.sg_interval(-0.0001, 0.00025) == "$[-0.010,\\ +0.025]$"
    assert tables.metres(9.144) == "9.14 (10.00)"
    assert tables.count(10000) == "10{,}000"


@pytest.mark.parametrize(
    "lower,upper,expected",
    [(0.1, 0.2, "positive"), (-0.2, -0.1, "negative"), (-0.1, 0.1, "spans_zero")],
)
def test_direction(lower, upper, expected):
    assert tables.direction({"lower_95": lower, "upper_95": upper}) == expected


def _interval(estimate, half=0.001):
    return {
        "estimate": estimate,
        "lower_95": estimate - half,
        "upper_95": estimate + half,
    }


def _tiny():
    """One cell per club, exercising every renderer."""
    cells = []
    for key, name in (
        ("C01", headline_cell_names()[0]),
        ("C09", headline_cell_names()[8]),
        ("C17", headline_cell_names()[16]),
    ):
        club = name.split("_lean")[0]
        paired = {"Draw": _interval(0.002), "Fade": _interval(-0.001)}
        cells.append(
            {
                "key": key,
                "cell": name,
                "club_id": club,
                "protocol": "original_primary/1",
                "target_x_m": 150.0,
                "paired_n": 10000,
                "geometry": {
                    p: {
                        "supported_n": 10000,
                        "lateral_sd_m": 1.0,
                        "downrange_sd_m": 2.0,
                        "mean_carry_m": 149.0,
                        "target_rmse_m": 2.5,
                    }
                    for p in ("Straight", "Draw", "Fade")
                },
                "mean_strokes_gained": {"Straight": 0.1, "Draw": 0.102, "Fade": 0.099},
                "paired_benefit_vs_straight": paired,
                "assumption_band": {
                    p: {
                        "band_lower_95": paired[p]["lower_95"] - 0.003,
                        "band_upper_95": paired[p]["upper_95"],
                        "direction": "spans_zero" if p == "Draw" else "negative",
                        "variant_estimates": {
                            "newton_contact_law": paired[p]["estimate"]
                        },
                    }
                    for p in ("Draw", "Fade")
                },
                "nominal_reproduction": {
                    "shots_compared": 30000,
                    "status_mismatches": 0,
                    "max_abs_endpoint_difference_m": 0.0,
                },
                "variant_status": {"newton_contact_law": "available"},
                "variant_target_x_m": {"newton_contact_law": 150.25},
            }
        )
    contrasts = {
        "E01": {
            "comparison_kind": "shaft_rotation_sd",
            "cells": [headline_cell_names()[0], headline_cell_names()[2]],
            "source_protocols": ["original_control/1", "original_control/1"],
            "common_six_n": 10000,
            "effect_modification": {"Draw": _interval(0.003), "Fade": _interval(0.0)},
        }
    }
    return {"bootstrap_resamples": 2000, "cells": cells, "contrasts": contrasts}


def test_renderers_cover_every_table_and_report_direction_changes():
    rendered = tables.render_all(_tiny())
    assert set(rendered) == {
        "values.tex",
        "conditions.tex",
        "synthesis.tex",
        "head_variants.tex",
        "contrasts.tex",
        *(
            f"{kind}_{club}.tex"
            for kind in ("geometry", "paired")
            for club in ("driver", "seven_iron", "pitching_wedge")
        ),
    }
    assert all(text.startswith(tables.HEADER) for text in rendered.values())
    values = rendered["values.tex"]
    # Draw: positive -> spans 0; Fade: spans 0 -> negative, in all three cells.
    assert "\\newcommand{\\BandDirectionChanges}{6}" in values
    assert "\\newcommand{\\ReproShots}{90{,}000}" in values
    assert "Newton contact law & 0.000 & 0.25" in rendered["head_variants.tex"]
    assert "Spans 0" in rendered["paired_driver.tex"]
    assert "OC $\\to$ OC" in rendered["contrasts.tex"]


def test_stale_tables_reports_edits_and_missing_files(tmp_path):
    data = _tiny()
    for name, text in tables.render_all(data).items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert tables.stale_tables(data, tmp_path) == []
    (tmp_path / "synthesis.tex").write_text("edited", encoding="utf-8")
    (tmp_path / "values.tex").unlink()
    assert tables.stale_tables(data, tmp_path) == ["synthesis.tex", "values.tex"]


def test_unknown_head_variant_fails_closed():
    data = copy.deepcopy(_tiny())
    data["cells"][0]["variant_status"]["loft_x2"] = "available"
    with pytest.raises(ValueError, match="no label"):
        tables.variant_table(data)


def test_condition_parts_rejects_non_headline_names():
    assert condition_parts("seven_iron_lean0_curve2_rotation1_combined2") == (
        "7-Iron",
        "2",
        "1",
        "Combined",
    )
    with pytest.raises(ValueError):
        condition_parts("driver_lean5_curve2_rotation2_combined2")
