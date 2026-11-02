"""Pure-data layer of the putting analytics tool (#14177)."""

from __future__ import annotations

import pytest

from src.shared.python.putting_analytics import service
from src.tools.putting_analytics_gui.core import (
    MODES,
    SERVICE_FUNCTIONS,
    StudyForm,
    flat_rows,
    parse_classes,
    parse_floats,
    parse_reads,
    result_rows,
    study_args,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FORM = StudyForm(
    n=40,
    baseline_distances="1, 3 6 10 20 40",
    baseline_putts="1.0 1.2 1.5 1.7 1.9 2.2",
    aims="-2 0 2",
    paces="1 2",
    extra={"reads": "1 1\n# comment\n0.6 1", "classes": "3 1 uphill 10 3"},
)


def test_parse_floats_accepts_commas_and_spaces() -> None:
    assert parse_floats("1, 2 3", "aims") == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("text", ["", "1 x"])
def test_parse_floats_names_the_field(text: str) -> None:
    with pytest.raises(ValueError, match="aims"):
        parse_floats(text, "aims")


def test_parse_reads_skips_comments_and_blank_lines() -> None:
    assert parse_reads("# hint\n\n1 1\n0.5 1.2") == [
        {"slope_scale": 1.0, "stimp_ratio": 1.0},
        {"slope_scale": 0.5, "stimp_ratio": 1.2},
    ]


@pytest.mark.parametrize("text", ["# only a comment", "1 2 3"])
def test_parse_reads_rejects_empty_or_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        parse_reads(text)


def test_parse_classes_builds_service_classes() -> None:
    assert parse_classes("6 1 uphill 10 4") == [
        {
            "distance_ft": 6.0,
            "slope_pct": 1.0,
            "orientation": "uphill",
            "stimp": 10.0,
            "count": 4.0,
        }
    ]


@pytest.mark.parametrize("text", ["", "6 1 uphill 10"])
def test_parse_classes_rejects_empty_or_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        parse_classes(text)


def test_study_args_adds_mode_specific_fields() -> None:
    assert "reads" not in study_args(FORM, "optimize")
    assert len(study_args(FORM, "misread_cost")["reads"]) == 2
    assert study_args(FORM, "library_study")["classes"][0]["orientation"] == "uphill"


def test_study_args_rejects_unknown_mode_and_blank_baseline() -> None:
    with pytest.raises(ValueError, match="mode"):
        study_args(FORM, "variance")
    with pytest.raises(ValueError, match="baseline distances"):
        study_args(StudyForm(), "optimize")


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs_through_the_shared_service(mode: str) -> None:
    result = getattr(service, SERVICE_FUNCTIONS[mode])(study_args(FORM, mode))
    rows = result_rows(mode, result)
    assert rows
    assert all(isinstance(name, str) and isinstance(value, str) for name, value in rows)


def test_optimize_rows_lead_with_the_best_action() -> None:
    result = service.optimize_putt(study_args(FORM, "optimize"))
    rows = result_rows("optimize", result)
    assert rows[0][0] == "Best"
    assert rows[0][1].startswith("aim ")
    assert any(name == "#1" for name, _ in rows)


def test_flat_rows_index_lists_of_mappings() -> None:
    rows = flat_rows({"bands": [{"lo": 1.0}, {"lo": 2.0}], "aims": [1, 2], "n": 3})
    assert rows == [
        ("bands[0].lo", "1"),
        ("bands[1].lo", "2"),
        ("aims", "1, 2"),
        ("n", "3"),
    ]
