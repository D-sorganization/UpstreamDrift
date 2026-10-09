"""Matrix summary heatmaps preserve unsupported and unavailable conditions."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _load_api():
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _format_cell_metric,
        export_matched_matrix_heatmaps,
        metric_value,
    )

    return export_matched_matrix_heatmaps, metric_value, _format_cell_metric


def _synthetic_cells():
    cells = []
    clubs = ("driver", "seven_iron", "pitching_wedge")
    patterns = ("Straight", "Draw", "Fade")
    for club_index, club in enumerate(clubs):
        for lean in (0.0, 5.0, 10.0):
            for curve in (1.0, 2.0):
                scenario = f"{club}_lean{lean:g}_curve{curve:g}"
                stats = {
                    pattern: {
                        "total_n": 12,
                        "supported_n": 11,
                        "supported_fraction": 11 / 12,
                        "target_rmse_m": 4.0 + club_index + curve,
                        "target_containment_fraction": 0.50 + 0.05 * curve,
                        "statistics": {"p95_radial_target_error_m": 8.0 + lean},
                    }
                    for pattern in patterns
                }
                course = {
                    "status": "conditional_only",
                    "source_backed_status": "available",
                    "patterns": {
                        pattern: {"mean_strokes_gained": 0.02 * (club_index + curve)}
                        for pattern in patterns
                    },
                }
                if scenario == "seven_iron_lean5_curve2":
                    cells.append(
                        {
                            "scenario": scenario,
                            "status": "unavailable",
                            "reason": "nominal target unreachable",
                            "config": {
                                "club_id": club,
                                "shaft_lean_deg": lean,
                                "curve_scale": curve,
                            },
                        }
                    )
                else:
                    cells.append(
                        {
                            "scenario": scenario,
                            "status": "completed",
                            "config": {
                                "club_id": club,
                                "shaft_lean_deg": lean,
                                "curve_scale": curve,
                            },
                            "summary": stats,
                            "course_scoring": course,
                        }
                    )
    return cells


def test_metric_extraction_distinguishes_physics_support_and_conditional_sg() -> None:
    _, metric_value, _ = _load_api()
    cell = _synthetic_cells()[0]

    assert metric_value(cell, "target_rmse_m", "Draw") == pytest.approx(5.0)
    assert metric_value(cell, "supported_fraction", "Fade") == pytest.approx(11 / 12)
    assert metric_value(cell, "radial_p95_m", "Straight") == pytest.approx(8.0)
    assert metric_value(cell, "conditional_sg", "Draw") == pytest.approx(0.02)
    unavailable = next(
        item for item in _synthetic_cells() if item["status"] == "unavailable"
    )
    assert metric_value(unavailable, "target_rmse_m", "Draw") is None


def test_conditional_sg_annotation_uses_common_paired_population() -> None:
    _, _, format_cell_metric = _load_api()
    cell = _synthetic_cells()[0]
    cell["course_scoring"].update({"paired_n": 8, "total_n": 12})

    assert format_cell_metric(cell, "conditional_sg", "Draw", 0.02) == (
        "+0.02\nconditional n=8/12"
    )
    assert format_cell_metric(cell, "target_rmse_m", "Draw", 5.0) == "5.00\nn=11/12"


@pytest.mark.slow
def test_report_exports_supported_metric_heatmaps_and_marks_unavailable_cells(
    tmp_path,
) -> None:
    export, _, _ = _load_api()
    paths = export(
        _synthetic_cells(),
        tmp_path,
        selection_label="Synthetic Layout Pilot; combined profile; rotation SD 1°; curve ×2",
        metrics=("target_rmse_m",),
        formats=("svg",),
    )

    assert set(paths) == {"target_rmse_m_svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    svg = paths["target_rmse_m_svg"].read_text(encoding="utf-8")
    assert "UNAVAILABLE" in svg
    assert "nominal target" in svg
    assert "unreachable" in svg
    assert "Synthetic Layout Pilot" in svg


def test_export_rejects_unknown_metric_or_format(tmp_path) -> None:
    export, _, _ = _load_api()
    with pytest.raises(ValueError, match="known metric"):
        export(_synthetic_cells(), tmp_path, selection_label="Pilot", metrics=("bad",))
    with pytest.raises(ValueError, match="png, pdf, svg"):
        export(_synthetic_cells(), tmp_path, selection_label="Pilot", formats=("jpg",))


def test_report_rejects_duplicate_or_incomplete_matrix_axes(tmp_path) -> None:
    export, _, _ = _load_api()
    cells = _synthetic_cells()

    with pytest.raises(ValueError, match="duplicate matrix cell"):
        export(cells + [cells[0]], tmp_path, selection_label="Synthetic Pilot")
    with pytest.raises(ValueError, match="incomplete matrix axes"):
        export(cells[:-1], tmp_path, selection_label="Synthetic Pilot")
