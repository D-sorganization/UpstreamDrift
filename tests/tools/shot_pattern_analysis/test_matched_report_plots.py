"""Matrix summary heatmaps preserve unsupported and unavailable conditions."""

from __future__ import annotations

from typing import Any

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


def _realistic_cells(
    *,
    correlation_sign: float = -0.6,
    fixed_target: float | None = None,
    custom_scoring: bool = False,
    unequal_strike: bool = False,
) -> list[dict]:
    import numpy as np

    matrix = np.eye(7)
    if correlation_sign != 0.0:
        matrix[0, 1] = matrix[1, 0] = correlation_sign
    strike_h = 0.00635
    strike_v = 0.0127 if unequal_strike else 0.00635
    noise: dict[str, Any] = {
        "standard_deviations": [1.0, 0.5, 0.44704, strike_h, strike_v, 0.0, 0.0],
        "correlation": matrix.tolist() if correlation_sign != 0.0 else None,
    }
    clubs = ("driver", "seven_iron", "pitching_wedge")
    patterns = ("Straight", "Draw", "Fade")
    cells: list[dict[str, Any]] = []
    for club_index, club in enumerate(clubs):
        for lean in (0.0, 5.0, 10.0):
            for curve in (1.0, 2.0):
                scenario = f"{club}_lean{lean:g}_curve{curve:g}"
                stats: dict[str, Any] = {
                    pattern: {
                        "total_n": 100,
                        "supported_n": 100,
                        "supported_fraction": 1.0,
                        "target_rmse_m": 4.0 + club_index + curve,
                        "target_containment_fraction": 1.0,
                        "statistics": {"p95_radial_target_error_m": 8.0 + lean},
                    }
                    for pattern in patterns
                }
                course: dict[str, Any]
                if custom_scoring:
                    course = {
                        "status": "conditional_only",
                        "patterns": {
                            pattern: {"mean_strokes_gained": 0.05}
                            for pattern in patterns
                        },
                    }
                elif club == "driver":
                    course = {
                        "status": "available",
                        "scenario_id": "tee_fairway_rough",
                        "start_lie": "tee",
                        "start_distance_m": 400.0,
                        "hole_distance_m": 400.0,
                        "fairway_half_width_m": 15.0,
                        "green_radius_m": None,
                        "limitations": [
                            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction."
                        ],
                        "patterns": {
                            pattern: {"mean_strokes_gained": 0.02 * curve}
                            for pattern in patterns
                        },
                        "paired_n": 100,
                        "total_n": 100,
                    }
                else:
                    course = {
                        "status": "conditional_only",
                        "scenario_id": "approach_green_rough",
                        "start_lie": "fairway",
                        "start_distance_m": 128.0 if club == "seven_iron" else 77.0,
                        "green_radius_m": 15.0,
                        "fairway_half_width_m": None,
                        "limitations": [
                            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction."
                        ],
                        "patterns": {
                            pattern: {"mean_strokes_gained": -0.01 * curve}
                            for pattern in patterns
                        },
                        "paired_n": 100,
                        "total_n": 100,
                    }

                cells.append(
                    {
                        "scenario": scenario,
                        "status": "completed",
                        "config": {
                            "club": {"club_id": club, "shaft_lean_deg": lean},
                            "curve_scale": curve,
                            "target_x_m": fixed_target,
                            "noise": noise,
                        },
                        "summary": stats,
                        "course_scoring": course,
                    }
                )
    return cells


def test_contrast_luminance_chooses_white_on_dark_green_and_viridis_minimum() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis.matched_report_plots import _cell_text_color

    norm = Normalize(vmin=0.0, vmax=1.0)
    greens = plt.get_cmap("Greens")
    viridis = plt.get_cmap("viridis")

    # Dark Green 1.0 must select white text for contrast
    assert _cell_text_color(greens, norm, 1.0).lower() in ("#ffffff", "white")
    # Light Green 0.0 must select dark/black text
    assert _cell_text_color(greens, norm, 0.0).lower() in ("#111827", "black")

    # Dark Viridis minimum (0.0) must select white text for contrast
    assert _cell_text_color(viridis, norm, 0.0).lower() in ("#ffffff", "white")
    # Bright Viridis maximum (1.0) must select dark/black text
    assert _cell_text_color(viridis, norm, 1.0).lower() in ("#111827", "black")

    # Hatched unavailable cell (None value) must retain dark/black text
    assert _cell_text_color(greens, norm, None).lower() in ("#111827", "black")


def test_noise_primitive_units_and_correlation_sign() -> None:
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _derive_delivery_noise,
    )

    # Positive correlation stratum (r = -0.6 for shaft twist vs path)
    pos_cells = _realistic_cells(correlation_sign=-0.6)
    pos_label = _derive_delivery_noise(pos_cells)
    assert pos_label is not None
    assert "Shaft Twist SD 1°" in pos_label
    assert "Path SD 0.5°" in pos_label
    assert "Speed SD 1 mph" in pos_label
    assert "Strike SD 0.25 in on Each Axis" in pos_label
    assert "Shaft Twist–Path Correlation r = -0.6" in pos_label
    assert "Face-Path Correlation" not in pos_label

    # Negative correlation stratum (r = +0.6 for shaft twist vs path)
    neg_cells = _realistic_cells(correlation_sign=0.6)
    neg_label = _derive_delivery_noise(neg_cells)
    assert neg_label is not None
    assert "Shaft Twist–Path Correlation r = +0.6" in neg_label
    assert "Face-Path Correlation" not in neg_label

    # Independent delivery stratum
    indep_cells = _realistic_cells(correlation_sign=0.0)
    indep_label = _derive_delivery_noise(indep_cells)
    assert indep_label is not None
    assert "Independent Delivery" in indep_label
    assert "Correlation r" not in indep_label

    # Unequal strike standard deviations
    unequal_cells = _realistic_cells(correlation_sign=0.0, unequal_strike=True)
    unequal_label = _derive_delivery_noise(unequal_cells)
    assert unequal_label is not None
    assert "Each Axis" not in unequal_label
    assert "Horizontal Strike SD 0.25 in" in unequal_label
    assert "Vertical Strike SD 0.5 in" in unequal_label

    # Fixtures lacking noise must remain honestly unspecified, not fabricated
    unspecified = _synthetic_cells()
    assert _derive_delivery_noise(unspecified) is None


def test_target_context_distinguishes_within_lean_and_fixed_target() -> None:
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _derive_target_context,
    )

    within_lean = _realistic_cells(fixed_target=None)
    assert _derive_target_context(within_lean) == "Within-Lean Nominal Targets"

    fixed_ctrl = _realistic_cells(fixed_target=180.0)
    assert _derive_target_context(fixed_ctrl) == "Fixed-Target Control (180 m)"

    # Existing synthetic cells lacking target config remain unspecified
    assert _derive_target_context(_synthetic_cells()) is None


def test_sg_context_derives_tee_fairway_approach_and_carry_as_final_lie(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _derive_course_context,
        _has_carry_as_final_lie,
        export_matched_matrix_heatmaps,
    )

    cells = _realistic_cells()
    assert _derive_course_context(cells, "driver") == "Tee, 400 m, 30 m Fairway"
    assert (
        _derive_course_context(cells, "seven_iron")
        == "Fairway Approach, 15 m Green Radius"
    )
    assert (
        _derive_course_context(cells, "pitching_wedge")
        == "Fairway Approach, 15 m Green Radius"
    )
    assert _has_carry_as_final_lie(cells) is True

    # Custom scoring without preset context should remain honestly unspecified
    custom_cells = _realistic_cells(custom_scoring=True)
    assert _derive_course_context(custom_cells, "driver") is None
    assert _has_carry_as_final_lie(custom_cells) is False

    # Export SG heatmap and verify SVG contents and title casing
    paths = export_matched_matrix_heatmaps(
        cells,
        tmp_path,
        selection_label="FINAL; source=final_v1; delivery=nominal_match",
        metrics=("conditional_sg",),
        formats=("svg",),
    )
    svg = paths["conditional_sg_svg"].read_text(encoding="utf-8")
    assert "Tee, 400 m, 30 m Fairway" in svg
    assert "7-Iron" in svg
    assert "Fairway Approach, 15 m Green Radius" in svg
    assert "Pitching Wedge" in svg
    assert (
        "Carry-as-Final-Lie Approximation" in svg
        or "Carry-As-Final-Lie Approximation" in svg
    )
    assert "Shaft Twist–Path Correlation r = -0.6" in svg
    assert "Within-Lean Nominal Targets" in svg


def test_layout_visible_captions_and_nonoverlap_at_figsize(tmp_path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        export_matched_matrix_heatmaps,
    )

    cells = _realistic_cells(correlation_sign=-0.6)
    paths = export_matched_matrix_heatmaps(
        cells,
        tmp_path,
        selection_label=(
            "FINAL; source=final_v1; delivery=nominal_match; "
            "contact=incremental_coulomb_energetic; noise SDs=[1.0, 0.5, 0.44704, 0.00635, 0.00635, 0.0, 0.0]"
        ),
        metrics=("conditional_sg", "target_hit_fraction"),
        formats=("png", "svg"),
    )
    svg_sg = paths["conditional_sg_svg"].read_text(encoding="utf-8")
    # Subtitle must contain readable delivery noise and correlation, not raw unlabelled noise SDs
    assert "noise SDs=[1.0" not in svg_sg
    assert "Delivery Noise: Shaft Twist SD 1°, Path SD 0.5°" in svg_sg
    assert "Shaft Twist–Path Correlation r = -0.6" in svg_sg

    # Target hit fraction (Greens colormap at 1.0) must use white text for 100% containment
    svg_th = paths["target_hit_fraction_svg"].read_text(encoding="utf-8")
    assert (
        "#ffffff" in svg_th.lower()
        or "rgb(100%,100%,100%)" in svg_th.lower()
        or "white" in svg_th.lower()
    )


def test_matrix_caption_rejects_mixed_noise_profiles() -> None:
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _derive_delivery_noise,
    )

    cells = _realistic_cells()
    from copy import deepcopy

    cells[-1]["config"]["noise"] = deepcopy(cells[-1]["config"]["noise"])
    cells[-1]["config"]["noise"]["standard_deviations"][0] = 2.0
    with pytest.raises(ValueError, match="one noise profile"):
        _derive_delivery_noise(cells)


def test_matrix_annotation_contrast_across_entire_palettes() -> None:
    _load_api()
    import matplotlib
    from matplotlib.colors import Normalize, to_rgb
    from src.tools.shot_pattern_analysis.matched_report_plots import _cell_text_color

    def luminance(rgb):
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb
        ]
        return sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    for name in ("Greens", "viridis", "coolwarm"):
        cmap = matplotlib.colormaps[name]
        for index in range(256):
            value = index / 255
            bg = luminance(cmap(value)[:3])
            fg = luminance(to_rgb(_cell_text_color(cmap, Normalize(0, 1), value)))
            assert (max(bg, fg) + 0.05) / (min(bg, fg) + 0.05) >= 4.5, (name, value)


@pytest.mark.parametrize("dense_covariance", [False, True])
def test_sg_panel_titles_and_unavailable_reasons_fit_their_cells(
    tmp_path, monkeypatch, dense_covariance
) -> None:
    _load_api()
    import matplotlib.pyplot as plt
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        export_matched_matrix_heatmaps,
    )

    cells = _realistic_cells()
    if dense_covariance:
        import numpy as np

        correlation = np.full((7, 7), 0.36)
        np.fill_diagonal(correlation, 1.0)
        for cell in cells:
            cell["config"]["noise"]["correlation"] = correlation.tolist()
    cells[-1].update(
        status="unavailable",
        reason="nominal landing unreachable within bounds: residual=0.351838 m",
    )
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: captured.append(figure))
    export_matched_matrix_heatmaps(
        cells,
        tmp_path,
        selection_label="Saved Data",
        metrics=("conditional_sg",),
        formats=("svg",),
    )
    figure = captured[-1]
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        subtitle = next(
            text for text in figure.texts if "Delivery Noise:" in text.get_text()
        )
        header_bottom = subtitle.get_window_extent(renderer).y0
        panel_top = max(
            axis.title.get_window_extent(renderer).y1 for axis in figure.axes[:3]
        )
        assert header_bottom - panel_top >= 4 * figure.dpi / 72
        for axis in figure.axes[:9]:
            bounds = axis.get_window_extent(renderer)
            title = axis.title.get_window_extent(renderer)
            assert title.x0 >= bounds.x0 - 1 and title.x1 <= bounds.x1 + 1
            for annotation in axis.texts:
                if annotation.get_text().startswith("UNAVAILABLE"):
                    x, y = annotation.get_position()
                    lower = axis.transData.transform((x - 0.5, y + 0.5))
                    upper = axis.transData.transform((x + 0.5, y - 0.5))
                    box = annotation.get_window_extent(renderer)
                    assert box.y0 >= lower[1] and box.y1 <= upper[1]
    finally:
        original_close(figure)


def test_containment_caption_names_actual_circular_radius(tmp_path) -> None:
    export, _, _ = _load_api()
    cells = _realistic_cells()
    for cell in cells:
        cell["config"]["target_radius_m"] = 7.5
    paths = export(
        cells,
        tmp_path,
        selection_label="Saved Data",
        metrics=("target_hit_fraction",),
        formats=("svg",),
    )
    assert (
        "Circular Target Radius: 7.5 m" in paths["target_hit_fraction_svg"].read_text()
    )
    cells[-1]["config"]["target_radius_m"] = 15.0
    with pytest.raises(ValueError, match="one circular target radius"):
        export(
            cells,
            tmp_path / "mixed",
            selection_label="Mixed",
            metrics=("target_hit_fraction",),
            formats=("svg",),
        )
