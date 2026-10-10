"""Regression tests for academic paired-density row spacing (issue #12131).

Ensures inter-row decorated vertical clearance is positive and readable across
all three columns in academic six-panel density layouts, specifically for narrow
and equal-aspect dispersion distributions such as Pitching Wedge and 7-Iron.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_plotting_deps() -> tuple[Any, Any]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt

        return FigureCanvasAgg, plt
    except (ImportError, OSError) as exc:
        pytest.skip(f"Matplotlib unavailable: {exc}")


def _synthetic_condition(
    club_id: str,
    target_x_m: float,
    scale_x_m: float,
    scale_y_m: float,
    n_shots: int = 600,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[list[float]]]]:
    """Build representative paired-condition data with controlled aspect ratio."""
    import numpy as np

    first_ref = {
        "cell": f"{club_id}_lean0_curve1_rotation1_shaft_only",
        "source_protocol": "original_control/1",
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": 0.0},
            "curve_scale": 1.0,
            "noise": {"standard_deviations": [1.0, 0, 0, 0, 0, 0, 0]},
            "n_shots": 10_000,
        },
        "summary": {"target_x_m": target_x_m},
    }
    second_ref = {
        "cell": f"{club_id}_lean10_curve1_rotation1_shaft_only",
        "source_protocol": "original_control/1",
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": 10.0},
            "curve_scale": 1.0,
            "noise": {"standard_deviations": [1.0, 0, 0, 0, 0, 0, 0]},
            "n_shots": 10_000,
        },
        "summary": {"target_x_m": target_x_m},
    }
    rng = np.random.default_rng(seed)
    point_sets: dict[str, list[list[float]]] = {}
    for pattern in ("Straight", "Draw", "Fade"):
        for cond_label in ("Lean 0°", "Lean 10°"):
            points = rng.normal(
                loc=[target_x_m, 0.0],
                scale=[scale_x_m, scale_y_m],
                size=(n_shots, 2),
            )
            point_sets[f"{pattern} · {cond_label}"] = points.tolist()
    return first_ref, second_ref, point_sets


@pytest.mark.parametrize(
    "case_name,club_id,target_x_m,scale_x_m,scale_y_m",
    [
        ("pitching_wedge_shaft_only_narrow", "pitching_wedge", 77.402, 2.0, 2.0),
        ("seven_iron_shaft_only_narrow", "seven_iron", 148.5, 4.0, 4.0),
        ("seven_iron_combined_noise", "seven_iron", 148.5, 5.0, 3.8),
    ],
)
def test_academic_density_row_gutter_has_readable_positive_clearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    club_id: str,
    target_x_m: float,
    scale_x_m: float,
    scale_y_m: float,
) -> None:
    """Academic 6-panel density layout must maintain positive decorated row clearance."""
    canvas_agg_cls, _ = _load_plotting_deps()

    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _write_pair_density_figure,
    )

    first_ref, second_ref, point_sets = _synthetic_condition(
        club_id=club_id,
        target_x_m=target_x_m,
        scale_x_m=scale_x_m,
        scale_y_m=scale_y_m,
    )

    captured_figures: list[Any] = []
    original_save = msf._save_figure

    def capturing_save(
        figure: Any, stem: Path, *args: Any, **kwargs: Any
    ) -> dict[str, Path]:
        captured_figures.append(figure)
        return original_save(figure, stem, *args, **kwargs)

    monkeypatch.setattr(msf, "_save_figure", capturing_save)

    output_stem = tmp_path / case_name
    outputs = _write_pair_density_figure(
        point_sets,
        first=first_ref,
        second=second_ref,
        comparison="across_lean",
        first_label="Lean 0°",
        second_label="Lean 10°",
        target_x=target_x_m,
        common_n=600,
        output_stem=output_stem,
    )

    # 1. Output files are created on disk
    for ext in ("png", "pdf", "svg"):
        assert ext in outputs, f"Missing output format {ext}"
        assert outputs[ext].exists(), f"File does not exist: {outputs[ext]}"
        assert outputs[ext].stat().st_size > 0, f"File is empty: {outputs[ext]}"

    assert len(captured_figures) == 1
    figure = captured_figures[0]

    # 2. Figure dimensions are preserved (16x9 inches)
    assert tuple(figure.get_size_inches()) == (16.0, 9.0)

    # 3. Re-draw onto Agg canvas to obtain exact window bounding boxes
    canvas_agg_cls(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    # 4. Measure decorated vertical clearance across all 3 columns
    for col_index in range(3):
        top_axis = figure.axes[col_index]
        bot_axis = figure.axes[3 + col_index]

        # Typography standards: retained font sizes
        assert top_axis.xaxis.label.get_fontsize() == 10.0
        assert bot_axis.title.get_fontsize() == 10.0

        # Equal aspect ratio preserved (equal maps to 1.0 in matplotlib)
        assert top_axis.get_aspect() in ("equal", 1.0)
        assert bot_axis.get_aspect() in ("equal", 1.0)

        # Shared limits preserved
        assert top_axis.get_xlim() == bot_axis.get_xlim()
        assert top_axis.get_ylim() == bot_axis.get_ylim()

        top_xlabel_box = top_axis.xaxis.label.get_window_extent(renderer)
        bot_title_box = bot_axis.title.get_window_extent(renderer)

        # Clearance: bottom edge of upper row's xlabel minus top edge of lower row's title
        clearance_pt = (top_xlabel_box.y0 - bot_title_box.y1) * 72.0 / figure.dpi

        # Minimum required readable positive clearance: >= 6.0 pt
        assert clearance_pt >= 6.0, (
            f"Case {case_name}, column {col_index}: decorated vertical clearance "
            f"is cramped/negative ({clearance_pt:.2f} pt < 6.0 pt). "
            f"Upper xlabel bbox: {top_xlabel_box}, lower title bbox: {bot_title_box}."
        )

    # 5. If pypdf is available, inspect the saved PDF for positive font box clearance
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(outputs["pdf"]))
        page = reader.pages[0]
        elements: list[tuple[str, float, float, float, Any]] = []

        def visitor(
            text: str,
            cm: list[float],
            tm: list[float],
            font_dict: Any,
            font_size: float,
        ) -> None:
            if text.strip():
                x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
                y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
                elements.append((text, font_size, x, y, font_dict))

        page.extract_text(visitor_text=visitor)
        downranges = [e for e in elements if e[0] == "Downrange (m)"]
        titles = [
            e
            for e in elements
            if any(
                pattern in e[0]
                for pattern in [
                    "Straight · Lean 10°",
                    "Draw · Lean 10°",
                    "Fade · Lean 10°",
                ]
            )
        ]

        assert len(downranges) >= 3, "Expected at least 3 Downrange labels"
        assert len(titles) == 3, "Expected 3 lower-row titles"

        for col_idx in range(3):
            upper_d_y = downranges[col_idx][3]
            lower_t_y = titles[col_idx][3]
            baseline_gap = upper_d_y - lower_t_y

            # Font metric box gap calculation
            f_desc = (
                titles[col_idx][4].get_object().get("/FontDescriptor", {}).get_object()
            )
            bbox = [float(v) / 1000.0 * 10.0 for v in f_desc.get("/FontBBox")]
            font_box_gap = (upper_d_y + bbox[1]) - (lower_t_y + bbox[3])

            assert baseline_gap >= 18.0, (
                f"PDF baseline gap too small in col {col_idx}: {baseline_gap:.2f} pt < 18.0 pt"
            )
            assert font_box_gap > 0.0, (
                f"PDF font box gap is negative/overlapping in col {col_idx}: {font_box_gap:.2f} pt"
            )
    except (ImportError, OSError):
        pass
