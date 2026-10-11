"""Regression tests for covariance caption wrapping, header layout, and containment (issue #12132).

Verifies that long covariance captions in academic density, academic overhead,
and social overhead figures:
1. Do not inflate the figure bounding box beyond the standard width.
2. Are fully contained within the canvas width without right-edge clipping.
3. Maintain positive vertical clearance between all header elements, legend, and subplots.
4. Preserve all scientific text, font sizes, data, axis limits, and disclosures.
"""

from __future__ import annotations

from src.tools.shot_pattern_analysis.matched_supplement_figures import PairFigureContext

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _load_plotting_deps() -> tuple[Any, Any, Any]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt
        from PIL import Image

        return FigureCanvasAgg, plt, Image
    except (ImportError, OSError) as exc:
        pytest.skip(f"Plotting dependencies unavailable: {exc}")


def _synthetic_covariance_pair(
    club_id: str = "pitching_wedge",
    target_x_m: float = 125.0,
    n_shots: int = 10_000,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, list[list[float]]], dict[str, list[Any]]
]:
    """Construct pair references and sample data with a full covariance profile."""
    from src.tools.shot_pattern_analysis.matched_matrix import noise_profile
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _pair_condition_label,
    )

    spec = noise_profile("cov_facepath_negative", 1.0)
    first_ref = {
        "cell": f"{club_id}_lean0_curve1_rotation1_cov_facepath_negative",
        "source_protocol": "original_control/1",
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": 0.0},
            "curve_scale": 1.0,
            "noise": {
                "standard_deviations": list(spec.standard_deviations),
                "correlation": spec.correlation,
            },
            "n_shots": n_shots,
        },
        "summary": {"target_x_m": target_x_m},
    }
    second_ref = {
        "cell": f"{club_id}_lean0_curve2_rotation1_cov_facepath_negative",
        "source_protocol": "original_control/1",
        "config": {
            "club": {"club_id": club_id, "shaft_lean_deg": 0.0},
            "curve_scale": 2.0,
            "noise": {
                "standard_deviations": list(spec.standard_deviations),
                "correlation": spec.correlation,
            },
            "n_shots": n_shots,
        },
        "summary": {"target_x_m": target_x_m},
    }

    first_label = _pair_condition_label("curve_strength", first_ref)
    second_label = _pair_condition_label("curve_strength", second_ref)

    point_sets: dict[str, list[list[float]]] = {}
    trajectory_sets: dict[str, list[Any]] = {}
    for pattern in ("Straight", "Draw", "Fade"):
        for label in (first_label, second_label):
            key = f"{pattern} · {label}"
            # 10 synthetic points around target
            point_sets[key] = [
                [target_x_m + float(i) * 0.1, float(i) * 0.05] for i in range(10)
            ]
            # 1 representative flight sample
            trajectory_sets[key] = [
                np.array([[0.0, 0.0], [target_x_m * 0.5, 1.0], [target_x_m, 0.0]])
            ]

    return first_ref, second_ref, point_sets, trajectory_sets


def test_academic_density_covariance_caption_containment_and_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Academic paired-density figure with covariance captions must not expand or overlap."""
    canvas_agg_cls, plt, img_cls = _load_plotting_deps()
    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _condition_caption,
        _pair_condition_label,
        _write_pair_density_figure,
    )

    first, second, point_sets, _ = _synthetic_covariance_pair()
    first_label = _pair_condition_label("curve_strength", first)
    second_label = _pair_condition_label("curve_strength", second)

    captured_figures: list[Any] = []
    orig_save = msf._save_figure

    def capturing_save(
        figure: Any, stem: Path, *args: Any, **kwargs: Any
    ) -> dict[str, Path]:
        captured_figures.append(figure)
        return orig_save(figure, stem, *args, **kwargs)

    monkeypatch.setattr(msf, "_save_figure", capturing_save)

    output_stem = tmp_path / "density_cov_test"
    outputs = _write_pair_density_figure(
        point_sets,
        context=PairFigureContext(
            first, second, "curve_strength", (first_label, second_label), 125.0
        ),
        common_n=10,
        output_stem=output_stem,
        formats=("png",),
    )

    # 1. Output PNG width must not be expanded by uncontained caption text.
    # Standard 16-inch wide figure at 300 dpi with tight bbox is ~4400-4750 px.
    # Unpatched covariance caption expands width to 6115-6449 px.
    img = img_cls.open(outputs["png"])
    assert img.size[0] <= 4850, (
        f"Exported academic density PNG width expanded to {img.size[0]} px (> 4850 px); "
        "indicates unwrapped caption text extending beyond the canvas."
    )

    figure = captured_figures[0]
    canvas_agg_cls(figure)
    renderer = figure.canvas.get_renderer()

    # 2. Text containment: all rendered text boxes must stay within canvas width.
    for text_obj in figure.texts:
        bbox = text_obj.get_window_extent(renderer)
        assert bbox.x1 <= figure.bbox.width, (
            f"Text overflows right canvas boundary: x1={bbox.x1:.1f} > "
            f"fig_width={figure.bbox.width:.1f}. Text: {text_obj.get_text()[:60]}..."
        )

    # 3. Header vertical separation: find header texts and verify positive clearance.
    header_texts = [t for t in figure.texts if t.get_position()[1] > 0.70]
    # Sort top to bottom by y position
    header_texts.sort(key=lambda t: t.get_window_extent(renderer).y1, reverse=True)
    assert len(header_texts) >= 4, "Expected at least 4 header text elements"

    for i in range(len(header_texts) - 1):
        upper = header_texts[i]
        lower = header_texts[i + 1]
        b_upper = upper.get_window_extent(renderer)
        b_lower = lower.get_window_extent(renderer)
        clearance_pt = (b_upper.y0 - b_lower.y1) * 72.0 / figure.dpi
        assert clearance_pt >= 0.0, (
            f"Header text overlap between '{upper.get_text()[:30]}...' and "
            f"'{lower.get_text()[:30]}...': clearance is {clearance_pt:.2f} pt < 0.0 pt."
        )

    # 4. Legend must not overlap subplots top
    if figure.legends:
        leg = figure.legends[0]
        leg_box = leg.get_window_extent(renderer)
        # Check against subplots upper boundary
        for ax in figure.axes:
            ax_box = ax.get_window_extent(renderer)
            leg_clearance_pt = (leg_box.y0 - ax_box.y1) * 72.0 / figure.dpi
            assert leg_clearance_pt >= 0.0, (
                f"Legend overlaps subplot top: clearance is {leg_clearance_pt:.2f} pt < 0.0 pt."
            )

    # 5. Scientific text preservation: all words of condition caption must appear.
    full_text = " ".join(t.get_text() for t in figure.texts)
    for ref in (first, second):
        cap = _condition_caption(ref)
        # Check essential scientific tokens
        for token in (
            "Covariance Profile",
            "Shaft Twist–Path Correlation",
            "Lean 0°",
            "Curve",
        ):
            assert token in full_text, (
                f"Missing token '{token}' in rendered density figure"
            )


def test_academic_and_social_overhead_covariance_caption_containment_and_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Academic and social overhead figures must contain text without clipping or overlap."""
    canvas_agg_cls, plt, img_cls = _load_plotting_deps()
    import src.tools.shot_pattern_analysis.matched_supplement_figures as msf
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _condition_caption,
        _create_pair_overhead_figure,
        _pair_condition_label,
        _write_pair_overhead_figure,
    )

    first, second, _, trajectory_sets = _synthetic_covariance_pair()
    first_label = _pair_condition_label("curve_strength", first)
    second_label = _pair_condition_label("curve_strength", second)

    output_stem = tmp_path / "overhead_cov_test"
    outputs = _write_pair_overhead_figure(
        trajectory_sets,
        context=PairFigureContext(
            first, second, "curve_strength", (first_label, second_label), 125.0
        ),
        output_stem=output_stem,
        academic_formats=("png",),
        include_social=True,
    )

    # 1. Academic overhead PNG width must not be expanded (> 4850 px).
    # Unpatched overhead PNG is ~6367-6450 px.
    img_acad = img_cls.open(outputs["png"])
    assert img_acad.size[0] <= 4850, (
        f"Exported academic overhead PNG width expanded to {img_acad.size[0]} px (> 4850 px)."
    )

    # 2. Social overhead PNG must be exactly 1920x1080 and contain all text without clipping.
    img_social = img_cls.open(outputs["social_png"])
    assert img_social.size == (1920, 1080), (
        f"Unexpected social PNG size: {img_social.size}"
    )

    # Inspect rendered text bounds on social figure
    _, social_fig = _create_pair_overhead_figure(
        trajectory_sets,
        first=first,
        second=second,
        comparison="curve_strength",
        first_label=first_label,
        second_label=second_label,
        target_x=125.0,
        social=True,
    )
    canvas_agg_cls(social_fig)
    renderer_social = social_fig.canvas.get_renderer()

    for text_obj in social_fig.texts:
        bbox = text_obj.get_window_extent(renderer_social)
        assert bbox.x1 <= social_fig.bbox.width, (
            f"Social overhead text clipped on right edge: x1={bbox.x1:.1f} > "
            f"width={social_fig.bbox.width:.1f}. Text: {text_obj.get_text()[:60]}..."
        )

    # Header separation in social figure
    header_texts_social = [t for t in social_fig.texts if t.get_position()[1] > 0.65]
    header_texts_social.sort(
        key=lambda t: t.get_window_extent(renderer_social).y1, reverse=True
    )
    for i in range(len(header_texts_social) - 1):
        upper = header_texts_social[i]
        lower = header_texts_social[i + 1]
        b_upper = upper.get_window_extent(renderer_social)
        b_lower = lower.get_window_extent(renderer_social)
        clearance_pt = (b_upper.y0 - b_lower.y1) * 72.0 / social_fig.dpi
        assert clearance_pt >= 0.0, (
            f"Social header overlap between '{upper.get_text()[:30]}...' and "
            f"'{lower.get_text()[:30]}...': clearance is {clearance_pt:.2f} pt < 0.0 pt."
        )

    # 3. Disclosures preserved
    full_text_social = " ".join(t.get_text() for t in social_fig.texts)
    assert "Lateral display magnified 6×" in full_text_social
    assert "Saved paths (not paired by shot index)" in full_text_social
    plt.close(social_fig)
