"""Shared header placement preserves scientific captions and physical spacing."""

import pytest

pytestmark = pytest.mark.unit


def test_header_stack_preserves_wrapping_font_and_point_gaps():
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except (ImportError, OSError) as exc:
        pytest.skip(f"Matplotlib not loadable: {exc}")
    from src.tools.shot_pattern_analysis.matched_supplement_figures import _HeaderStack

    figure = Figure(figsize=(16, 9), dpi=100)
    renderer = FigureCanvasAgg(figure).get_renderer()
    header = _HeaderStack(figure, renderer, 0.90)
    first = header.add(
        "Scientific condition with a long_unbroken_protocol",
        8.5,
        gap_pt=5.0,
        wrap_chars=24,
    )
    assert first.get_text() == "Scientific condition\nwith a\nlong_unbroken_protocol"
    assert first.get_fontsize() == 8.5
    assert first.get_position()[0] == 0.06
    assert (0.90 - first.get_position()[1]) * figure.bbox.height == pytest.approx(
        5 / 72 * 100
    )
    first_bottom = first.get_window_extent(renderer).y0
    second = header.add("Carry Only", 9.0, gap_pt=3.5, wrap_chars=180)
    assert first_bottom - second.get_position()[
        1
    ] * figure.bbox.height == pytest.approx(3.5 / 72 * 100)
    assert header.y_cursor == second.get_window_extent(renderer).y0 / figure.bbox.height
    assert first.get_ha() == second.get_ha() == "left"
    assert first.get_va() == second.get_va() == "top"


def test_noise_labels_follow_canonical_field_order():
    from src.tools.shot_pattern_analysis.noise import NOISE_FIELDS, NOISE_DISPLAY_LABELS

    assert dict(zip(NOISE_FIELDS, NOISE_DISPLAY_LABELS, strict=True)) == {
        "shaft_rotation_deg": "Shaft Twist",
        "path_deg": "Path",
        "speed_mps": "Speed",
        "contact_horizontal_m": "Horizontal Strike",
        "contact_vertical_m": "Vertical Strike",
        "shaft_lean_deg": "Shaft Lean",
        "attack_angle_deg": "Attack Angle",
    }


@pytest.mark.parametrize("rho", [5e-7, 1e-6, 2e-6, -2e-6])
def test_distinct_caption_correlation_thresholds_are_preserved(rho):
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        _derive_delivery_noise,
    )
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        _noise_profile_description,
    )

    correlation = [[float(i == j) for j in range(7)] for i in range(7)]
    correlation[0][1] = correlation[1][0] = rho
    noise = {"standard_deviations": [1, 0.5, 0, 0, 0, 0, 0], "correlation": correlation}
    matrix = _derive_delivery_noise([{"config": {"noise": noise}}])
    pair = _noise_profile_description("shaft_path", noise)
    correlation_text = f"Shaft Twist–Path Correlation r = {rho:+g}"
    assert correlation_text in pair
    assert (correlation_text in matrix) == (abs(rho) > 1e-6)
    assert "Shaft Twist SD 1°" in matrix
    assert "Shaft Twist SD" not in pair


def test_report_provenance_binds_noise_display_labels():
    import hashlib
    from pathlib import Path

    from src.tools.shot_pattern_analysis import noise
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        _report_source_hashes,
    )

    expected = hashlib.sha256(Path(noise.__file__).read_bytes()).hexdigest()
    assert _report_source_hashes()["noise.py"] == expected
