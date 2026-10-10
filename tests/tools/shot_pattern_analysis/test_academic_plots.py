"""Contracts for normalized endpoint density and academic figures."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _density_api():
    from src.tools.shot_pattern_analysis.density import shared_histogram_density

    return shared_histogram_density


def _plot_api():
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib unavailable: {exc}")
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_academic_dispersion_figure,
        write_overhead_trajectory_figure,
    )

    return write_academic_dispersion_figure, write_overhead_trajectory_figure


def _assert_pilot_footer_is_inside_canvas(figure) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    footer = next(
        text
        for text in figure.texts
        if text.get_text().startswith(
            "Illustrative Simulation • Assumed Head Properties • Carry Only"
        )
    )
    bounds = footer.get_window_extent(renderer)
    assert bounds.x0 >= 0
    assert bounds.y0 >= 0
    assert bounds.x1 <= figure.bbox.width
    assert bounds.y1 <= figure.bbox.height


def test_shared_histogram_density_normalizes_and_reports_empirical_hdr_mass() -> None:
    estimate = _density_api()(
        {
            "Straight": np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            "Draw": np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]),
        },
        bins=8,
        target_point_m=(0.0, 0.0),
    )

    assert set(estimate.patterns) == {"Straight", "Draw"}
    cell_areas = (
        np.diff(estimate.x_edges_m)[:, None] * np.diff(estimate.y_edges_m)[None, :]
    )
    for pattern_density in estimate.patterns.values():
        assert np.sum(pattern_density.density_per_m2 * cell_areas) == pytest.approx(1)
        for requested_mass, threshold in pattern_density.hdr_density_thresholds.items():
            achieved = pattern_density.hdr_achieved_mass[requested_mass]
            assert achieved >= requested_mass
            direct_mass = np.sum(
                pattern_density.density_per_m2[
                    pattern_density.density_per_m2 >= threshold
                ]
                * cell_areas[pattern_density.density_per_m2 >= threshold]
            )
            assert achieved == pytest.approx(direct_mass)


def test_singular_and_nearly_one_dimensional_data_do_not_gain_synthetic_spread() -> (
    None
):
    shared_density = _density_api()
    same_point = np.repeat([[0.0, 0.0]], 5, axis=0)
    result = shared_density({"Straight": same_point}, bins=16)
    density = result.patterns["Straight"].density_per_m2

    assert np.isfinite(density).all()
    assert np.count_nonzero(density) == 1
    assert np.sum(density) > 0

    line = np.column_stack((np.linspace(-2, 2, 101), np.zeros(101)))
    line_result = shared_density({"Straight": line}, bins=16)
    line_density = line_result.patterns["Straight"].density_per_m2
    assert np.isfinite(line_density).all()
    assert np.count_nonzero(np.any(line_density > 0, axis=0)) == 1


def test_patterns_share_bin_edges_and_one_density_normalization() -> None:
    result = _density_api()(
        {
            "Straight": np.array([[-2.0, 0.0], [2.0, 0.0]]),
            "Fade": np.array([[0.0, -1.0], [0.0, 1.0]]),
        },
        bins=10,
    )

    assert result.density_vmax_per_m2 == max(
        float(np.max(item.density_per_m2)) for item in result.patterns.values()
    )
    assert result.x_limits_m[0] < -2 and result.x_limits_m[1] > 2
    assert result.y_limits_m[0] < -1 and result.y_limits_m[1] > 1


def test_density_rejects_invalid_or_empty_pattern_data() -> None:
    shared_density = _density_api()
    with pytest.raises(ValueError, match="at least one pattern"):
        shared_density({})
    with pytest.raises(ValueError, match="finite"):
        shared_density({"Draw": np.array([[0.0, np.nan]])})


@pytest.mark.slow
def test_academic_dispersion_figure_exports_all_formats_with_scenario_labels(
    tmp_path,
) -> None:
    write_dispersion, _ = _plot_api()
    outputs = write_dispersion(
        {
            "Straight": np.array([[-0.5, -0.2], [0.0, 0.0], [0.6, 0.2]]),
            "Draw": np.array([[-0.2, 0.5], [0.2, 0.8], [0.5, 1.0]]),
            "Fade": np.array([[0.1, -0.7], [0.3, -0.6], [0.7, -0.4]]),
        },
        tmp_path / "pilot_dispersion",
        scenario="Synthetic Layout Pilot",
        metadata={
            "face_sd_deg": 1.0,
            "shaft_lean_deg": 10.0,
            "delivery_mode": "shaft_rotation",
            "total_shots_by_pattern": {
                "Straight": 1000,
                "Draw": 1000,
                "Fade": 1000,
            },
        },
        target_point_m=(0.0, 0.0),
        bins=16,
        dpi=96,
    )

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    svg = outputs["svg"].read_text(encoding="utf-8")
    assert "Synthetic Layout Pilot" in svg
    assert "Conditional on Supported Airborne Shots" in svg
    assert "delivery mode: shaft_rotation" in svg
    assert "Target" in svg
    assert "not calibrated player precision" in svg
    assert "Illustrative Simulation • Assumed Head Properties • Carry Only" in svg
    assert "sample-derived from simulated endpoints" in svg
    assert "Straight 3/1,000 valid" in svg


@pytest.mark.slow
def test_density_metadata_caption_does_not_overlap_scenario(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    points = {
        "Straight": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]),
        "Draw": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]),
        "Fade": np.array([[0.0, -1.0], [1.0, -2.0], [2.0, 0.0]]),
    }
    metadata = {
        "club": "seven_iron",
        "nominal_shaft_lean_deg": 0.0,
        "contact_law": "incremental_coulomb_energetic",
        "club_speed_mps": 36.0,
        "static_loft_deg": 34.0,
        "curve_scale": 1.0,
        "shaft_rotation_sd_deg": 1.0,
        "path_sd_deg": 0.5,
        "speed_sd_mps": 0.89408,
        "contact_horizontal_sd_m": 0.00635,
        "contact_vertical_sd_m": 0.00635,
        "shaft_lean_sd_deg": 0.0,
        "attack_angle_sd_deg": 0.0,
    }
    scenario = "PILOT V1 — 7-Iron; lean 0°; curve scale ×1; combined delivery variation; 32 attempts/pattern"

    write_dispersion(
        points,
        tmp_path / "caption_layout",
        scenario=scenario,
        metadata=metadata,
        target_point_m=(1.0, 0.0),
        bins=8,
        dpi=96,
    )

    figure = captured["figure"]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    scenario_artist = next(text for text in figure.texts if text.get_text() == scenario)
    metadata_artist = next(
        text for text in figure.texts if text.get_text().startswith("club: seven_iron")
    )
    assert not scenario_artist.get_window_extent(renderer).overlaps(
        metadata_artist.get_window_extent(renderer)
    )
    _assert_pilot_footer_is_inside_canvas(figure)


@pytest.mark.slow
def test_overhead_figure_uses_raw_trajectories_and_exports_all_formats(
    tmp_path,
) -> None:
    _, write_overhead = _plot_api()
    outputs = write_overhead(
        {
            "Straight": [np.array([[0.0, 0.0], [100.0, 0.0], [195.0, 0.0]])],
            "Draw": [np.array([[0.0, 0.0], [100.0, 2.0], [195.0, 4.0]])],
            "Fade": [np.array([[0.0, 0.0], [100.0, -2.0], [195.0, -4.0]])],
        },
        tmp_path / "pilot_raw_flight",
        target_x_m=195.0,
        scenario="Synthetic Raw-Trajectory Pilot",
        metadata={
            "face_sd_deg": 1.0,
            "shaft_lean_deg": 0.0,
            "total_shots_by_pattern": {
                "Straight": 1000,
                "Draw": 1000,
                "Fade": 1000,
            },
            "valid_shots_by_pattern": {
                "Straight": 997,
                "Draw": 996,
                "Fade": 998,
            },
        },
        dpi=96,
    )

    assert set(outputs) == {"png", "pdf", "svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    svg = outputs["svg"].read_text(encoding="utf-8")
    assert "Synthetic Raw-Trajectory Pilot" in svg
    assert "no aim-line rotation" in svg
    assert "Straight 997/1,000 valid" in svg
    assert "Illustrative Simulation • Assumed Head Properties • Carry Only" in svg
    assert "Vertical display magnification" in svg


def test_density_overlay_is_a_disclosed_deterministic_subset(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    points = np.column_stack(
        (np.arange(1000, dtype=float), np.arange(1000, dtype=float))
    )
    write_dispersion(
        {"Straight": points},
        tmp_path / "disclosed_subset",
        scenario="Synthetic Plot Subset",
        metadata={},
        target_point_m=(500.0, 500.0),
        bins=32,
        dpi=96,
    )
    figure = captured["figure"]
    endpoint_axis = figure.axes[1]
    overlay = endpoint_axis.collections[1]
    expected = points[np.linspace(0, 999, 600, dtype=int)]
    np.testing.assert_array_equal(overlay.get_offsets(), expected)
    assert any("600/1,000" in artist.get_text() for artist in figure.texts)


def test_density_marginals_use_common_probability_scales(tmp_path, monkeypatch) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    broad = np.column_stack((np.linspace(-10, 10, 100), np.linspace(-5, 5, 100)))
    narrow = np.column_stack((np.linspace(-2, 2, 100), np.linspace(-1, 1, 100)))
    write_dispersion(
        {"Straight": broad, "Draw": narrow},
        tmp_path / "common_marginal_scale",
        scenario="Synthetic Marginal Scale",
        metadata={},
        target_point_m=(0.0, 0.0),
        bins=32,
        dpi=96,
    )
    axes = captured["figure"].axes
    assert axes[0].get_ylim() == axes[3].get_ylim()
    assert axes[2].get_xlim() == axes[5].get_xlim()


def test_overhead_discloses_actual_axis_magnification_and_legible_paths(
    tmp_path, monkeypatch
) -> None:
    _, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    write_overhead(
        {"Straight": [np.array([[0.0, 0.0], [180.0, 10.0]])]},
        tmp_path / "axis_magnification",
        target_x_m=180.0,
        scenario="Synthetic Axis Contract",
        metadata={},
        dpi=96,
    )
    figure = captured["figure"]
    figure.canvas.draw()
    axis = figure.axes[0]
    origin = axis.transData.transform((0.0, 0.0))
    x_step = axis.transData.transform((1.0, 0.0))
    y_step = axis.transData.transform((0.0, 1.0))
    expected = (y_step[1] - origin[1]) / (x_step[0] - origin[0])
    assert any(f"{expected:.1f}×" in artist.get_text() for artist in figure.texts)
    path = axis.lines[0]
    assert path.get_alpha() >= 0.6
    assert path.get_linewidth() >= 1.1


@pytest.mark.slow
def test_social_presentation_keeps_data_and_uses_readable_16_by_9_layout(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = []

    def capture_figure(figure, _stem, *, dpi):
        captured.append(figure)
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    endpoints = {
        "Straight": np.array([[170.0, -2.0], [180.0, 2.0]]),
        "Draw": np.array([[171.0, -1.0], [179.0, 1.0]]),
        "Fade": np.array([[172.0, -3.0], [178.0, 3.0]]),
    }
    social_metadata = {
        "social_club": "Driver",
        "protocol": "Original Matched-Delivery Protocol",
        "noise": "Shaft Twist SD 2° • Path SD 0.5° • Speed SD 0.9 m/s",
        "valid_shots_by_pattern": dict.fromkeys(endpoints, 2),
        "total_shots_by_pattern": dict.fromkeys(endpoints, 2),
    }
    write_dispersion(
        endpoints,
        tmp_path / "social_density",
        scenario="Lean 0° • Curve ×1",
        metadata=social_metadata,
        target_point_m=(180.0, 0.0),
        presentation="social",
        dpi=96,
    )
    write_overhead(
        {
            name: [np.array([[0.0, 0.0], endpoint[0]])]
            for name, endpoint in endpoints.items()
        },
        tmp_path / "social_overhead",
        target_x_m=180.0,
        scenario="Lean 0° • Curve ×1",
        metadata=social_metadata,
        presentation="social",
        dpi=96,
    )
    write_dispersion(
        endpoints,
        tmp_path / "academic_density_same_data",
        scenario="Synthetic Academic Control",
        metadata=social_metadata,
        target_point_m=(180.0, 0.0),
        dpi=96,
    )
    assert len(captured) == 3
    overhead_w, overhead_h = captured[1].get_size_inches()
    assert overhead_w / overhead_h == pytest.approx(16 / 9)
    density_w, density_h = captured[0].get_size_inches()
    assert density_w == pytest.approx(10.8)
    assert density_h >= density_w
    for figure in captured[:2]:
        full_text = " ".join(item.get_text() for item in figure.texts)
        assert "Lean 0° • Curve ×1" in full_text
        assert "Original Matched-Delivery Protocol" in full_text
        assert "Model-Conditional" in full_text
        assert "2/2" in full_text
    assert any(
        "Driver: Shot Dispersion" in item.get_text() for item in captured[0].texts
    )
    assert any(
        "Driver: Sample Flight Paths" in item.get_text() for item in captured[1].texts
    )
    np.testing.assert_array_equal(
        captured[0].axes[1].collections[0].get_array(),
        captured[2].axes[1].collections[0].get_array(),
    )
    assert captured[0].axes[1].get_xlim() == captured[2].axes[1].get_xlim()
    assert captured[0].axes[1].get_ylim() == captured[2].axes[1].get_ylim()


@pytest.mark.slow
def test_overhead_pilot_footer_remains_inside_figure_bounds(
    tmp_path, monkeypatch
) -> None:
    _, write_overhead = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    trajectories = {
        name: [np.array([[0.0, 0.0], [100.0, offset], [195.0, 2 * offset]])]
        for name, offset in (("Straight", 0.0), ("Draw", 1.0), ("Fade", -1.0))
    }

    write_overhead(
        trajectories,
        tmp_path / "footer_bounds",
        target_x_m=195.0,
        scenario="Synthetic Layout Pilot",
        metadata={"club": "driver", "shaft_rotation_sd_deg": 1.0},
        dpi=96,
    )

    _assert_pilot_footer_is_inside_canvas(captured["figure"])


def test_overhead_figure_rejects_aim_adjusted_or_nontrajectory_inputs(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_overhead_trajectory_figure as write_overhead,
    )

    with pytest.raises(ValueError, match="at least one trajectory"):
        write_overhead(
            {"Straight": []},
            tmp_path / "empty",
            target_x_m=195.0,
            scenario="Synthetic Pilot",
            metadata={"face_sd_deg": 1.0, "shaft_lean_deg": 0.0},
        )
    with pytest.raises(ValueError, match="raw, not aim-adjusted"):
        write_overhead(
            {"Straight": [np.array([[0.0, 0.0], [1.0, 0.0]])]},
            tmp_path / "aimed",
            target_x_m=195.0,
            scenario="Synthetic Pilot",
            metadata={},
            trajectories_are_aimed=True,
        )


def test_density_rightmost_downrange_tick_does_not_overlap_marginal_zero_tick(
    tmp_path, monkeypatch
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)
    rng = np.random.default_rng(0)
    points = {
        "Straight": rng.normal(loc=[180.0, 0.0], scale=[15.0, 8.0], size=(100, 2)),
        "Draw": rng.normal(loc=[185.0, 5.0], scale=[12.0, 6.0], size=(100, 2)),
        "Fade": rng.normal(loc=[175.0, -5.0], scale=[14.0, 7.0], size=(100, 2)),
    }
    write_dispersion(
        points,
        tmp_path / "density_tick_check",
        scenario="Tick Overlap Regression Check",
        metadata={
            "valid_shots_by_pattern": {"Straight": 100, "Draw": 100, "Fade": 100},
            "total_shots_by_pattern": {"Straight": 100, "Draw": 100, "Fade": 100},
        },
        target_point_m=(180.0, 0.0),
        bins=16,
        dpi=96,
    )
    figure = captured["figure"]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for panel_index in range(3):
        ax_main = figure.axes[1 + panel_index * 3]
        ax_right = figure.axes[2 + panel_index * 3]
        main_ticks = [
            t for t in ax_main.get_xticklabels() if t.get_text() and t.get_visible()
        ]
        right_ticks = [
            t for t in ax_right.get_xticklabels() if t.get_text() and t.get_visible()
        ]
        assert main_ticks, f"Panel {panel_index} main axis has no visible x-ticks"
        assert right_ticks, f"Panel {panel_index} right marginal has no visible x-ticks"
        bb_main_last = main_ticks[-1].get_window_extent(renderer)
        for r_tick in right_ticks:
            bb_r = r_tick.get_window_extent(renderer)
            assert not bb_main_last.overlaps(bb_r), (
                f"Panel {panel_index} downrange tick '{main_ticks[-1].get_text()}' "
                f"overlaps right marginal tick '{r_tick.get_text()}': "
                f"main_bbox={bb_main_last}, right_bbox={bb_r}"
            )
            assert bb_r.x0 >= bb_main_last.x1, (
                f"Panel {panel_index} right marginal tick '{r_tick.get_text()}' "
                f"is not to the right of downrange tick '{main_ticks[-1].get_text()}': "
                f"gap={bb_r.x0 - bb_main_last.x1}"
            )


@pytest.mark.parametrize("n_panels", [2, 3])
def test_social_density_phone_preview_geometry_and_typography(
    tmp_path, monkeypatch, n_panels: int
) -> None:
    write_dispersion, _ = _plot_api()
    import src.tools.shot_pattern_analysis.academic_plots as plots

    captured = {}

    def capture_figure(figure, _stem, *, dpi):
        captured["figure"] = figure
        return {}

    monkeypatch.setattr(plots, "_save_formats", capture_figure)

    rng = np.random.default_rng(42)
    if n_panels == 2:
        names = ["SD 1°", "SD 2°"]
        points = {
            "SD 1°": rng.normal(loc=[180.0, 0.0], scale=[10.0, 5.0], size=(120, 2)),
            "SD 2°": rng.normal(loc=[180.0, 0.0], scale=[16.0, 8.0], size=(120, 2)),
        }
        metadata = {
            "social_club": "Driver",
            "Delivery Noise": (
                "Shaft Twist SD 2° • Path SD 0.5° • Speed SD 0.9 m/s • "
                "Horizontal Strike SD 0.25 in • Vertical Strike SD 0.5 in • "
                "Shaft Twist–Path Correlation r = +0.35"
            ),
            "Population": "Same Shot Indices Supported in All Six Populations",
            "Scales": "Shared Within This Figure; Other Pattern Figures May Differ",
            "valid_shots_by_pattern": {"SD 1°": 120, "SD 2°": 120},
            "total_shots_by_pattern": {"SD 1°": 150, "SD 2°": 150},
        }
        scenario = (
            "Draw: Lean 5° • Curve ×1 • Combined Delivery Variation with Correlation"
        )
    else:
        names = ["Straight", "Draw", "Fade"]
        points = {
            "Straight": rng.normal(loc=[180.0, 0.0], scale=[12.0, 6.0], size=(120, 2)),
            "Draw": rng.normal(loc=[185.0, 6.0], scale=[10.0, 5.0], size=(120, 2)),
            "Fade": rng.normal(loc=[175.0, -6.0], scale=[14.0, 7.0], size=(120, 2)),
        }
        metadata = {
            "social_club": "Driver",
            "Delivery Noise": (
                "Shaft Twist SD 1° • Path SD 0.5° • Speed SD 0.9 m/s • "
                "Horizontal Strike SD 0.25 in • Vertical Strike SD 0.5 in"
            ),
            "Population": "One Cell Supported Populations",
            "valid_shots_by_pattern": {"Straight": 120, "Draw": 120, "Fade": 120},
            "total_shots_by_pattern": {"Straight": 150, "Draw": 150, "Fade": 150},
        }
        scenario = "Driver Lean 5° • Curve ×1 • Combined Delivery Variation"

    write_dispersion(
        points,
        tmp_path / f"social_density_{n_panels}p",
        scenario=scenario,
        metadata=metadata,
        target_point_m=(180.0, 0.0),
        presentation="social",
        dpi=96,
    )
    figure = captured["figure"]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    scale_1080 = 1080.0 / figure.bbox.width

    visible_texts = [t for t in figure.texts if t.get_text().strip()]
    for i, t1 in enumerate(visible_texts):
        bb1 = t1.get_window_extent(renderer)
        for t2 in visible_texts[i + 1 :]:
            bb2 = t2.get_window_extent(renderer)
            assert not bb1.overlaps(bb2), (
                f"Figure texts overlap: {t1.get_text()[:30]!r} and {t2.get_text()[:30]!r}"
            )

    if figure.legends:
        legend = figure.legends[0]
        bb_leg = legend.get_window_extent(renderer)
        for t in visible_texts:
            bb = t.get_window_extent(renderer)
            assert not bb.overlaps(bb_leg), (
                f"Text overlaps legend: {t.get_text()[:30]!r}"
            )

    for ax in figure.axes:
        ax_bb = ax.get_window_extent(renderer)
        for t in visible_texts:
            bb = t.get_window_extent(renderer)
            assert not bb.overlaps(ax_bb), (
                f"Figure text overlaps axis: {t.get_text()[:30]!r} vs {ax}"
            )

    for t in visible_texts:
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.y0 >= 0, (
            f"Text extends outside canvas (left/bottom): {t.get_text()[:30]!r}"
        )
        assert bb.x1 <= figure.bbox.width and bb.y1 <= figure.bbox.height, (
            f"Text extends outside canvas (right/top): {t.get_text()[:30]!r}"
        )

    for t in visible_texts:
        h_px = t.get_window_extent(renderer).height * scale_1080
        assert h_px >= 11.0, (
            f"Figure text too small at 1080px preview: {t.get_text()[:30]!r} height={h_px:.1f}px < 11.0px"
        )

    for ax in figure.axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            if tick.get_text() and tick.get_visible():
                h_px = tick.get_window_extent(renderer).height * scale_1080
                assert h_px >= 11.0, (
                    f"Tick label '{tick.get_text()}' too small at 1080px preview: height={h_px:.1f}px < 11.0px"
                )
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if label.get_text() and label.get_visible():
                h_px = label.get_window_extent(renderer).height * scale_1080
                assert h_px >= 11.0, (
                    f"Axis label/title '{label.get_text()}' too small at 1080px preview: height={h_px:.1f}px < 11.0px"
                )

    full_text = " ".join(t.get_text() for t in visible_texts)
    normalized_text = " ".join(full_text.split())
    assert scenario in normalized_text
    assert "Model-Conditional" in full_text
    assert "120/150" in full_text
    for name in names:
        assert name in full_text or any(name in ax.get_title() for ax in figure.axes)
