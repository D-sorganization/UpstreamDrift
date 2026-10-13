"""Frozen reproduction calls retain the public figure-writer boundary."""

from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import academic_plots, matched_pair_social

pytestmark = pytest.mark.unit


def test_pair_social_writer_accepts_frozen_keyword_context(
    monkeypatch, tmp_path: Path
) -> None:
    observed = []

    def capture_context(context: matched_pair_social.PairSocialContext) -> None:
        observed.append(context)
        raise RuntimeError("context reached")

    monkeypatch.setattr(
        matched_pair_social, "_validate_context_labels", capture_context
    )
    with pytest.raises(RuntimeError, match="context reached"):
        matched_pair_social.write_pair_social_density(
            {},
            tmp_path / "pair",
            condition_labels=("Shaft Twist SD 1°", "Shaft Twist SD 2°"),
            scenario="Fixed Scenario",
            social_club="Driver",
            target_point_m=(180.0, 0.0),
            total_n=10_000,
        )
    assert observed == [
        matched_pair_social.PairSocialContext(
            condition_labels=("Shaft Twist SD 1°", "Shaft Twist SD 2°"),
            scenario="Fixed Scenario",
            social_club="Driver",
            target_point_m=(180.0, 0.0),
            total_n=10_000,
        )
    ]


def test_academic_writer_accepts_frozen_presentation_and_dpi(
    monkeypatch, tmp_path: Path
) -> None:
    observed = []

    def capture_options(scenario: str, dpi: int, presentation: str) -> None:
        observed.append((scenario, dpi, presentation))
        raise RuntimeError("options reached")

    monkeypatch.setattr(academic_plots, "_validate_figure_options", capture_options)
    with pytest.raises(RuntimeError, match="options reached"):
        academic_plots.write_academic_dispersion_figure(
            {},
            tmp_path / "density",
            scenario="Fixed Scenario",
            metadata={"social_club": "Driver"},
            target_point_m=(180.0, 0.0),
            presentation="social",
            dpi=180,
        )
    assert observed == [("Fixed Scenario", 180, "social")]


def test_legacy_figure_options_preserve_all_previous_controls() -> None:
    defaults = academic_plots._resolve_figure_options(None, {})
    assert defaults == academic_plots.FigureOptions()
    custom = academic_plots._resolve_figure_options(
        None,
        {
            "bins": (32, 24),
            "region_masses": (0.40, 0.75),
            "presentation": "social",
            "dpi": 180,
        },
    )
    assert custom == academic_plots.FigureOptions(
        bins=(32, 24),
        region_masses=(0.40, 0.75),
        presentation="social",
        dpi=180,
    )
    assert academic_plots._resolve_figure_options(custom, {}) is custom
    with pytest.raises(ValueError, match="either options or legacy"):
        academic_plots._resolve_figure_options(custom, {"dpi": 180})
    with pytest.raises(ValueError, match="unknown figure option"):
        academic_plots._resolve_figure_options(None, {"not_an_option": 1})


def test_pair_context_preserves_optional_fields_and_rejects_mixed_inputs() -> None:
    legacy = {
        "condition_labels": ("Control", "Treatment"),
        "scenario": "Fixed Scenario",
        "social_club": "Driver",
        "target_point_m": (180.0, 0.0),
        "total_n": 10_000,
        "delivery_noise": "Declared Noise",
        "source_protocols": ("A", "B"),
        "target_modes": ("Fixed", "Fixed"),
    }
    context = matched_pair_social._resolve_pair_context(None, legacy)
    assert context == matched_pair_social.PairSocialContext(**legacy)
    assert matched_pair_social._resolve_pair_context(context, {}) is context
    with pytest.raises(ValueError, match="either context or legacy"):
        matched_pair_social._resolve_pair_context(context, legacy)
    with pytest.raises(ValueError, match="unknown pair context"):
        matched_pair_social._resolve_pair_context(None, {"not_a_field": 1})
