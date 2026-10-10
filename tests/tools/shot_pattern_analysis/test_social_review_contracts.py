"""Independent publication regressions; root-owned while AGY finishes integration."""

import numpy as np
import pytest
from src.tools.shot_pattern_analysis import matched_supplement_figures as figures
from src.tools.shot_pattern_analysis import matched_supplement_pipeline as pipeline


pytestmark = pytest.mark.unit


def noise(correlation=0.0, horizontal=0.00635, vertical=0.0127):
    matrix = np.eye(7)
    matrix[0, 1] = matrix[1, 0] = correlation
    return {
        "standard_deviations": [1, 0.5, 0.89408, horizontal, vertical, 0, 0],
        "correlation": matrix.tolist(),
    }


def test_serialized_zero_correlation_overrides_profile_name():
    label = figures._noise_profile_description("cov_facepath_positive", noise())
    assert "Correlation r" not in label


def test_nonzero_correlation_names_primitive_shaft_twist():
    label = figures._noise_profile_description("combined2", noise(0.35))
    assert "Shaft Twist–Path Correlation r = +0.35" in label
    assert "Face-Path Correlation" not in label


def test_unequal_strike_standard_deviations_are_not_called_equal():
    label = figures._noise_profile_description("combined2", noise())
    assert "Each Axis" not in label
    assert "Horizontal Strike SD 0.25 in" in label
    assert "Vertical Strike SD 0.5 in" in label


def test_available_figure_without_path_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        pipeline._verify_figure_assets(
            tmp_path, {"figures": [{"figure_kind": "density_png", "sha256": "a" * 64}]}
        )


def test_unavailable_figure_cannot_claim_an_unverified_artifact(tmp_path):
    with pytest.raises(ValueError):
        pipeline._verify_figure_assets(
            tmp_path,
            {
                "figures": [
                    {
                        "status": "unavailable",
                        "reason": "low support",
                        "relative_path": "missing.png",
                        "sha256": "a" * 64,
                    }
                ]
            },
        )
