"""Learned model geometry must not reverse segments to follow a capture."""

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import (
    ArticulatedModel,
    FitOptions,
    Joint,
    ModelSpec,
    fit_trajectory,
)

pytestmark = pytest.mark.unit


def test_fitted_segment_length_cannot_become_negative() -> None:
    model = ArticulatedModel(
        ModelSpec(
            "fixed-direction",
            (
                Joint("root", None, axes=""),
                Joint(
                    "tip", "root", direction=(1.0, 0.0, 0.0), length="segment", axes=""
                ),
            ),
            {"segment": 0.5},
        )
    )
    # Incompatible observations would otherwise make a negative length the
    # easiest solution. Preserve physical geometry and expose the residual.
    observed = np.tile([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], (3, 1, 1))
    result = fit_trajectory(
        model,
        observed,
        30,
        options=FitOptions(
            fit_lengths=("segment",), sigma_length_m=100, gate=1000, huber_delta=1000
        ),
    )
    assert result.lengths_m["segment"] > 0
    assert result.rms_m > 0.4
    assert np.isfinite(result.q).all()
