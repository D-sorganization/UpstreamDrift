"""Portable contract checks for diagnostic-only mixed marker scores."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_marker_observation import (
    NativeMixedMarkerObservation,
    NativeMixedMarkerScore,
)
from src.shared.python.motion_matching.replay_metrics import NativeMarkerPositionOutput

pytestmark = pytest.mark.unit


def _observation() -> NativeMixedMarkerObservation:
    output = NativeMarkerPositionOutput(
        time_s=np.array([0.0, 0.01]),
        positions_m=np.zeros((2, 1, 3)),
        marker_labels=("marker",),
        frame_id="opensim-ground",
        timebase_id="simulation_relative",
    )
    return NativeMixedMarkerObservation(
        native_output=output,
        replay_identity_sha256="0" * 64,
        marker_binding_sha256="1" * 64,
        state_output_sha256="2" * 64,
        executed_loaded_model_sha256="3" * 64,
    )


def test_mixed_marker_diagnostic_qualification_is_not_replaceable() -> None:
    observation = _observation()
    score = NativeMixedMarkerScore(
        observation=observation,
        alignment=None,  # type: ignore[arg-type]
        metrics=None,  # type: ignore[arg-type]
        receipt_sha256="4" * 64,
        scorer_provider_sha256="5" * 64,
    )

    assert observation.qualification == "unqualified"
    assert score.qualification == "unqualified"
    with pytest.raises(TypeError, match="init=False"):
        replace(observation, qualification="qualified")
    with pytest.raises(TypeError, match="init=False"):
        replace(score, qualification="qualified")


def test_mixed_marker_diagnostic_qualification_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="qualification"):
        NativeMixedMarkerObservation(
            native_output=None,  # type: ignore[arg-type]
            replay_identity_sha256="0" * 64,
            marker_binding_sha256="1" * 64,
            state_output_sha256="2" * 64,
            executed_loaded_model_sha256="3" * 64,
            qualification="qualified",  # type: ignore[call-arg]
        )
