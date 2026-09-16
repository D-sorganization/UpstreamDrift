"""Unit tests for pipeline address solving, closure fitting, and posture summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_scaled_offsets_scales_femur_and_tibia() -> None:
    from src.shared.python.motion_matching.pipeline.address import scaled_offsets

    offsets = {
        "RKneeOut": ("femur_r", (0.0, -0.4, 0.06)),
        "RAnkleOut": ("tibia_r", (-0.01, -0.44, 0.055)),
        "RToeIn": ("calcn_r", (0.19, 0.03, -0.03)),
    }
    scaled = scaled_offsets(offsets, femur=1.1, tibia=1.2)

    # Femur should be scaled by 1.1
    np.testing.assert_allclose(scaled["RKneeOut"][1], (0.0, -0.44, 0.066), rtol=1e-5)
    # Tibia should be scaled by 1.2
    np.testing.assert_allclose(
        scaled["RAnkleOut"][1], (-0.012, -0.528, 0.066), rtol=1e-5
    )
    # Calcaneus unchanged (scale 1.0)
    np.testing.assert_allclose(scaled["RToeIn"][1], (0.19, 0.03, -0.03), rtol=1e-5)


def test_scaled_offsets_validates_inputs() -> None:
    from src.shared.python.motion_matching.pipeline.address import scaled_offsets

    with pytest.raises(ValueError, match="positive scale factors"):
        scaled_offsets({}, femur=-1.0, tibia=1.0)
    with pytest.raises(ValueError, match="positive scale factors"):
        scaled_offsets({}, femur=1.0, tibia=0.0)
