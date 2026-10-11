"""Annotation text of the per-engine address stills (OSV-4, #11730)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import render_address_engine_stills as stills

pytestmark = [pytest.mark.unit]

DOC = {
    "targets_deg": {"left": 16.4, "right": 4.0},
    "engines": {
        "drake": {
            "available": True,
            "model_deg": {"left": 16.6, "right": 3.9},
            "error_deg": {"left": 0.2, "right": -0.1},
        },
        "myosuite": {"available": False, "reason": "no viewer"},
    },
}


def test_annotation_names_both_feet_model_and_capture() -> None:
    lines = stills.annotation_lines("drake", "driver", DOC)
    assert "drake / driver" in lines[0]
    assert "lead (left): model +16.6  capture +16.4  error +0.2" in lines[1]
    assert "trail (right): model +3.9  capture +4.0  error -0.1" in lines[2]


@pytest.mark.parametrize("engine", ["myosuite", "opensim"])
def test_unavailable_or_missing_engine_is_rejected_not_zeroed(engine: str) -> None:
    with pytest.raises(ValueError, match=engine):
        stills.annotation_lines(engine, "driver", DOC)


def test_annotate_keeps_shape_and_changes_pixels() -> None:
    image = np.zeros((60, 200, 3), dtype=np.uint8)
    out = stills.annotate(image, ["abc"])
    assert out.shape == image.shape
    assert out.sum() > 0
