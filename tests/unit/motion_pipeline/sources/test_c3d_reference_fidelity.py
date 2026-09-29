"""C3D imports must not invent visible markers or silently collapse labels."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_pipeline.sources import c3d_adapter as module

pytestmark = pytest.mark.unit


def fixture(labels: list[str]) -> dict:
    points = np.zeros((4, len(labels), 2))
    points[0, :, :] = 1000
    residuals = np.zeros((1, len(labels), 2))
    residuals[0, -1, :] = -1  # Finite coordinates with an invalid residual.
    return {
        "parameters": {
            "POINT": {
                "LABELS": {"value": labels},
                "RATE": {"value": [100]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "data": {"points": points, "meta_points": {"residuals": residuals}},
    }


def test_ezc3d_retains_fully_missing_labels_and_masks_invalid_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module, "_ezc3d", SimpleNamespace(c3d=lambda _: fixture(["hip", "hand"]))
    )
    result = module.C3DAdapter()._load_via_ezc3d(tmp_path / "sample.c3d", None)
    assert result.metadata["source_labels"] == ["hip", "hand"]
    assert result.metadata["units_declared"] is True
    assert result.frames[0].markers["hip"].x == 1
    assert "hand" not in result.frames[0].markers


def test_duplicate_c3d_labels_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module, "_ezc3d", SimpleNamespace(c3d=lambda _: fixture(["hip", "hip"]))
    )
    with pytest.raises(ValueError, match="labels"):
        module.C3DAdapter()._load_via_ezc3d(tmp_path / "sample.c3d", None)
