"""Source-clock/missingness contracts for the bounded pelvis diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics.native_pelvis_geometry import sample_observations
from scripts.diagnostics.frozen_capture_trc import export_capture
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)

pytestmark = pytest.mark.unit


def _capture() -> TourCapture:
    return TourCapture(
        np.arange(6) / 360,
        MARKER_SEGMENTS["pelvis"],
        np.full((6, 4, 3), 0.123456789),
        np.ones((6, 4), dtype=bool),
        "a" * 64,
    )


def test_probe_retains_original_sample_times_without_frame_retiming() -> None:
    source = _capture()
    selected, indices = sample_observations(source)
    np.testing.assert_array_equal(indices, [0, 2, 5])
    np.testing.assert_array_equal(selected.time_s, source.time_s[indices])
    assert selected.source_sha256 == source.source_sha256


def test_probe_does_not_synthesize_missing_pelvis_observations() -> None:
    from dataclasses import replace

    source = _capture()
    valid = source.valid.copy()
    valid[2, 0] = False
    with pytest.raises(ValueError, match="observed"):
        sample_observations(replace(source, valid=valid))


def test_existing_trc_export_records_rounding_and_preserves_missingness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.diagnostics import frozen_capture_trc

    source = _capture()
    monkeypatch.setattr(
        frozen_capture_trc.tour_capture_contract,
        "load_tour_capture",
        lambda path: source,
    )
    receipt = export_capture(tmp_path / "source.c3d", tmp_path / "export.trc")
    assert receipt["capture_sha256"] == source.source_sha256
    assert 0 <= receipt["max_position_rounding_m"] <= 5.1e-7
    assert 0 <= receipt["max_clock_rounding_s"] <= 5.1e-10
