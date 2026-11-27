"""Reproducible, source-bound back-squat trajectory metrics (#11745)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.report_back_squat_trajectory_metrics import extract_metrics

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_BUNDLE = _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_bundle.npz"


def test_committed_optimizer_and_ik_paths_have_reproducible_literature_metrics() -> (
    None
):
    report = extract_metrics(_BUNDLE)

    assert report["source_commit"] == "c7eed64da2a3440637f20e9c28a079bfb91b911c"
    assert report["tools_pin"] == "160c9c785e4559067e2b532f189227f662f50579"
    assert report["sampling"]["samples"] == 151
    assert report["sampling"]["dt_s"] == pytest.approx(0.02)

    optimizer = report["paths"]["optimizer_planar"]["metrics"]
    assert optimizer["vertical_excursion_m"] == pytest.approx(0.7285199048, abs=1e-9)
    assert optimizer["bottom_time_s"] == pytest.approx(1.5)
    assert optimizer["descent_s"] == pytest.approx(1.5)
    assert optimizer["ascent_s"] == pytest.approx(1.5)
    assert optimizer["forward_range_m"] == pytest.approx(0.0934071094, abs=1e-9)
    assert optimizer["sagittal_path_length_m"] == pytest.approx(1.4813071628, abs=1e-9)
    assert optimizer["peak_down_vertical_speed_m_s"] == pytest.approx(
        -1.0490305492, abs=1e-9
    )
    assert optimizer["peak_down_vertical_speed_time_s"] == pytest.approx(1.34)
    assert optimizer["peak_up_vertical_speed_m_s"] == pytest.approx(
        1.0490305492, abs=1e-9
    )
    assert optimizer["peak_up_vertical_speed_time_s"] == pytest.approx(1.66)

    retargeted = report["paths"]["retargeted_high_bar_ik"]["metrics"]
    assert retargeted["vertical_excursion_m"] == pytest.approx(0.5102013875, abs=1e-9)
    assert retargeted["forward_range_m"] == pytest.approx(0.0675918, abs=1e-7)
    assert retargeted["bottom_time_s"] == pytest.approx(1.5)
    assert retargeted["descent_s"] == pytest.approx(1.5)
    assert retargeted["ascent_s"] == pytest.approx(1.5)
    assert retargeted["sagittal_path_length_m"] == pytest.approx(1.0512985163, abs=1e-9)
    assert retargeted["peak_down_vertical_speed_m_s"] == pytest.approx(
        -0.6100124581, abs=1e-9
    )
    assert retargeted["peak_down_vertical_speed_time_s"] == pytest.approx(1.32)
    assert retargeted["peak_up_vertical_speed_m_s"] == pytest.approx(
        0.6085487940, abs=1e-9
    )
    assert retargeted["peak_up_vertical_speed_time_s"] == pytest.approx(1.68)


def _write_bundle(path: Path, time_s: list[float]) -> None:
    manifest = {
        "provenance": {
            "sagittal_solution": {
                "time_s": time_s,
                "bar_m": [[0.0, 1.0], [0.1, 0.0], [0.2, 1.0]],
            },
            "ik_bar_path_m": [[0.0, 0.0, 1.0], [0.1, 0.0, 0.0], [0.2, 0.0, 1.0]],
            "ik": {"sagittal_forward": [1.0, 0.0, 0.0]},
            "optimizer": {},
        }
    }
    np.savez(path, manifest=np.asarray(json.dumps(manifest)))


@pytest.mark.parametrize("time_s", ([0.0, 0.1], [0.0, float("nan"), 0.2]))
def test_rejects_short_or_nonfinite_bundle_time_axis(
    tmp_path: Path, time_s: list[float]
) -> None:
    bundle = tmp_path / "malformed.npz"
    _write_bundle(bundle, time_s)

    with pytest.raises(
        ValueError, match="time_s must contain at least three finite samples"
    ):
        extract_metrics(bundle)


def test_sampling_description_does_not_assume_twenty_millisecond_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "different-sampling.npz"
    _write_bundle(bundle, [0.0, 0.1, 0.2])

    report = extract_metrics(bundle)

    assert report["sampling"]["dt_s"] == pytest.approx(0.1)
    assert report["sampling"]["velocity_method"] == (
        "numpy.gradient against saved sample times; no added filtering"
    )
