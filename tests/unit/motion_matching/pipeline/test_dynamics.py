from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.dynamics import (
    com_report,
    rom_flags,
    segment_rms,
    zmp_summary,
)


@pytest.mark.unit
def test_segment_rms_computes_per_segment() -> None:
    # Pelvis markers: WaistLeft, WaistRight, WaistLBack, WaistRBack
    labels = ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack", "OtherMarker")
    errors = np.array(
        [
            [0.01, 0.02, 0.03, 0.04, 0.10],
            [0.01, 0.02, 0.03, 0.04, 0.10],
        ]
    )
    valid = np.ones((2, 5), dtype=bool)

    result = segment_rms(labels, errors, valid)
    assert "pelvis" in result
    expected_pelvis_rms = np.sqrt(np.mean(errors[:, :4] ** 2))
    assert pytest.approx(result["pelvis"]) == expected_pelvis_rms


@pytest.mark.unit
def test_segment_rms_dbc_validation() -> None:
    with pytest.raises(ValueError, match="errors must be a 2D array"):
        segment_rms(("M1",), np.zeros(5), np.ones((5, 1), dtype=bool))

    with pytest.raises(ValueError, match="Shape mismatch"):
        segment_rms(("M1",), np.zeros((2, 1)), np.ones((3, 1), dtype=bool))

    with pytest.raises(ValueError, match="Column count mismatch"):
        segment_rms(("M1", "M2"), np.zeros((2, 1)), np.ones((2, 1), dtype=bool))


@pytest.mark.unit
def test_zmp_summary_metrics() -> None:
    times = np.array([0.5, 1.1, 1.2, 1.4, 1.6])
    zmp = {
        "outside_m": np.array([0.0, 0.05, 0.10, 0.0, 0.02]),
        "unloaded": np.array([False, False, True, False, False]),
    }
    summary = zmp_summary(zmp, times)

    assert summary["outside_fraction"] == pytest.approx(3 / 5)
    # window 1.0 <= t < 1.5 has indices 1, 2, 3 -> values [0.05, 0.10, 0.0] -> 2 / 3 outside
    assert summary["outside_fraction_1s_to_1_5s"] == pytest.approx(2 / 3)
    assert summary["outside_max_m"] == pytest.approx(0.10)
    assert summary["unloaded_fraction"] == pytest.approx(1 / 5)


@pytest.mark.unit
def test_rom_flags_detects_violations() -> None:
    names = ["SpineInputX", "TorsoInput"]
    # 2 frames, degrees: SpineInputX at 80 deg (violates [-35, 35] in HUMAN_RANGES_DEG)
    q = np.array([[np.radians(80.0), 0.0], [np.radians(80.0), 0.0]])
    flags = rom_flags(q, names)

    assert "SpineInputX" in flags
    assert flags["SpineInputX"]["frames"] == 2
    assert flags["SpineInputX"]["max_excess_deg"] > 0


@pytest.mark.unit
def test_com_report_structure() -> None:
    sim = MagicMock()
    kin = MagicMock()
    ground = MagicMock()
    ground.height_m = 0.0

    sim.centre_of_mass.return_value = (np.array([0.1, 0.2, 0.8]), MagicMock())
    kin.sphere_ground_points.return_value = {
        "s1": np.array([0.0, 0.0, 0.0]),
        "s2": np.array([1.0, 0.0, 0.0]),
        "s3": np.array([0.5, 1.0, 0.0]),
    }
    q = np.zeros(10)
    report = com_report(sim, kin, q, ground)

    assert "com_m" in report
    assert report["com_m"] == [0.1, 0.2, 0.8]
    assert report["height_above_ground_m"] == pytest.approx(0.8)
    assert report["inside_support_polygon"] is True
    assert "polygon_centroid_offset_m" in report
