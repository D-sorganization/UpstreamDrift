"""Engine-agnostic tour-average capture contract (OS-1 marker/frame/clock)."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import tour_capture_contract as module

pytestmark = pytest.mark.unit
C3D = Path(__file__).resolve().parents[2] / "data/C3D_TA_Driver.c3d"


def test_capture_spec_is_frozen_and_complete() -> None:
    spec = module.TOUR_CAPTURE
    assert spec.frames == 654 and spec.rate_hz == 360.0
    assert spec.duration_s == pytest.approx(653 / 360)
    assert spec.units == "m" and spec.vertical_axis == "y"
    assert len(spec.labels) == 38 and len(set(spec.labels)) == 38
    assert spec.sha256.startswith("545405ccdbae87a2")


def test_every_label_has_exactly_one_segment() -> None:
    seen: list[str] = []
    for labels in module.MARKER_SEGMENTS.values():
        seen.extend(labels)
    assert sorted(seen) == sorted(module.TOUR_CAPTURE.labels)
    assert set(module.MARKER_SEGMENTS["left_leg"]) == {
        "LKneeOut",
        "LAnkleOut",
        "LToeIn",
        "LToeOut",
    }
    assert "Marker_0:0:0" in module.MARKER_SEGMENTS["unassigned"]
    assert module.tracked_labels() == tuple(
        label
        for label in module.TOUR_CAPTURE.labels
        if label not in module.MARKER_SEGMENTS["unassigned"]
    )


def test_synthetic_capture_validates_shapes_and_validity() -> None:
    t = np.arange(4) / 360
    points = np.zeros((4, 38, 3))
    valid = np.ones((4, 38), dtype=bool)
    points[1, 0] = np.nan
    valid[1, 0] = False
    capture = module.TourCapture(t, module.TOUR_CAPTURE.labels, points, valid)
    assert capture.frames == 4
    assert capture.valid_count() == 4 * 38 - 1
    sub = capture.subset(("WaistLeft", "HeadTop"))
    assert sub.labels == ("WaistLeft", "HeadTop") and sub.points_m.shape == (4, 2, 3)
    with pytest.raises(ValueError):
        module.TourCapture(t, module.TOUR_CAPTURE.labels, points[:, :5], valid)
    with pytest.raises(ValueError):
        module.TourCapture(t, module.TOUR_CAPTURE.labels, points, valid[:, :5])
    with pytest.raises(ValueError):
        capture.subset(("NotALabel",))
    bad = points.copy()
    bad[2, 3] = np.nan  # finite point required wherever valid is true
    with pytest.raises(ValueError):
        module.TourCapture(t, module.TOUR_CAPTURE.labels, bad, valid)


@pytest.mark.skipif(not C3D.is_file(), reason="canonical C3D not checked out")
def test_real_capture_matches_frozen_spec() -> None:
    capture = module.load_tour_capture(C3D)
    assert capture.frames == 654
    assert capture.labels == module.TOUR_CAPTURE.labels
    assert capture.time_s[0] == 0.0 and capture.time_s[-1] == pytest.approx(653 / 360)
    assert capture.points_m.shape == (654, 38, 3)
    # Y is vertical: HeadTop sits above the toes in the first frame.
    head = capture.points_m[0, capture.index("HeadTop")]
    toe = capture.points_m[0, capture.index("LToeOut")]
    assert head[1] > toe[1] + 1.0
    assert capture.valid_count() == 24135  # frozen file: residual>=0 and finite
    assert capture.source_sha256 == module.TOUR_CAPTURE.sha256


def test_loader_rejects_other_files(tmp_path: Path) -> None:
    other = tmp_path / "x.c3d"
    other.write_bytes(b"not a c3d")
    with pytest.raises(ValueError):
        module.load_tour_capture(other)
