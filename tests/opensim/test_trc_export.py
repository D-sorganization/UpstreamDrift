"""TRC marker-file export/import for OpenSim IK and Moco tracking."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching import trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _capture() -> TourCapture:
    t = np.arange(3) / 360
    labels = ("A", "B")
    points = np.array(
        [
            [[0.1, 1.5, -0.2], [0.3, 0.9, 0.4]],
            [[0.11, 1.51, -0.21], [np.nan, np.nan, np.nan]],
            [[0.12, 1.52, -0.22], [0.32, 0.92, 0.42]],
        ]
    )
    valid = np.array([[True, True], [True, False], [True, True]])
    return TourCapture(t, labels, points, valid)


def test_roundtrip_preserves_values_and_blanks_invalid(tmp_path: Path) -> None:
    path = tmp_path / "cap.trc"
    trc.write_trc(_capture(), path, rate_hz=360.0)
    text = path.read_text()
    lines = text.splitlines()
    assert lines[0].startswith("PathFileType\t4\t(X/Y/Z)\tcap.trc")
    assert lines[2].split("\t")[:5] == ["360.00", "360.00", "3", "2", "m"]
    assert lines[3].split("\t")[:3] == ["Frame#", "Time", "A"]
    assert lines[4].split("\t")[2:5] == ["X1", "Y1", "Z1"]
    back = trc.read_trc(path)
    assert back.labels == ("A", "B")
    np.testing.assert_allclose(back.time_s, _capture().time_s)
    np.testing.assert_array_equal(back.valid, _capture().valid)
    np.testing.assert_allclose(
        back.points_m[back.valid], _capture().points_m[_capture().valid]
    )
    assert np.isnan(back.points_m[1, 1]).all()
    assert lines[7].split("	")[5:8] == ["NaN", "NaN", "NaN"]


def test_write_rejects_bad_rate_and_unit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        trc.write_trc(_capture(), tmp_path / "x.trc", rate_hz=0.0)
    with pytest.raises(ValueError):
        trc.write_trc(_capture(), tmp_path / "x.trc", rate_hz=360.0, units="cm")


def test_read_rejects_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.trc"
    bad.write_text("PathFileType\t4\n")
    with pytest.raises(ValueError):
        trc.read_trc(bad)
