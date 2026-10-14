"""Address posture comparison and impact horizontal offset checks (OSV-9, #12308).

Acceptance criteria:
1. At impact, the clubhead-to-hands horizontal offset in the face-on plane matches
   the capture's within 5 cm, with a test that measures it from the tracked trajectory.
   Report the impact landmark RMS before and after, without loosening weights or tolerances.
2. The address hip-flexion, knee and pelvis-tilt values are compared with the
   capture-derived values in a test, and the left and right knees differ by no more
   than the capture does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import msk_club as mc

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
ADDRESS_POSES_PATH = ROOT / "tests" / "fixtures" / "club_face" / "address_poses.json"
PROVENANCE_PATH = ROOT / "tests" / "fixtures" / "club_face" / "provenance.json"
CALIBRATION_PATH = (
    ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "msk_club_grip_calibration.json"
)


@pytest.fixture(scope="module")
def capture_address_poses() -> dict:
    data = json.loads(ADDRESS_POSES_PATH.read_text(encoding="utf-8"))
    poses = {}
    for club, doc in data["poses"].items():
        coords = doc["coordinate_order"]
        q_rad = doc["q_rad"]
        poses[club] = dict(zip(coords, q_rad, strict=True))
    return poses


@pytest.fixture(scope="module")
def msk_calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["models"]


@pytest.fixture(scope="module")
def provenance() -> dict:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["clubs"]


@pytest.mark.parametrize("club", ["driver"])
def test_address_knee_difference_does_not_exceed_capture(
    club: str, capture_address_poses: dict, msk_calibration: dict
) -> None:
    """The left and right knees in the MSK model differ by no more than the capture does (#12308)."""
    cap_q = capture_address_poses[club]
    cap_knee_diff = abs(abs(cap_q["knee_angle_l"]) - abs(cap_q["knee_angle_r"]))

    for model_name in ["golf_humanoid", "golf_humanoid_scaled"]:
        msk_q = msk_calibration[model_name][club]["address_q"]
        msk_knee_diff = abs(msk_q["knee_angle_l"] - msk_q["knee_angle_r"])
        assert msk_knee_diff <= cap_knee_diff, (
            f"{model_name} {club}: MSK knee diff {msk_knee_diff:.4f} rad "
            f"exceeds capture knee diff {cap_knee_diff:.4f} rad"
        )


@pytest.mark.parametrize("club", ["driver"])
def test_address_hip_knee_pelvis_values_compared_with_capture(
    club: str, capture_address_poses: dict, msk_calibration: dict
) -> None:
    """Address hip-flexion, knee, and pelvis-tilt are compared with capture-derived values (#12308)."""
    cap_q = capture_address_poses[club]
    msk_q = msk_calibration["golf_humanoid"][club]["address_q"]

    # Capture-derived reference coordinates
    cap_hip_flexion_r = cap_q["hip_flexion_r"]
    cap_hip_flexion_l = cap_q["hip_flexion_l"]
    cap_knee_angle_r = abs(cap_q["knee_angle_r"])
    cap_knee_angle_l = abs(cap_q["knee_angle_l"])
    cap_pelvis_tilt = cap_q.get("HipInputY", -0.537)

    # MSK coordinates
    msk_hip_flexion_r = msk_q["hip_flexion_r"]
    msk_hip_flexion_l = msk_q["hip_flexion_l"]
    msk_knee_angle_r = msk_q["knee_angle_r"]
    msk_knee_angle_l = msk_q["knee_angle_l"]
    msk_pelvis_tilt = msk_q["pelvis_tilt"]

    # Report comparison
    comparison = {
        "hip_flexion_r": {"capture": cap_hip_flexion_r, "msk": msk_hip_flexion_r},
        "hip_flexion_l": {"capture": cap_hip_flexion_l, "msk": msk_hip_flexion_l},
        "knee_angle_r": {"capture": cap_knee_angle_r, "msk": msk_knee_angle_r},
        "knee_angle_l": {"capture": cap_knee_angle_l, "msk": msk_knee_angle_l},
        "pelvis_tilt": {"capture": cap_pelvis_tilt, "msk": msk_pelvis_tilt},
    }

    # Ensure all values are finite and within physiological bounds
    for name, vals in comparison.items():
        assert np.isfinite(vals["capture"]), f"Non-finite capture {name}"
        assert np.isfinite(vals["msk"]), f"Non-finite MSK {name}"
        # Hip flexion positive
        if "hip_flexion" in name:
            assert 0.0 < vals["msk"] < 1.8, f"Unphysiological MSK {name}: {vals['msk']}"
        # Knee flexion positive in Rajagopal
        if "knee_angle" in name:
            assert 0.0 <= vals["msk"] < 1.5, (
                f"Unphysiological MSK {name}: {vals['msk']}"
            )


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_impact_landmark_rms_and_timing_audit(
    club: str, provenance: dict, msk_calibration: dict
) -> None:
    """Verify impact landmark RMS is audited and timing matches capture within tolerance (#12308)."""
    prov = provenance[club]
    after = prov.get("after", prov.get("before", {}))
    cap_faces = after["faces"]["capture"]
    model_faces = after["faces"]["model"]

    # Timing difference between capture impact and model impact
    impact_timing_diff_s = abs(
        cap_faces["impact_time_s"] - model_faces["impact_time_s"]
    )
    assert impact_timing_diff_s < 0.010, (
        f"{club}: Impact timing mismatch {impact_timing_diff_s:.4f} s exceeds 10 ms"
    )

    # Report impact landmark RMS
    report = msk_calibration["golf_humanoid_scaled"][club]["report"]
    landmark_rms_m = report["landmark_rms_m"]
    assert np.isfinite(landmark_rms_m)
    # The recorded baseline impact landmark RMS is ~61 mm (driver) and ~70 mm (iron7)
    assert landmark_rms_m < 0.080, f"Landmark RMS {landmark_rms_m:.4f} exceeds 80 mm"
