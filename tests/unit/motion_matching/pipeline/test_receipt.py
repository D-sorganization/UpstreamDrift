from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.receipt import (
    GroundSupportReceiptInputs,
    build_ground_support_receipt,
)


@pytest.mark.unit
def test_build_ground_support_receipt_structure(tmp_path: Path) -> None:
    c3d_file = tmp_path / "dummy.c3d"
    c3d_file.write_bytes(b"dummy c3d content")

    base_spec = {
        "name": "test_spec",
        "club": "driver",
        "subject": {"grip_rotation_deg": {"L": [0.0, 0.0, 0.0]}},
    }
    spec_path = Path("spec.json")
    scaled_path = Path("scaled.json")
    hipcal_path = Path("hipcal.json")

    lane = MagicMock()
    lane.bounds = {}
    lane.labels = ("m1", "m2")
    lane.ground_cal.height_m = 0.0
    lane.ground_cal.lowest_marker_height_m = 0.03
    lane.ground_cal.standoff_m = 0.03
    lane.ground_cal.policy = "lowest_toe"
    lane.stance = [("heel_r", "forefoot_r")] * 10

    kin = MagicMock()
    q_ref = np.zeros((300, 10))

    from unittest.mock import patch

    with patch(
        "src.shared.python.motion_matching.pipeline.receipt.posture_summary",
        return_value={"spine_bend_deg": {}},
    ):
        receipt = build_ground_support_receipt(
            GroundSupportReceiptInputs(
                base_spec=base_spec,
                spec_path=spec_path,
                scaled_path=scaled_path,
                hipcal_path=hipcal_path,
                recalibrate_upper=False,
                anthropometric=None,
                qualification_note="test note",
                spec_bytes=b"{}",
                hip_report={"centre_r": [0, 0, 0]},
                candidate_bytes=b"{}",
                c3d_path=c3d_file,
                capture_name="driver",
                lane=lane,
                address_report={"rms": 0.005},
                ik_report={"rms": 0.027},
                dynamics_report={"rms": 0.074},
                kin=kin,
                q_ref=q_ref,
                elapsed_s=42.0,
                validate=False,
            )
        )

    assert receipt["base_spec_file"] == "spec.json"
    assert receipt["spec_file"] == "scaled.json"
    assert receipt["hipcal_spec_file"] == "hipcal.json"
    assert receipt["recalibrate_upper"] is False
    assert receipt["anthropometric"] is None
    assert "test note" in receipt["qualification"]
    assert receipt["elapsed_s"] == 42.0
    assert "ground" in receipt
    assert receipt["ground"]["stance_fraction"]["heel_r"] == 1.0
    assert receipt["address"]["rms"] == 0.005
    assert receipt["ik"]["rms"] == 0.027
    assert receipt["dynamics"]["rms"] == 0.074


@pytest.mark.unit
def test_build_ground_support_receipt_calls_validate(tmp_path: Path) -> None:
    c3d_file = tmp_path / "dummy.c3d"
    c3d_file.write_bytes(b"dummy c3d content")

    base_spec = {
        "name": "test_spec",
        "club": "driver",
        "subject": {"grip_rotation_deg": {"L": [0.0, 0.0, 0.0]}},
    }
    lane = MagicMock()
    lane.bounds = {}
    lane.labels = ("m1",)
    lane.ground_cal.height_m = 0.0
    lane.ground_cal.lowest_marker_height_m = 0.03
    lane.ground_cal.standoff_m = 0.03
    lane.ground_cal.policy = "lowest_toe"
    lane.stance = [("heel_r",)]

    from unittest.mock import patch

    with (
        patch(
            "src.shared.python.motion_matching.pipeline.receipt.posture_summary",
            return_value={},
        ),
        patch(
            "src.shared.python.motion_matching.pipeline.receipt.validate_receipt"
        ) as mock_validate,
    ):
        build_ground_support_receipt(
            GroundSupportReceiptInputs(
                base_spec=base_spec,
                spec_path=Path("spec.json"),
                scaled_path=Path("scaled.json"),
                hipcal_path=Path("hipcal.json"),
                recalibrate_upper=False,
                anthropometric=None,
                qualification_note="note",
                spec_bytes=b"{}",
                hip_report={},
                candidate_bytes=b"{}",
                c3d_path=c3d_file,
                capture_name="driver",
                lane=lane,
                address_report={},
                ik_report={},
                dynamics_report={},
                kin=MagicMock(),
                q_ref=np.zeros((10, 10)),
                elapsed_s=1.0,
                validate=True,
            )
        )
        mock_validate.assert_called_once()


@pytest.mark.unit
def test_log_pipeline_summary() -> None:
    from src.shared.python.motion_matching.pipeline.receipt import log_pipeline_summary

    log = MagicMock()
    receipt = {"address": {"marker_rms_m": 0.01}, "dynamics": {"marker_rms_m": 0.02}}
    ik_report = {
        "marker_rms_m": 0.015,
        "segment_rms_m": {},
        "reference": {},
        "segment_scaling": {},
        "leg_angle_ranges_deg": {},
    }
    cal1 = MagicMock()
    cal1.rms_per_iteration_m = [0.03, 0.02]
    cal2 = MagicMock()
    cal2.rms_per_iteration_m = [0.02, 0.015]

    log_pipeline_summary(log, receipt, ik_report, cal1, cal2)
    assert log.info.call_count == 3
