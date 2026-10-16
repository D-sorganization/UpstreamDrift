"""Unit tests for the calibrated-address recording helpers (R4, #12684)."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import address as addr

_P = "src.shared.python.motion_matching.pipeline"


@pytest.mark.unit
def test_record_calibrated_address_records_summary_feet_and_knees() -> None:
    report: dict = {}
    kin, lane = MagicMock(), MagicMock()
    address = MagicMock()
    address.q = np.zeros(3)
    with (
        patch.object(addr, "calibrated_address_summary", return_value={"k": 1}) as s,
        patch(f"{_P}.address_feet.record_foot_progression") as feet,
        patch.object(addr.address_knee, "record_knee_flexion") as knees,
    ):
        addr.record_calibrated_address(report, "sim", kin, address, lane, ("m",), "ad")
    assert report["calibrated"] == {"k": 1}
    s.assert_called_once_with("sim", kin, address, lane, ("m",), "ad")
    feet.assert_called_once_with(report, kin, address.q, lane.feet)
    knees.assert_called_once_with(report, kin, address.q, lane.knees)


@pytest.mark.unit
def test_record_calibrated_address_rejects_non_dict_report() -> None:
    with pytest.raises(TypeError, match="address_report must be a dict"):
        addr.record_calibrated_address(
            None,
            "sim",
            MagicMock(),
            MagicMock(),
            MagicMock(),
            (),
            "ad",  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["off", "prior", "capture"])
def test_knee_targets_from_args_forwards_requested_mode(mode: str) -> None:
    args = argparse.Namespace(knee_flexion=mode)
    with patch.object(addr, "build_knee_targets", return_value="t") as b:
        assert addr.knee_targets_from_args("lane", {"s": 1}, args) == "t"
    b.assert_called_once_with("lane", {"s": 1}, mode)


@pytest.mark.unit
def test_knee_targets_from_args_defaults_to_off_without_flag() -> None:
    with patch.object(addr, "build_knee_targets", return_value=None) as b:
        assert addr.knee_targets_from_args("lane", {}, argparse.Namespace()) is None
    b.assert_called_once_with("lane", {}, "off")
