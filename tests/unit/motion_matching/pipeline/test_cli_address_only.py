"""``--address-only`` dispatch and the address report writer (#11737)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import cli

pytestmark = pytest.mark.unit


def test_parser_accepts_address_only() -> None:
    args = cli.build_parser().parse_args(["--address-only"])
    assert args.address_only is True
    assert cli.build_parser().parse_args([]).address_only is False


@pytest.mark.parametrize("flag", [True, False])
def test_main_dispatches_on_address_only(
    monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_address_stage", lambda a: calls.append("address"))
    monkeypatch.setattr(cli, "run_pipeline", lambda a: calls.append("full"))
    argv = ["cli", "--address-only"] if flag else ["cli"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == (["address"] if flag else ["full"])


def test_run_address_stage_writes_report_with_hip_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = ("pelvis_tx", "hip_rotation_l", "hip_adduction_l", "knee_angle_l")
    cal_res = SimpleNamespace(
        address_report={"foot_progression": {"left": {"error_deg": 0.4}}},
        kin=SimpleNamespace(coordinate_order=names),
        address2=SimpleNamespace(q=np.radians([0.0, -12.0, 5.0, 30.0])),
    )
    monkeypatch.setattr(
        cli, "calibrate_run", lambda a: SimpleNamespace(cal_res=cal_res)
    )
    report = cli.run_address_stage(SimpleNamespace(out=tmp_path))
    assert report["hip_coordinates_deg"] == pytest.approx(
        {"hip_rotation_l": -12.0, "hip_adduction_l": 5.0}
    )
    written = json.loads((tmp_path / "address_report.json").read_text())
    assert written["foot_progression"]["left"]["error_deg"] == 0.4


def test_calibrated_run_is_a_plain_record() -> None:
    run = cli.CalibratedRun(None, None, {}, (), None)  # type: ignore[arg-type]
    assert run.base_spec == {}
