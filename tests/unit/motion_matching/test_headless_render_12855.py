"""Playback GIF rendering is best-effort on headless hosts (#12855)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.motion_matching.pipeline import cli  # noqa: E402
from src.shared.python.motion_matching.pipeline.cli import (  # noqa: E402
    PipelineContext,
    _render_playbacks,
    _simulate_and_receipt,
)

pytestmark = pytest.mark.unit

_NO_GL = "an OpenGL platform library has not been loaded into this process"


def _lane_kin() -> tuple[MagicMock, MagicMock]:
    lane = MagicMock()
    lane.points = np.zeros((2, 1, 3))
    lane.rate_hz = 360.0
    kin = MagicMock()
    kin.coordinate_order = ["j1"]
    return lane, kin


def test_render_playbacks_skips_on_fatal_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise mujoco.FatalError(_NO_GL)

    monkeypatch.setattr(cli, "render_playback", boom)
    lane, kin = _lane_kin()
    with caplog.at_level(logging.WARNING, logger=cli.logger.name):
        reason = _render_playbacks(
            tmp_path,
            MagicMock(spec_bytes=b"{}"),
            kin,
            lane,
            np.zeros((2, 1)),
            np.zeros((2, 1)),
        )
    assert reason is not None and "FatalError" in reason
    assert "MUJOCO_GL" in caplog.text


def test_render_playbacks_does_not_swallow_other_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise ValueError("real bug")

    monkeypatch.setattr(cli, "render_playback", boom)
    lane, kin = _lane_kin()
    with pytest.raises(ValueError, match="real bug"):
        _render_playbacks(
            tmp_path,
            MagicMock(spec_bytes=b"{}"),
            kin,
            lane,
            np.zeros((2, 1)),
            np.zeros((2, 1)),
        )


def test_render_playbacks_returns_none_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "render_playback", lambda *a, **k: None)
    lane, kin = _lane_kin()
    assert (
        _render_playbacks(
            tmp_path,
            MagicMock(spec_bytes=b"{}"),
            kin,
            lane,
            np.zeros((2, 1)),
            np.zeros((2, 1)),
        )
        is None
    )


def test_receipt_still_produced_and_records_render_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise mujoco.FatalError(_NO_GL)

    monkeypatch.setattr(cli, "render_playback", boom)
    monkeypatch.setattr(cli, "replay", lambda *a, **k: (MagicMock(), np.zeros((2, 1))))
    monkeypatch.setattr(
        cli, "build_dynamics_report", lambda *a, **k: ({}, np.zeros((2, 1)))
    )
    monkeypatch.setattr(cli, "finish_feasibility_report", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_save_dynamics_record", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_ground_support_receipt", lambda *a, **k: {"k": "v"})
    monkeypatch.setattr(cli, "log_pipeline_summary", lambda *a, **k: None)
    monkeypatch.setattr(cli.fs, "reference_zmp", lambda *a, **k: None)

    lane = MagicMock()
    lane.times = np.array([0.0, 0.05])
    lane.rate_hz = 360.0
    lane.impact_index = None
    lane.points = np.zeros((2, 1, 3))
    kin = MagicMock()
    kin.coordinate_order = ["j1"]
    cal_res = MagicMock()
    cal_res.qualification_note = "ok"
    cal_res.spec_bytes = b"{}"
    for attr in ("hip_report", "calibration", "calibration2", "address_report"):
        setattr(cal_res, attr, {})
    args = argparse.Namespace(
        trajectory_optimiser="none",
        mjx_iterations=40,
        zmp_filter=False,
        shooting_fit=0,
        tracking="kkt",
        backend="mujoco",
        spec="dummy.json",
        recalibrate_upper=False,
        anthropometric=None,
        capture="driver",
    )
    ctx = PipelineContext(
        args=args,
        out_dir=tmp_path,
        c3d_path=Path("dummy.c3d"),
        engine="mujoco",
        log=logging.getLogger("t"),
        t_start=0.0,
    )
    receipt = _simulate_and_receipt(
        ctx,
        lane,
        kin,
        MagicMock(),
        MagicMock(),
        ("M1",),
        np.zeros((2, 1)),
        cal_res,
        {},
        {},
    )
    assert "FatalError" in receipt["render_skipped"]
    on_disk = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert on_disk["render_skipped"] == receipt["render_skipped"]
    assert on_disk["k"] == "v"
