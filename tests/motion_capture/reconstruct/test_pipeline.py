"""Session chain: observations -> clean -> fit, driven like a real take."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct import __main__ as recon_cli
from src.motion_capture.reconstruct.pipeline import (
    CLEAN_REPORT_FILE,
    RECONSTRUCT_DIR,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.rig import __main__ as rig_cli

pytestmark = pytest.mark.unit


def _session(tmp_path: Path) -> tuple[Path, Path]:
    """A session directory shaped like rig ingest output, from the renderer."""
    session = tmp_path / "take"
    assert (
        recon_cli.main(
            ["synth", "--out", str(session), "--frames", "24", "--outliers", "0.02"]
        )
        == 0
    )
    truth = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    cameras = session / "cameras.json"
    cameras.write_text(json.dumps(truth["cameras"]), encoding="utf-8")
    (session / "truth.json").unlink()  # a real take has no truth
    return session, cameras


def test_reconstruct_session_writes_every_stage(tmp_path: Path) -> None:
    session, cameras = _session(tmp_path)
    summary = reconstruct_session(
        session, start_cameras=start_cameras_from(cameras), scale_anchor=("neck", 0.5)
    )
    out = session / RECONSTRUCT_DIR
    assert (out / "observations" / "face_on.json").is_file()
    assert (out / CLEAN_REPORT_FILE).is_file() and (
        out / "reconstruction.json"
    ).is_file()
    assert summary.views == ("face_on", "down_line", "overhead")
    assert summary.rms_px < 3.0
    record = json.loads((out / "reconstruction.json").read_text(encoding="utf-8"))
    assert record["metrics"] == {}  # no truth: no accuracy claims
    assert np.load(out / "joints_3d_m.npy").shape == (24, 15, 3)


def test_rig_reconstruct_command(tmp_path: Path) -> None:
    session, cameras = _session(tmp_path)
    code = rig_cli.main(
        [
            "reconstruct",
            "--session",
            str(session),
            "--cameras",
            str(cameras),
            "--anchor",
            "neck=0.5",
        ]
    )
    assert code == 0
    assert (session / RECONSTRUCT_DIR / "session_reconstruction.json").is_file()


def test_contracts(tmp_path: Path) -> None:
    session, cameras = _session(tmp_path)
    with pytest.raises(Exception, match="acceleration_sigma_px"):
        reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            acceleration_sigma_px=0,
        )
    with pytest.raises(Exception, match="must exist"):
        start_cameras_from(tmp_path / "nope.json")
