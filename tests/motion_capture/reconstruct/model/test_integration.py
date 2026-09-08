"""fit-model through the rig CLI, the Simscape CSV, the workflow step and the tile."""

from __future__ import annotations

import json
import shutil
import os
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.model.golfer import (
    GOLFER_LANDMARK_MAP,
    SIMSCAPE_NAMES,
    golfer_model,
    write_simscape_csv,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.motion_capture.rig import __main__ as rig_cli
from src.tools.capture_rig import commands
from src.tools.capture_rig.session import SessionMedia, ViewMedia
from src.tools.capture_rig.workflow import STEPS, Status, evaluate

pytestmark = pytest.mark.unit


def _golfer_session(tmp_path: Path, frames: int = 20, fps: float = 60.0) -> Path:
    """A reconstruct/ directory holding a golfer motion in the 15-joint layout."""
    model = golfer_model()
    t = np.arange(frames) / fps
    q = np.zeros((frames, model.n_dof))
    q[:, 1] = 1.0
    q[:, model.dof_slice("hub")] = (0.5 * np.sin(1.5 * t))[:, None]
    q[:, model.dof_slice("left_scapula").start] = 0.3 * np.sin(2.0 * t)
    q[:, model.dof_slice("left_elbow")] = 0.8
    q[:, model.dof_slice("right_elbow")] = 0.5
    q[:, model.dof_slice("left_knee")] = 0.3
    q[:, model.dof_slice("right_knee")] = 0.3
    landmarks = model.landmarks(q)
    joints = np.full((frames, len(JOINT_NAMES), 3), np.nan)
    for name, source in GOLFER_LANDMARK_MAP.to_reconstruct.items():
        joints[:, JOINT_NAMES.index(source)] = landmarks[
            :, model.landmark_names.index(name)
        ]
    recon = tmp_path / "reconstruct"
    recon.mkdir()
    np.save(recon / "joints_3d_m.npy", joints)
    (recon / "session_reconstruction.json").write_text(
        json.dumps({"fps": fps, "views": ["a", "b"], "rms_px": 1.0}), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.timeout(180)  # whole-trajectory fits; CI's default is 60 s
def test_rig_fit_model_writes_model_dir_and_export_adds_simscape_csv(
    tmp_path: Path,
) -> None:
    root = _golfer_session(tmp_path)
    assert (
        rig_cli.main(["fit-model", "--session", str(root), "--sigma-accel", "300"]) == 0
    )
    angles = json.loads((root / "model" / "joint_angles.json").read_text("utf-8"))
    assert angles["model"] == "golfer-scapula/2.0" and len(angles["q"]) == 20
    report = json.loads((root / "model" / "fit_report.json").read_text("utf-8"))
    assert report["rms_mm"] < 5.0 and report["velocity_violations"] == 0
    assert report["landmarks"]["left_shoulder"]["frames"] == 20
    csv_path = write_simscape_csv(root / "model" / "joint_angles.json")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# upstreamdrift golfer-scapula")
    header = lines[1].split(",")
    assert set(header) == {"time_s", *(n for n, _ in SIMSCAPE_NAMES.values())}
    assert len(lines) == 2 + 20
    with pytest.raises(ValueError, match="lack Simscape"):
        bad = root / "model" / "bad.json"
        bad.write_text(
            json.dumps({"dof_names": ["a"], "q": [[0.0]], "fps": 60}), encoding="utf-8"
        )
        write_simscape_csv(bad)


def test_workflow_has_a_fit_model_step_gated_on_the_reconstruction() -> None:
    keys = [s.key for s in STEPS]
    assert keys.index("fit_model") == keys.index("reconstruct") + 1
    view = ViewMedia("a", "a", Path("a.avi"), None, Path("a.json"), 60.0)
    other = ViewMedia("b", "b", Path("b.avi"), None, Path("b.json"), 60.0)
    base = {
        "root": Path("s"),
        "plan_name": "p",
        "views": (view, other),
        "swing_summary": None,
        "reconstruction": None,
        "problems": (),
    }
    states = {s.step.key: s for s in evaluate(SessionMedia(**base))}
    assert states["fit_model"].status is Status.BLOCKED
    states = {
        s.step.key: s
        for s in evaluate(SessionMedia(**{**base, "reconstruction": {"rms_px": 1}}))
    }
    assert states["fit_model"].status is Status.READY
    fitted = SessionMedia(
        **{**base, "reconstruction": {"rms_px": 1}, "model_fit": {"rms_mm": 3}}
    )
    assert {s.step.key: s for s in evaluate(fitted)}["fit_model"].status is Status.DONE
    single = {
        s.step.key: s for s in evaluate(SessionMedia(**{**base, "views": (view,)}))
    }
    assert single["fit_model"].status is Status.SKIPPED


def test_fit_model_command_and_tile_action(tmp_path: Path) -> None:
    argv = commands.fit_model_command(tmp_path, sigma_accel=200, max_velocity=30)
    assert argv[3] == "fit-model" and argv[argv.index("--sigma-accel") + 1] == "200"
    with pytest.raises(Exception, match="positive"):
        commands.fit_model_command(tmp_path, sigma_accel=0)
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests.tools.capture_rig.test_gui import _app

    from src.tools.capture_rig.gui import CaptureRigWidget

    _app()
    widget = CaptureRigWidget()
    widget.capture.session_edit.setText(str(tmp_path))
    assert widget.command_for("fit_model")[3] == "fit-model"
    assert widget.results.tabText(widget.results.count() - 1) == "Kinetics"
    assert widget.process.model_name() == "golfer"


@pytest.mark.timeout(180)  # whole-trajectory fits; CI's default is 60 s
def test_rig_compare_models_and_kinetics_commands(tmp_path: Path) -> None:
    root = _golfer_session(tmp_path)
    assert (
        rig_cli.main(
            [
                "compare-models",
                "--session",
                str(root),
                "--models",
                "double_pendulum,triple_pendulum",  # cheap: CI's 60 s test budget
                "--max-iterations",
                "20",
            ]
        )
        == 0
    )
    comparison = json.loads((root / "model" / "comparison.json").read_text("utf-8"))
    ranked = {s["model"] for s in comparison["ranking"]}
    assert ranked == {"double_pendulum", "triple_pendulum"}
    assert (
        rig_cli.main(
            [
                "fit-model",
                "--session",
                str(root),
                "--model",
                "golfer",
                "--max-iterations",
                "20",
            ]
        )
        == 0
    )
    assert (
        rig_cli.main(
            [
                "kinetics",
                "--session",
                str(root),
                "--model",
                "golfer",
                "--body-mass",
                "80",
            ]
        )
        == 0
    )
    kinetics = json.loads((root / "model" / "kinetics.json").read_text("utf-8"))
    assert kinetics["model"] == "golfer-scapula/2.0" and len(kinetics["tau"]) == 20
    assert "LScapStartPositionX" in kinetics["simscape_names"].values()
    shutil.rmtree(root / "model" / "triple_pendulum")  # kinetics needs a fit first
    with pytest.raises(SystemExit, match="fit-model"):
        rig_cli.main(
            [
                "kinetics",
                "--session",
                str(root),
                "--model",
                "triple_pendulum",
                "--body-mass",
                "80",
            ]
        )
    assert (
        commands.compare_models_command(root, models=("golfer",), fit_lengths=True)[-1]
        == "--fit-lengths"
    )
    kc = commands.kinetics_command(root, model="golfer", body_mass_kg=75)
    assert kc[kc.index("--body-mass") + 1] == "75"
    with pytest.raises(Exception, match="positive"):
        commands.kinetics_command(root, model="golfer", body_mass_kg=0)
    fm = commands.fit_model_command(root, model="triple_pendulum", fit_lengths=True)
    assert (
        fm[fm.index("--model") + 1] == "triple_pendulum" and fm[-1] == "--fit-lengths"
    )
    assert dict(commands.model_choices())["golfer"].startswith("Scapula-capable")
