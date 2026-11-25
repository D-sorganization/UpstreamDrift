"""Every clip says how its motion was produced (#15150, D3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.sim_run.native_export_jobs import manifest_of
from src.tools.native_viewer_export import replay_mode as rm
from src.tools.native_viewer_export.core import ExportResult

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _bundle(efforts: np.ndarray, provenance: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        efforts=efforts, hold_efforts=None, provenance=provenance or {}
    )


def test_zero_efforts_are_a_kinematic_replay() -> None:
    assert rm.replay_mode_of(_bundle(np.zeros((4, 2)))) is rm.ReplayMode.KINEMATIC


def test_tracker_controller_provenance_is_feedback_tracked() -> None:
    bundle = _bundle(np.ones((4, 2)), {"controller": "computed-torque"})
    assert rm.replay_mode_of(bundle) is rm.ReplayMode.FEEDBACK_TRACKED


def test_declared_open_loop_label_is_honoured() -> None:
    bundle = _bundle(np.ones((4, 2)), {"replay_mode": "open_loop_dynamics"})
    assert rm.replay_mode_of(bundle) is rm.ReplayMode.OPEN_LOOP


def test_declared_inverse_dynamics_is_its_own_honest_mode() -> None:
    """Kinetic-match efforts come from inverse dynamics of the recorded motion
    (#15685): not simulated, so neither tracked nor open-loop dynamics."""
    bundle = _bundle(np.ones((4, 2)), {"replay_mode": "inverse_dynamics"})
    assert rm.replay_mode_of(bundle) is rm.ReplayMode.INVERSE_DYNAMICS
    text = rm.hud_line(rm.ReplayMode.INVERSE_DYNAMICS)
    assert "Inverse Dynamics" in text and "not simulated" in text


def test_declared_label_beats_the_controller_key() -> None:
    bundle = _bundle(
        np.ones((4, 2)),
        {"controller": "x", "replay_mode": "open_loop_dynamics"},
    )
    assert rm.replay_mode_of(bundle) is rm.ReplayMode.OPEN_LOOP


def test_zero_efforts_override_a_dynamics_label() -> None:
    bundle = _bundle(np.zeros((4, 2)), {"replay_mode": "open_loop_dynamics"})
    assert rm.replay_mode_of(bundle) is rm.ReplayMode.KINEMATIC


def test_undeterminable_is_unknown_never_guessed() -> None:
    assert rm.replay_mode_of(_bundle(np.ones((4, 2)))) is rm.ReplayMode.UNKNOWN
    stale = _bundle(np.ones((4, 2)), {"replay_mode": "teleport"})
    assert rm.replay_mode_of(stale) is rm.ReplayMode.UNKNOWN


def test_empty_efforts_raise() -> None:
    with pytest.raises(ValueError, match="efforts"):
        rm.replay_mode_of(_bundle(np.zeros((0, 2))))


def test_display_text_comes_from_settings_for_every_mode() -> None:
    texts = {m: rm.display_text(m) for m in rm.ReplayMode}
    assert len(set(texts.values())) == len(rm.ReplayMode)
    assert (
        texts[rm.ReplayMode.UNKNOWN] == "Unknown Replay Mode (provenance not recorded)"
    )
    assert "kinematic replay" in texts[rm.ReplayMode.KINEMATIC].lower()
    with pytest.raises(TypeError):
        rm.display_text("kinematic_replay")  # type: ignore[arg-type]


def test_hud_line_includes_the_label() -> None:
    line = rm.hud_line(rm.ReplayMode.FEEDBACK_TRACKED)
    assert line.startswith("Replay: ")
    assert rm.display_text(rm.ReplayMode.FEEDBACK_TRACKED) in line


def test_settings_reject_a_missing_mode(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"labels": {"kinematic_replay": "x"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        rm.load_labels(path)


def test_manifest_carries_the_replay_mode() -> None:
    result = ExportResult("mujoco", {}, 0, replay_mode="kinematic_replay")
    row = manifest_of([result])["engines"][0]
    assert row["replay_mode"] == "kinematic_replay"
    assert ExportResult("mujoco", {}, 0).replay_mode == "unknown_replay_mode"


def test_hud_line_of_reads_the_bundle() -> None:
    """``hud_line_of`` labels the bundle's recorded mode."""
    bundle = _bundle(np.ones((4, 2)))
    assert rm.hud_line_of(bundle) == rm.hud_line(rm.ReplayMode.UNKNOWN)
