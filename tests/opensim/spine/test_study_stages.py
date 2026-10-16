"""Failure attribution of the spine study stage runner (SPN-13a, #12766).

The native stages are replaced by recording stubs so these tests need no
OpenSim; they pin that a failure after all five numerical stages (loads export,
context bands) is recorded as a failed stage rather than raising IndexError.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.engines.physics_engines.opensim.python.spine import study_stages as ss
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    StageRecord,
)

pytestmark = pytest.mark.unit


def _ok(name: str):
    def stage(*args, **kwargs):
        out = next(a for a in args if isinstance(a, ss.StageOutcome))
        out.stages.append(StageRecord(stage=name, status="ok"))
        return SimpleNamespace(body_weight_n=700.0)

    return stage


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    def drive(request, model, scrub, out):
        out.stages.append(StageRecord(stage="golf_input", status="ok"))
        out.stages.append(StageRecord(stage="drive", status="ok"))
        return SimpleNamespace()

    def inv(request, model, drive_result, scrub, out):
        out.stages.append(StageRecord(stage="inverse_dynamics", status="ok"))
        return SimpleNamespace(), SimpleNamespace(), SimpleNamespace(body_weight_n=1.0)

    monkeypatch.setattr(ss, "_drive", drive)
    monkeypatch.setattr(ss, "_inverse_dynamics", inv)
    monkeypatch.setattr(ss, "_static_optimization", _ok("static_optimization"))
    monkeypatch.setattr(ss, "_joint_reactions", _ok("joint_reactions"))
    monkeypatch.setattr(ss, "_write_loads", lambda *a, **k: None)
    monkeypatch.setattr(ss, "_context", lambda *a, **k: None)
    return SimpleNamespace(output_dir=tmp_path / "out")


def test_all_stages_ok_is_not_failed(stubbed) -> None:
    out = ss.run_stages(stubbed, Path("m.osim"), str)
    assert out.failed is False
    assert len(out.stages) == 5


@pytest.mark.parametrize(
    ("target", "exc", "stage"),
    [
        ("_write_loads", OSError("disk full"), "loads_export"),
        ("_context", ValueError("bad band"), "context_bands"),
    ],
)
def test_failure_after_five_stages_is_recorded(
    stubbed, monkeypatch, target, exc, stage
) -> None:
    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(ss, target, boom)
    out = ss.run_stages(stubbed, Path("m.osim"), str)
    assert out.failed is True
    assert out.stages[-1].status == "failed"
    assert out.stages[-1].stage == stage
    assert stage in ss.STAGE_NAMES
    assert any(stage in r for r in out.reasons)


def test_failure_in_drive_after_golf_input_is_attributed_to_drive(
    stubbed, monkeypatch
) -> None:
    def partial(request, model, scrub, out):
        out.stages.append(StageRecord(stage="golf_input", status="ok"))
        out.active = "drive"
        raise ValueError("drive failed")

    monkeypatch.setattr(ss, "_drive", partial)
    out = ss.run_stages(stubbed, Path("m.osim"), str)
    assert [s.stage for s in out.stages] == ["golf_input", "drive"]
    assert out.stages[-1].status == "failed"
