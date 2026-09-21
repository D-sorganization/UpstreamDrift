"""One-shot evidence writer for MS-52 committed artifacts (run locally in CI prep)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.requires_mujoco]

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "evidence/matched/driver_g1_crocoddyl_rk45_b100/candidate.npz"
OUT = ROOT / "evidence/matched/driver_g1_myosuite"


def test_write_driver_g1_myosuite_evidence() -> None:
    pytest.importorskip("mujoco")
    if not SOURCE.is_file():
        pytest.skip(f"Missing source candidate: {SOURCE}")

    from src.engines.physics_engines.myosuite.python.replay import (
        ReplayConfig,
        run_kinematic_replay,
    )

    receipt = run_kinematic_replay(
        ReplayConfig(candidate=SOURCE, output_dir=OUT, source_engine="mujoco")
    )
    assert receipt["stage"] == "replay"
    assert (OUT / "receipt.json").is_file()
    assert (OUT / "candidate.npz").is_file()
    assert (OUT / "playback.gif").is_file()
