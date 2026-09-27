"""Unit tests for MJX knot optimisation in the matching pipeline (#11051)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
    KnotOptimisationSettings,
)
from src.shared.python.motion_matching.pipeline.constants import DEFAULT_MJX_ITERATIONS
from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
    run_trajectory_optimiser,
)
from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

pytestmark = pytest.mark.unit


def test_pipeline_constant_equals_knot_optimisation_settings_default() -> None:
    """Pipeline constant must equal KnotOptimisationSettings default iterations."""
    assert KnotOptimisationSettings().iterations == DEFAULT_MJX_ITERATIONS


@pytest.mark.timeout(900)
def test_run_trajectory_optimiser_mjx_knots_on_toy_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mjx-knots stage writes both files and never worsens the port check."""
    write_toy_package(tmp_path)
    # The toy writes the MJX package directly; there is no pipeline receipt
    # for the exporter to read.
    monkeypatch.setattr(
        "src.shared.python.motion_matching.execution.mjx_export.export_mjx_package",
        lambda run: {},
    )

    summary = run_trajectory_optimiser("mjx-knots", tmp_path, iterations=1)
    assert summary is not None
    assert summary["name"] == "mjx-knots"
    assert summary["iterations"] == 1
    assert (tmp_path / "mjx_optimised_reference.npz").is_file()
    assert (tmp_path / "mjx_optimisation_receipt.json").is_file()
    assert (
        summary["best_replay_marker_rms_m"] <= summary["port_check_replay_marker_rms_m"]
    )
