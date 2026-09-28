"""Tests for the optional MJX knot optimiser stage of the matching pipeline (#11051).

The prototype-level checks this file used to hold (knot basis, horizon mask,
contact-law parity, weld wrenches, foot preload, rollout gradients) load
private copies from the evidence script that #11046 replaced with the tested
``src`` modules; they are covered by ``test_knot_gradient_optimiser.py``,
``test_jax_contact.py`` and ``tests/unit/engines/mujoco/test_mjx_tracking_plant.py``.
"""

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

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_pipeline_iteration_default_matches_knot_settings_default() -> None:
    """The shared pipeline constant mirrors the engine settings default."""
    assert KnotOptimisationSettings().iterations == DEFAULT_MJX_ITERATIONS


@pytest.mark.timeout(900)
def test_pipeline_mjx_knots_stage_on_toy_package(
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
