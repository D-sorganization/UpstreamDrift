"""Unit tests for MJX trajectory optimisation evidence CLI (#11046)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

import numpy as np

from docs.development.full_body_models.evidence.ground_support.mjx_trajectory_optimisation import (
    main,
)
from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_jax_x64() -> Iterator[None]:
    """The CLI forces float32 globally; keep that out of later x64 tests."""
    previous = bool(jax.config.jax_enable_x64)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_mjx_evidence_cli_iterations_2(tmp_path: Path) -> None:
    """Run CLI main with --iterations 2 on a tiny package and verify receipt assertions."""
    write_toy_package(tmp_path)

    # Run CLI main
    main(
        [
            "--run",
            str(tmp_path),
            "--iterations",
            "2",
            "--substeps",
            "2",
            "--knot-spacing",
            "0.05",
            "--weld-stiffness",
            "1000.0",
            "--weld-damping",
            "10.0",
            "--horizon",
            "0.2",
        ]
    )

    receipt_path = tmp_path / "mjx_optimisation_receipt.json"
    assert receipt_path.exists(), "Receipt JSON must exist"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (
        receipt["port_check_replay_marker_rms_m"]
        == receipt["history"][0]["replay_marker_rms_m"]
    ), "Port check replay marker RMS must equal history[0]"
    assert (
        receipt["best_replay_marker_rms_m"]
        <= receipt["history"][0]["replay_marker_rms_m"]
    ), "Best replay marker RMS must be <= iteration 0 RMS"

    opt_ref_path = tmp_path / "mjx_optimised_reference.npz"
    assert opt_ref_path.exists(), "Optimised reference npz must exist"
