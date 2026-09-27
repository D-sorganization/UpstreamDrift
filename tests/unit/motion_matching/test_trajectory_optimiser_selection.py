"""Tests for trajectory optimiser selection in the full-body matching pipeline (#11051).

Runs in the default Python environment (no JAX required).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.workspace.installed_journeys import DependencyUnavailableError
from src.shared.python.motion_matching.pipeline.cli import (
    PipelineContext,
    _simulate_and_receipt,
    build_parser,
)
from src.shared.python.motion_matching.pipeline.constants import DEFAULT_MJX_ITERATIONS
from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
    TRAJECTORY_OPTIMISERS,
    run_trajectory_optimiser,
    validate_trajectory_optimiser,
)

pytestmark = pytest.mark.unit


def test_parser_defaults() -> None:
    """Argparse parser default for --trajectory-optimiser is 'none'."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.trajectory_optimiser == "none"
    assert args.mjx_iterations == DEFAULT_MJX_ITERATIONS


def test_unknown_name_rejected_by_argparse() -> None:
    """Argparse rejects unknown trajectory optimiser name."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--trajectory-optimiser", "unknown-optimiser"])


def test_unknown_name_rejected_by_validate() -> None:
    """validate_trajectory_optimiser rejects unknown names and non-strings."""
    with pytest.raises(ValueError, match="Unknown trajectory optimiser"):
        validate_trajectory_optimiser("unknown-optimiser")

    with pytest.raises((ValueError, TypeError)):
        validate_trajectory_optimiser(123)  # type: ignore[arg-type]


def test_validate_trajectory_optimiser_valid_names() -> None:
    """validate_trajectory_optimiser accepts supported names and normalises case."""
    assert validate_trajectory_optimiser("none") == "none"
    assert validate_trajectory_optimiser("NONE") == "none"
    assert validate_trajectory_optimiser("mjx-knots") == "mjx-knots"
    assert validate_trajectory_optimiser("MJX-KNOTS") == "mjx-knots"


def test_run_trajectory_optimiser_none_returns_none_and_creates_no_files(
    tmp_path: Path,
) -> None:
    """run_trajectory_optimiser('none', ...) returns None and touches nothing."""
    result = run_trajectory_optimiser("none", tmp_path, iterations=10)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_mjx_iterations_rejected_zero_or_negative() -> None:
    """--mjx-iterations 0 or negative value is rejected."""
    parser = build_parser()
    with pytest.raises((argparse.ArgumentTypeError, SystemExit)):
        parser.parse_args(["--mjx-iterations", "0"])

    with pytest.raises((argparse.ArgumentTypeError, SystemExit)):
        parser.parse_args(["--mjx-iterations", "-5"])


def test_mjx_knots_raises_dependency_unavailable_when_jax_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When jax is missing, 'mjx-knots' raises DependencyUnavailableError naming jax."""
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(DependencyUnavailableError) as exc_info:
        run_trajectory_optimiser("mjx-knots", tmp_path, iterations=1)
    err_msg = str(exc_info.value)
    assert "jax" in err_msg
    assert "--trajectory-optimiser none" in err_msg
    assert not (tmp_path / "mjx_optimised_reference.npz").exists()


def test_mjx_knots_raises_dependency_unavailable_when_mjx_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing mujoco.mjx is named too; there is no silent fallback to 'none'."""
    monkeypatch.setitem(sys.modules, "jax", MagicMock())
    monkeypatch.setitem(sys.modules, "mujoco.mjx", None)
    with pytest.raises(DependencyUnavailableError, match="mujoco.mjx"):
        run_trajectory_optimiser("mjx-knots", tmp_path, iterations=1)
    assert list(tmp_path.iterdir()) == []


def test_receipt_absent_for_none_and_present_for_stubbed_optimiser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt omits 'trajectory_optimiser' for 'none', and includes summary when enabled."""
    log = logging.getLogger("test_receipt")
    lane = MagicMock()
    lane.times = np.array([0.0, 0.05])
    lane.points = np.zeros((2, 1, 3))
    lane.ground = MagicMock()

    kin = MagicMock()
    kin.coordinate_order = ["j1"]

    sim = MagicMock()
    adapter = MagicMock()
    labels = ("M1",)
    q_ref = np.zeros((2, 1))

    cal_res = MagicMock()
    cal_res.qualification_note = "ok"
    cal_res.spec_bytes = b"{}"
    cal_res.hip_report = {}
    cal_res.calibration = {}
    cal_res.calibration2 = {}
    cal_res.address_report = {}

    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli.replay",
        lambda *a, **k: (MagicMock(), np.zeros((2, 1))),
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli.build_dynamics_report",
        lambda *a, **k: ({}, np.zeros((2, 1))),
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli._persist_dynamics_artifacts",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli.build_ground_support_receipt",
        lambda *a, **k: {"base_key": "base_val"},
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli.log_pipeline_summary",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.full_body_forward_dynamics.reference_zmp",
        lambda *a, **k: None,
    )

    stub_summary: dict[str, Any] = {
        "name": "mjx-knots",
        "iterations": 5,
        "port_check_replay_marker_rms_m": 0.065,
        "best_replay_marker_rms_m": 0.055,
        "stop_reason": "completed",
        "receipt": "mjx_optimisation_receipt.json",
    }
    monkeypatch.setattr(
        "src.shared.python.motion_matching.pipeline.cli.run_trajectory_optimiser",
        lambda *a, **kw: stub_summary,
    )

    # 1. Run with trajectory_optimiser = "none"
    out_dir_none = tmp_path / "run_none"
    out_dir_none.mkdir()
    args_none = argparse.Namespace(
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
    ctx_none = PipelineContext(
        args=args_none,
        out_dir=out_dir_none,
        c3d_path=Path("dummy.c3d"),
        engine="mujoco",
        log=log,
        t_start=0.0,
    )

    receipt_none = _simulate_and_receipt(
        ctx_none,
        lane,
        kin,
        sim,
        adapter,
        labels,
        q_ref,
        cal_res,
        {},
        {},
    )
    assert "trajectory_optimiser" not in receipt_none
    receipt_file_none = json.loads(
        (out_dir_none / "receipt.json").read_text(encoding="utf-8")
    )
    assert "trajectory_optimiser" not in receipt_file_none

    # 2. Run with trajectory_optimiser = "mjx-knots"
    out_dir_mjx = tmp_path / "run_mjx"
    out_dir_mjx.mkdir()
    args_mjx = argparse.Namespace(
        trajectory_optimiser="mjx-knots",
        mjx_iterations=5,
        zmp_filter=False,
        shooting_fit=0,
        tracking="kkt",
        backend="mujoco",
        spec="dummy.json",
        recalibrate_upper=False,
        anthropometric=None,
        capture="driver",
    )
    ctx_mjx = PipelineContext(
        args=args_mjx,
        out_dir=out_dir_mjx,
        c3d_path=Path("dummy.c3d"),
        engine="mujoco",
        log=log,
        t_start=0.0,
    )

    receipt_mjx = _simulate_and_receipt(
        ctx_mjx,
        lane,
        kin,
        sim,
        adapter,
        labels,
        q_ref,
        cal_res,
        {},
        {},
    )
    assert receipt_mjx["trajectory_optimiser"] == stub_summary
    receipt_file_mjx = json.loads(
        (out_dir_mjx / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_file_mjx["trajectory_optimiser"] == stub_summary
