"""Unit tests for MJX trajectory optimisation evidence CLI (#11046)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

import numpy as np

from docs.development.full_body_models.evidence.ground_support.mjx_trajectory_optimisation import (
    main,
)
from tests.unit.engines.mujoco.test_mjx_tracking_plant import (
    TOY_TRACKING_XML,
    _sample_contact,
)

pytestmark = pytest.mark.unit


def test_mjx_evidence_cli_iterations_2(tmp_path: Path) -> None:
    """Run CLI main with --iterations 2 on a tiny package and verify receipt assertions."""
    # 1. Write mjx_package.xml
    (tmp_path / "mjx_package.xml").write_text(TOY_TRACKING_XML, encoding="utf-8")

    # 2. Write mjx_package.json
    meta = {
        "rate_hz": 50.0,
        "coordinate_order": ["TranslationInputZ", "joint_1", "joint_2"],
        "controller": {"omega_rad_s": 60.0, "zeta": 1.0},
        "contact": _sample_contact().as_document(),
        "mass_kg": 3.0,
        "baseline": {"replay_marker_rms_m": 0.05},
    }
    (tmp_path / "mjx_package.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # 3. Write mjx_package.npz
    n_frames = 11
    times = np.linspace(0.0, 0.2, n_frames)
    q_track = np.zeros((n_frames, 3), dtype=np.float32)
    q_track[:, 0] = 0.5  # TranslationInputZ
    q_track[:, 1] = 0.1  # joint_1
    q_track[:, 2] = -0.1  # joint_2

    # Targets: slightly perturbed so cost > 0
    targets = np.zeros((n_frames, 2, 3), dtype=np.float32)
    targets[:, 0, :] = [0.0, 0.0, 0.55]
    targets[:, 1, :] = [0.0, 0.0, 0.10]
    valid = np.ones((n_frames, 2), dtype=bool)

    np.savez(
        tmp_path / "mjx_package.npz",
        time_s=times,
        q_track=q_track,
        targets_m=targets,
        valid=valid,
        qpos_adr=np.array([0, 1, 2], dtype=np.int32),
        dof_adr=np.array([0, 1, 2], dtype=np.int32),
        root_mask=np.array([True, False, False], dtype=bool),
        sphere_site_ids=np.array([1], dtype=np.int32),
        sphere_body_ids=np.array([3], dtype=np.int32),
        sphere_radii_m=np.array([0.05], dtype=np.float32),
        marker_body_ids=np.array([1, 3], dtype=np.int32),
        marker_local_m=np.array(
            [[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]], dtype=np.float32
        ),
        ground_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ground_height_m=np.array(0.0, dtype=np.float32),
    )

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
