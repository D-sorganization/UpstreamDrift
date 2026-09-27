"""Shared helper to write a minimal toy MJX package for unit tests (#11049)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.unit.engines.mujoco.test_mjx_tracking_plant import (
    TOY_TRACKING_XML,
    _sample_contact,
)

# The tracking-plant toy model plus one weld, so tests can see
# load_mjx_package strip the equality block.
TOY_PACKAGE_XML = TOY_TRACKING_XML.replace(
    "  <worldbody>",
    "  <equality>\n"
    '    <weld body1="shin" body2="thigh"/>\n'
    "  </equality>\n"
    "  <worldbody>",
    1,
)


def write_toy_package(
    package_dir: Path,
    *,
    xml: str = TOY_PACKAGE_XML,
    n_frames: int = 11,
) -> Path:
    """Write a minimal toy MJX package directory for testing."""
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "mjx_package.xml").write_text(xml, encoding="utf-8")

    meta = {
        "rate_hz": 50.0,
        "coordinate_order": ["TranslationInputZ", "joint_1", "joint_2"],
        "controller": {"omega_rad_s": 60.0, "zeta": 1.0},
        "contact": _sample_contact().as_document(),
        "mass_kg": 3.0,
        "baseline": {"replay_marker_rms_m": 0.05},
    }
    (package_dir / "mjx_package.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    times = np.linspace(0.0, 0.2, n_frames)
    q_track = np.zeros((n_frames, 3), dtype=np.float32)
    q_track[:, 0] = 0.5  # TranslationInputZ
    q_track[:, 1] = 0.1  # joint_1
    q_track[:, 2] = -0.1  # joint_2

    targets = np.zeros((n_frames, 2, 3), dtype=np.float32)
    targets[:, 0, :] = [0.0, 0.0, 0.55]
    targets[:, 1, :] = [0.0, 0.0, 0.10]
    valid = np.ones((n_frames, 2), dtype=bool)

    np.savez(
        package_dir / "mjx_package.npz",
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
    return package_dir
