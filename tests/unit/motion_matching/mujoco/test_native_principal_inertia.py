"""MJCF inertia is emitted in principal form, exact to round-off (#11607)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_mjcf import _principal_inertia

pytestmark = pytest.mark.unit


def test_principal_inertia_reconstructs_an_elongated_tensor() -> None:
    mujoco = pytest.importorskip("mujoco")
    rng = np.random.default_rng(7)
    axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    tensor = axes @ np.diag([1e-4, 1.5e-2, 1.5e-2]) @ axes.T  # club-like ratio
    attrs = _principal_inertia(np.zeros(3), tensor)
    quat = np.array(attrs["quat"].split(), dtype=float)
    moments = np.array(attrs["diaginertia"].split(), dtype=float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat / np.linalg.norm(quat))
    rot = rot.reshape(3, 3)
    rebuilt = rot @ np.diag(moments) @ rot.T
    assert np.abs(rebuilt - tensor).max() <= 1e-15 * np.abs(tensor).max() * 10
