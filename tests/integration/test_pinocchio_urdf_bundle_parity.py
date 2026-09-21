"""Integration test for Pinocchio model bundle parity and precision qualification.

Tests MV-01:
- Pinocchio direct-spec vs URDF bundle named FK and mass matrices agree within
  strict scale-aware numerical tolerances.
- Separates translational error (metres) from rotational SO(3) difference
  and mass matrix mixed-unit difference.
- Verifies machine-precision preservation (< 1e-12) achieved by 17-digit float serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_generation.export.model_bundle import export_model_bundle

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]


@pytest.fixture
def calibrated_spec_path() -> Path:
    """Path to the 44-coordinate calibrated full-body driver spec."""
    path = (
        Path(__file__).parents[2]
        / "docs"
        / "development"
        / "full_body_models"
        / "evidence"
        / "ground_support"
        / "anthro_driver"
        / "full_body_spec_hipcal_scaled.json"
    )
    if not path.exists():
        pytest.skip(f"Calibrated spec fixture not found at {path}")
    return path


def test_pinocchio_urdf_bundle_fk_and_mass_parity(
    calibrated_spec_path: Path,
) -> None:
    """Verify native Pinocchio direct-spec vs URDF bundle FK and mass matrix parity."""
    pin = pytest.importorskip(
        "pinocchio", reason="pinocchio not installed in this environment"
    )

    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    spec_bytes = calibrated_spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    bundle = export_model_bundle(spec_bytes)

    m = pin.buildModelFromXML(bundle.urdf_xml)
    d = m.createData()
    native = FullBodyPinocchioModel(spec)

    assert m.nq == 44
    assert m.nv == 44

    coord_order = spec["coordinate_order"]
    qidx = [m.joints[m.getJointId(n)].idx_q for n in coord_order]
    vidx = [m.joints[m.getJointId(n)].idx_v for n in coord_order]
    nidx = [native.model.joints[native.model.getJointId(n)].idx_v for n in coord_order]

    # Test neutral pose and several perturbated configurations
    rng = np.random.default_rng(42)
    test_configs = [np.zeros(44)]
    for _ in range(5):
        test_configs.append(rng.uniform(-0.5, 0.5, size=44))

    max_trans_err = 0.0
    max_rot_err = 0.0
    max_mass_err = 0.0

    meta = bundle.sidecar or {}
    frame_links = meta.get("frame_links", {})

    for q_arr in test_configs:
        state = dict(zip(coord_order, q_arr, strict=True))
        q = pin.neutral(m)
        q[qidx] = q_arr

        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        poses = native.frame_poses(state)

        for name, link in frame_links.items():
            h_native = poses[name]
            h_urdf = d.oMf[m.getFrameId(link)].homogeneous

            trans_err = float(np.linalg.norm(h_native[:3, 3] - h_urdf[:3, 3]))
            rot_err = float(np.linalg.norm(h_native[:3, :3] - h_urdf[:3, :3]))

            max_trans_err = max(max_trans_err, trans_err)
            max_rot_err = max(max_rot_err, rot_err)

        mass_native = native.mass_matrix(state)[np.ix_(nidx, nidx)]
        mass_urdf = np.array(pin.crba(m, d, q))[np.ix_(vidx, vidx)]
        mass_err = float(np.max(np.abs(mass_native - mass_urdf)))
        max_mass_err = max(max_mass_err, mass_err)

    # Scale-aware tolerances (machine precision with 17-digit float serialization)
    assert max_trans_err < 1e-12, f"Max translation error exceeded: {max_trans_err} m"
    assert max_rot_err < 1e-12, f"Max rotation error exceeded: {max_rot_err}"
    assert max_mass_err < 1e-11, f"Max mass matrix error exceeded: {max_mass_err}"
