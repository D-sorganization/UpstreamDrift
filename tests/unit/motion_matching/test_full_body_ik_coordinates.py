"""TDD Red test for Issue #10140: Align MuJoCo full-body IK coordinates with named dynamics coordinates.

Verifies that MujocoFullBodyIK.pose_fn maps declared coordinates by joint name
rather than copying q[i] directly into qpos[i], guaranteeing exact FK and dual-grip
closure parity with NativeMujocoFullBodyModel on nonsymmetric coordinate vectors.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_ik import MujocoFullBodyIK
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.full_body_spec import (
    load_full_body_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL_BODY_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture
def upper_spec() -> dict:
    return json.loads(UPPER_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fb_spec(upper_spec: dict) -> dict:
    return load_full_body_spec(FULL_BODY_PATH, upper_spec)


def test_mujoco_ik_coordinate_order_matches_dynamics_model(fb_spec: dict) -> None:
    """MujocoFullBodyIK.pose_fn and NativeMujocoFullBodyModel must yield identical FK.

    Nonsymmetric coordinate values ensure any permutation between declared coordinate
    order and native qpos order is immediately exposed.
    """
    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    model = NativeMujocoFullBodyModel(spec_bytes)
    ik = MujocoFullBodyIK(spec_bytes)

    names = model.coordinate_order
    assert len(names) == 41

    # Verify that declared order genuinely differs from native qpos order
    native = model.model
    addresses = [(int(native.joint(name).qposadr[0]), name) for name in names]
    native_order = [name for _, name in sorted(addresses)]
    assert names != native_order, (
        "Declared order unexpectedly matches native qpos order"
    )

    # Distinct nonsymmetric joint values: 0.01, 0.02, ..., 0.41
    q = np.linspace(0.01, 0.41, len(names))
    coords_dict = dict(zip(names, q, strict=True))

    model_poses = model.frame_poses(coords_dict)
    ik_poses = ik.pose_fn(q)

    # 1. Compare frame sites present in both model.metadata["frame_sites"] and ik.marker_bodies
    common_frames = set(model.metadata["frame_sites"]).intersection(ik.marker_bodies)
    assert len(common_frames) > 0
    for frame_name in sorted(common_frames):
        assert frame_name in model_poses
        assert frame_name in ik_poses
        r_model, t_model = (
            model_poses[frame_name][:3, :3],
            model_poses[frame_name][:3, 3],
        )
        r_ik, t_ik = ik_poses[frame_name]
        np.testing.assert_allclose(
            t_ik,
            t_model,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Translation mismatch at frame {frame_name}",
        )
        np.testing.assert_allclose(
            r_ik,
            r_model,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Rotation mismatch at frame {frame_name}",
        )

    # 2. Compare bodies in marker_bodies that are direct bodies (not frame sites)
    body_markers = set(ik.marker_bodies) - set(model.metadata["frame_sites"])
    assert len(body_markers) > 0
    for body_name in sorted(body_markers):
        assert body_name in ik_poses
        bid = model.model.body(body_name).id
        r_model = model.data.xmat[bid].reshape((3, 3))
        t_model = model.data.xpos[bid]
        r_ik, t_ik = ik_poses[body_name]
        np.testing.assert_allclose(
            t_ik,
            t_model,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Translation mismatch at body {body_name}",
        )
        np.testing.assert_allclose(
            r_ik,
            r_model,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Rotation mismatch at body {body_name}",
        )

    # Closure residuals between dual-grip weld sites must match
    model.accelerations(
        coords_dict,
        dict.fromkeys(names, 0.0),
        dict.fromkeys(names, 0.0),
    )
    model_closure_disp, _ = model.closure_errors()
    ik_closure_disp = ik.closure_residuals(q)

    np.testing.assert_allclose(
        ik_closure_disp,
        model_closure_disp[:3],
        rtol=1e-10,
        atol=1e-10,
        err_msg="Closure residual mismatch between IK and dynamics model",
    )
