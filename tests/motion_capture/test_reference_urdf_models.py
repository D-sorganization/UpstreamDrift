"""URDF import preserves actual kinematics instead of relabeling a fallback."""

from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reference.urdf_models import load_urdf_model
from src.motion_capture.reconstruct.model import ArticulatedModel

pytestmark = pytest.mark.unit


def test_arbitrary_axis_origin_and_fixed_tip(tmp_path: Path) -> None:
    path = tmp_path / "arm.urdf"
    path.write_text("""<robot name="arm"><link name="base"/><link name="arm"/><link name="tip"/>
    <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 1 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/></joint>
    <joint name="fixed" type="fixed"><parent link="arm"/><child link="tip"/>
    <origin xyz="1 0 0"/></joint></robot>""")
    registered = load_urdf_model(
        path, name="test-arm", landmark_map={"base": "neck", "tip": "left_wrist"}
    )
    model = ArticulatedModel(registered.spec)
    q = np.zeros((1, model.n_dof))
    q[0, -1] = np.pi / 2
    points = model.forward(q)
    np.testing.assert_allclose(points[0, -1], [0, 1, -1], atol=1e-12)
    assert registered.spec.joints[-2].limits_rad == ((-2, 2),)


def test_unsupported_joint_is_not_silently_frozen(tmp_path: Path) -> None:
    path = tmp_path / "slide.urdf"
    path.write_text("""<robot name="slide"><link name="base"/><link name="tip"/>
    <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
    <axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/>
    </joint></robot>""")
    with pytest.raises(ValueError, match="prismatic"):
        load_urdf_model(path, name="slide", landmark_map={"tip": "left_wrist"})


def test_unknown_link_mapping_rejected(tmp_path: Path) -> None:
    path = tmp_path / "one.urdf"
    path.write_text('<robot name="one"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="landmark"):
        load_urdf_model(path, name="one", landmark_map={"typo": "neck"})


def test_external_world_floating_joint_uses_absolute_fitted_root(
    tmp_path: Path,
) -> None:
    path = tmp_path / "floating.urdf"
    path.write_text("""<robot name="free"><link name="base"/>
    <joint name="free" type="floating"><parent link="world"/><child link="base"/>
    <origin xyz="0 0 1"/></joint></robot>""")
    result = load_urdf_model(path, name="free", landmark_map={"base": "mid_hip"})
    model = ArticulatedModel(result.spec)
    assert model.n_dof == 6
    q = np.zeros((1, 6))
    q[0, :3] = [1, 2, 3]
    np.testing.assert_allclose(model.forward(q)[0, 0], [1, 2, 3])
