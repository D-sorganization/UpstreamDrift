"""Hand backend registry and Shadow humanization (#15122)."""

from __future__ import annotations

import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.hand_backends import shadow
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    default_backend_name,
    list_hand_backends,
)

pytestmark = pytest.mark.unit

mujoco = pytest.importorskip("mujoco")


def test_shadow_backend_registered() -> None:
    assert default_backend_name() == "shadow"
    assert "shadow" in list_hand_backends()


def test_build_hand_visual_returns_skin_material() -> None:
    spec = GripSpec.legacy()
    result = build_hand_visual("shadow", spec, "trail", None, grip_radius_m=0.012)
    assert len(result.mesh.vertices) > 10_000
    assert result.material_rgba[3] == pytest.approx(1.0)
    assert result.material_rgba[0] > result.material_rgba[2]


def test_humanized_mesh_omits_forearm_extent() -> None:
    spec = GripSpec.legacy()
    result = build_hand_visual("shadow", spec, "lead", None, grip_radius_m=0.012)
    z_span = float(result.mesh.vertices[:, 2].max() - result.mesh.vertices[:, 2].min())
    assert z_span < 0.22, "forearm hardware should not dominate the span"


def test_joint_angle_dict_round_trip_wrist() -> None:
    spec = GripSpec.legacy()
    from src.shared.python.model_appearance.grip_pose import Hand

    solved = shadow.solve_shadow_hand_qpos(
        Hand.TRAIL,
        spec.trail,
        grip_spec=spec,
        grip_radius_m=0.012,
    )
    overrides = {"WRJ1": float(solved.qpos[0]), "WRJ2": float(solved.qpos[1])}
    built = shadow.build_hand_visual(spec, "trail", overrides, grip_radius_m=0.012)
    assert built.mesh.volume() > 0.0
