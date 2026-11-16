"""Shadow backend on the shared hand registry (#15122, ADR-0057 A3)."""

from __future__ import annotations

import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.hand_backends import (
    HandJointAngles,
    build_hand_visual,
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends import shadow
from src.shared.python.model_appearance.hand_backends.shadow_solver import (
    shadow_joint_name,
)

pytestmark = pytest.mark.unit

mujoco = pytest.importorskip("mujoco")


def test_shadow_backend_registered_and_enabled() -> None:
    assert "shadow" in list_hand_visual_backends()


def test_build_hand_visual_returns_skin_colour() -> None:
    spec = GripSpec.legacy()
    result = build_hand_visual(spec, "trail", grip_radius_m=0.012, backend="shadow")
    assert result.backend == "shadow"
    assert len(result.mesh.vertices) > 10_000
    assert result.skin_rgb[0] > result.skin_rgb[2]


def test_humanized_mesh_omits_forearm_extent() -> None:
    spec = GripSpec.legacy()
    result = build_hand_visual(spec, "lead", grip_radius_m=0.012, backend="shadow")
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


@pytest.mark.parametrize(
    ("canonical", "native"),
    [
        ("index_prox", "FFJ3"),
        ("middle_mid", "MFJ2"),
        ("little_dist", "LFJ1"),
        ("thumb_press", "THJ4"),
        ("thumb_dist", "THJ1"),
        ("WRJ1", "WRJ1"),
    ],
)
def test_canonical_joint_names_map_to_shadow(canonical: str, native: str) -> None:
    assert shadow_joint_name(canonical) == native


def test_canonical_joint_angles_curl_the_hand() -> None:
    spec = GripSpec.legacy()
    straight = shadow.build_hand_visual(
        spec, "lead", HandJointAngles.from_mapping({}), grip_radius_m=0.012
    )
    curled = shadow.build_hand_visual(
        spec,
        "lead",
        HandJointAngles.from_mapping({"index_prox": 1.2, "index_mid": 1.2}),
        grip_radius_m=0.012,
    )
    assert not (
        straight.mesh.vertices.shape == curled.mesh.vertices.shape
        and (straight.mesh.vertices == curled.mesh.vertices).all()
    )
