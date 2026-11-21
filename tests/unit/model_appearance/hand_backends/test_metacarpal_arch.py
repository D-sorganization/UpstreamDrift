"""The set metacarpal arch of the skinned hand (#15329, epic #15150).

The joint itself, on the sized hand and without a grip build; a seat that
uses it is judged in ``test_trail_seat_setting.py``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.metacarpal_arch import (
    MetacarpalArchSettings,
    arch_rotation,
    set_metacarpal_arch,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)

pytestmark = pytest.mark.unit

_SETTINGS = default_settings()
_ASSET = skinned_human._sized_assets(0.19, 0.085, _SETTINGS)[0]
_RAW_ARCH = load_settings_mapping()["metacarpal_arch"]
_LIMITS = _RAW_ARCH["limits_deg"]


def _settings(trail: dict[str, dict[str, float]]) -> skinned_human.SkinnedHumanSettings:
    raw = load_settings_mapping()
    raw["metacarpal_arch"] = {**raw["metacarpal_arch"], "trail": trail}
    return settings_from_mapping(raw)


def _ray(flexion_deg: float, ulnar_deviation_deg: float) -> dict[str, float]:
    return {"flexion_deg": flexion_deg, "ulnar_deviation_deg": ulnar_deviation_deg}


def _far_end(span: list[float]) -> float:
    return max(span, key=abs)


#: Every ray at the far end of its ranges, and the steepest arch two rays make.
_AT_LIMITS = {
    ray: _ray(_far_end(limit["flexion"]), _far_end(limit["ulnar_deviation"]))
    for ray, limit in _LIMITS.items()
}
_STEEPEST = {"ring": _ray(13.0, 7.0), "little": _ray(26.0, 13.0)}
_ARCHES = pytest.mark.parametrize("arch", (_AT_LIMITS, _STEEPEST))


def _hand(settings: skinned_human.SkinnedHumanSettings) -> skinned_human._GripHand:
    hand = skinned_human._GripHand(_ASSET, settings)
    set_metacarpal_arch(hand, "trail")
    return hand


def _direction(bone: str) -> np.ndarray:
    i = _ASSET.bone_index(bone)
    d = _ASSET.bone_tails[i] - _ASSET.bone_heads[i]
    return np.asarray(d / np.linalg.norm(d))


def test_the_shipped_arch_is_none_and_leaves_every_hand_as_it_was() -> None:
    arch = _SETTINGS.metacarpal_arch
    assert arch.of("lead") == {} and arch.of("trail") == {}
    for side in ("lead", "trail"):
        assert _SETTINGS.with_set_arch(side) is _SETTINGS
        plain = skinned_human._GripHand(_ASSET, _SETTINGS)
        before = dict(plain.fixed)
        set_metacarpal_arch(plain, side)
        assert plain.fixed.keys() == before.keys()
        assert all(plain.fixed[bone] is before[bone] for bone in before)


def test_the_limits_are_the_cited_joint_ranges() -> None:
    """El-Shennawy et al., J Hand Surg Am 2001;26(6):1030-5: overall arcs of
    the second to fifth carpometacarpal joints."""
    cited = {"index": (11, 2), "middle": (7, 4), "ring": (20, 7), "little": (27, 13)}
    for ray, (flexion, deviation) in cited.items():
        assert _LIMITS[ray]["flexion"] == [0.0, flexion]
        assert max(map(abs, _LIMITS[ray]["ulnar_deviation"])) == deviation
        assert min(map(abs, _LIMITS[ray]["ulnar_deviation"])) == 0.0
    assert "10.1053/jhsu.2001.28761" in _RAW_ARCH["_source"]


def test_a_set_ray_flexes_and_fans_about_its_base_by_the_set_angles() -> None:
    hand = _hand(_settings({"ring": _ray(10.0, 0.0), "little": _ray(20.0, 0.0)}))
    straight = skinned_human._straight_angles()
    i = _ASSET.bone_index("meta_little")
    turn = hand.transforms(straight)[i]
    base = _ASSET.bone_heads[i]
    assert np.allclose(turn[:3, :3] @ base + turn[:3, 3], base, atol=1e-12)
    along = turn[:3, :3] @ _direction("meta_little")
    assert math.degrees(math.acos(along @ _direction("meta_little"))) == pytest.approx(
        20.0
    )
    assert along[2] > _direction("meta_little")[2], "flexion is toward the palm"
    fanned = _hand(_settings({"ring": _ray(0.0, 7.0), "little": _ray(0.0, 7.0)}))
    j = _ASSET.bone_index("meta_ring")
    along = fanned.transforms(straight)[j][:3, :3] @ _direction("meta_ring")
    assert along[0] < _direction("meta_ring")[0], "ulnar is away from the thumb"
    assert abs(along[2] - _direction("meta_ring")[2]) < 5e-3, "it stays in the palm"
    # The finger's hinge turns with its ray, in the plane of the palm.
    k = _ASSET.bone_index("ring_mid")
    plain = skinned_human._GripHand(_ASSET, _SETTINGS).transforms(straight)
    before, after = (
        world[k][:3, :3] @ _ASSET.flex_axes[k]
        for world in (plain, fanned.transforms(straight))
    )
    gained = math.degrees(
        math.atan2(after[1], after[0]) - math.atan2(before[1], before[0])
    )
    assert 6.5 < gained <= 7.0, "the ray leans out of the palm's plane a little"


@_ARCHES
def test_no_ray_and_no_finger_twists_about_its_own_length(arch: dict) -> None:
    hand = _hand(_settings(arch))
    world = hand.transforms(skinned_human._straight_angles())
    for ray in arch:
        meta, prox = (_ASSET.bone_index(b) for b in (f"meta_{ray}", f"{ray}_prox"))
        turn = Rotation.from_matrix(world[meta][:3, :3]).as_rotvec()
        assert np.linalg.norm(turn) > 0.0
        assert abs(turn @ _direction(f"meta_{ray}")) < 1e-12
        # Each finger keeps its place on its own metacarpal: the arch moves
        # the ray, never a knuckle.
        plain = skinned_human._GripHand(_ASSET, _SETTINGS)
        own = plain.transforms(skinned_human._straight_angles())
        relative = np.linalg.inv(world[meta]) @ world[prox]
        assert np.allclose(relative, np.linalg.inv(own[meta]) @ own[prox], atol=1e-12)


def test_a_set_ray_gives_up_the_arch_coupled_to_its_knuckle() -> None:
    settings = _settings({"ring": _ray(5.0, 0.0)})
    assert settings.with_set_arch("lead") is settings
    held = settings.with_set_arch("trail")
    assert dict(held.arch_gain)["ring"] == 0.0
    assert dict(held.arch_gain)["little"] == dict(settings.arch_gain)["little"]
    knuckles = {f"{finger}_prox": math.radians(80.0) for finger in FINGERS}
    arch = skinned_human._arch_from_knuckles(knuckles, held)
    assert arch["meta_ring"] == 0.0 and arch["meta_little"] > 0.0


@pytest.mark.parametrize(
    "trail",
    (
        {"ring": _ray(13.0, 0.0), "little": _ray(27.5, 0.0)},  # past the range
        {"ring": _ray(0.0, -1.0)},  # leans the way that folds the skin
        {"index": _ray(-1.0, 0.0)},
        {"ring": _ray(20.0, 0.0), "little": _ray(27.0, 0.0)},  # 20 beside none
        {"little": _ray(0.0, 13.0)},  # 13 beside none
        {"little": _ray(math.nan, 0.0)},
        {"thumb": _ray(1.0, 0.0)},
        {"ring": {"flexion_deg": 1.0}},
    ),
)
def test_an_arch_the_joints_or_the_skin_cannot_take_is_refused(trail: dict) -> None:
    with pytest.raises(ValueError):
        _settings(trail)


def test_bad_arch_inputs_are_refused() -> None:
    arch = _settings(_AT_LIMITS).metacarpal_arch
    with pytest.raises(ValueError):
        arch.of("left")
    with pytest.raises(ValueError):
        arch_rotation(_ASSET, "thumb", 0.1, 0.0)
    with pytest.raises(ValueError):
        arch_rotation(_ASSET, "ring", math.inf, 0.0)
    raw = copy.deepcopy(_RAW_ARCH)
    raw["limits_deg"]["ring"]["flexion"] = [1.0, 20.0]
    with pytest.raises(ValueError):
        MetacarpalArchSettings.from_mapping(raw)
    raw = copy.deepcopy(_RAW_ARCH)
    raw["max_neighbour_step_deg"]["flexion"] = -1.0
    with pytest.raises(ValueError):
        MetacarpalArchSettings.from_mapping(raw)
