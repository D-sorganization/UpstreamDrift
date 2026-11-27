"""The fold joint across the palm (#15270, epic #15150).

The joint itself, on the sized hand and without a pair build; the grip it
gives is judged in ``test_palm_on_thumb.py``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.palm_fold import (
    FoldLine,
    PalmFoldSettings,
    close_fold,
    fold_line,
    fold_palm,
    fold_weights,
    folded,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)

pytestmark = pytest.mark.unit

_SETTINGS = default_settings()
_FOLD = dataclasses.replace(_SETTINGS.pair_palm_fold, max_deg=30.0, press_m=0.002)
_ASSET = skinned_human._sized_assets(0.19, 0.085, _SETTINGS)[0]
_ANGLE = math.radians(20.0)


def _thumb(asset=_ASSET) -> np.ndarray:
    """A thumb lying on the palm, heel to the base of the middle finger, on
    the little-finger side of the hand's own thumb."""
    along = np.linspace(0.0, 1.0, 40)[:, None]
    start, end = np.array([-0.030, 0.020, 0.030]), np.array([-0.005, 0.070, 0.030])
    ring = 0.008 * np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    return np.vstack([start + along * (end - start) + offset for offset in ring])


def _line(settings: PalmFoldSettings = _FOLD) -> FoldLine:
    return fold_line(_ASSET, _thumb(), settings)


def test_the_hinge_lies_along_the_thumb_on_the_side_of_the_thumb_base() -> None:
    line = _line()
    assert line.direction @ np.array([0.025, 0.050, 0.0]) > 0.99 * math.hypot(
        0.025, 0.050
    )
    assert line.direction[2] == 0.0 and line.side[2] == 0.0
    assert abs(line.direction @ line.side) < 1e-12
    base = _ASSET.bone_heads[_ASSET.bone_index("thumb_meta")]
    assert (base - line.point) @ line.side > 0.0
    centre = _thumb().mean(axis=0)
    assert (line.point - centre) @ line.side == pytest.approx(_FOLD.offset_m)
    assert centre[2] - line.point[2] == pytest.approx(_FOLD.below_thumb_m)


def test_only_the_thenar_side_of_the_palm_folds() -> None:
    line = _line()
    weights = fold_weights(_ASSET, line, _FOLD)
    assert weights.min() >= 0.0 and weights.max() <= 1.0 + 1e-12
    rest = _ASSET.rest_vertices
    far_side = (rest - line.point) @ line.side <= 0.0
    palm = np.setdiff1d(  # palm skin that no thumb bone pulls on
        _ASSET.dominant_vertex_ids(_FOLD.bones),
        _ASSET.influenced_vertex_ids(("thumb_meta", "thumb_prox", "thumb_dist")),
    )
    assert np.all(weights[palm][far_side[palm]] == 0.0)
    assert weights[palm].max() > 0.9, "no palm skin takes the whole fold"
    fingers = _ASSET.dominant_vertex_ids(
        tuple(
            f"{f}_{s}"
            for f in ("index", "middle", "ring", "little")
            for s in ("mid", "dist")
        )
    )
    assert np.all(weights[fingers] == 0.0), "the fold reaches the fingers"


def test_no_fold_is_the_same_hand() -> None:
    assert folded(_ASSET, _line(), 0.0, _FOLD) is _ASSET


def test_the_fold_brings_the_thenar_side_toward_the_palm_side() -> None:
    line = _line()
    out = folded(_ASSET, line, _ANGLE, _FOLD)
    weights = fold_weights(_ASSET, line, _FOLD)
    moved = out.rest_vertices - _ASSET.rest_vertices
    still = weights == 0.0
    assert np.array_equal(out.rest_vertices[still], _ASSET.rest_vertices[still])
    whole = np.flatnonzero(weights > 0.99)
    assert len(whole) > 20
    arm = (_ASSET.rest_vertices[whole] - line.point) @ line.side
    # Palmar is +z: skin further out on the fold side rises more.
    assert np.all(moved[whole, 2] > 0.5 * arm * math.sin(_ANGLE) - 1e-4)
    assert np.array_equal(out.faces, _ASSET.faces)
    assert np.array_equal(out.joint_weights, _ASSET.joint_weights)
    assert np.array_equal(out.joint_indices, _ASSET.joint_indices)


def test_the_thumb_rides_on_the_fold_and_keeps_its_skin() -> None:
    line = _line()
    out = folded(_ASSET, line, _ANGLE, _FOLD)
    i, j = (_ASSET.bone_index(b) for b in ("thumb_meta", "thumb_dist"))
    assert np.linalg.norm(out.bone_heads[i] - _ASSET.bone_heads[i]) > 1e-3
    # Bones keep their lengths and the thumb's skin its place on its bone.
    for b in ("thumb_meta", "thumb_prox", "thumb_dist"):
        k = _ASSET.bone_index(b)
        assert np.linalg.norm(out.bone_tails[k] - out.bone_heads[k]) == pytest.approx(
            np.linalg.norm(_ASSET.bone_tails[k] - _ASSET.bone_heads[k])
        )
        assert np.linalg.norm(out.flex_axes[k]) == pytest.approx(
            np.linalg.norm(_ASSET.flex_axes[k])
        )
    tip = _ASSET.dominant_vertex_ids(("thumb_dist",))
    dense = _ASSET.dense_weights()
    tip = tip[
        dense[tip][:, [i, _ASSET.bone_index("thumb_prox"), j]].sum(axis=1) > 0.999
    ]
    assert len(tip) > 10
    was = np.linalg.norm(_ASSET.rest_vertices[tip] - _ASSET.bone_tails[j], axis=1)
    now = np.linalg.norm(out.rest_vertices[tip] - out.bone_tails[j], axis=1)
    assert np.abs(now - was).max() < 2e-4
    # The other bones are where they were.
    k = _ASSET.bone_index("index_prox")
    assert np.array_equal(out.bone_heads[k], _ASSET.bone_heads[k])


def test_the_fold_closes_until_the_pad_presses_and_no_further() -> None:
    step = math.radians(_FOLD.step_deg)

    def gap(angle: float) -> float:  # 1 mm clear, closing 1 mm per step
        return 0.001 - 0.001 * angle / step

    # Gaps by step: 0, -1, -2 (the limit), -3 mm: three steps close.
    assert close_fold(_FOLD, gap) == pytest.approx(3 * step)
    assert close_fold(_FOLD, lambda angle: 1.0) == pytest.approx(
        math.radians(_FOLD.max_deg)
    )
    # A palm that already presses in may press press_m further, not more.
    assert close_fold(_FOLD, lambda angle: -0.004 - 0.0015 * angle / step) == (
        pytest.approx(step)
    )
    off = dataclasses.replace(_FOLD, max_deg=0.0)
    assert close_fold(off, lambda angle: 1.0) == 0.0


def _hand(fold: PalmFoldSettings) -> skinned_human._GripHand:
    settings = dataclasses.replace(_SETTINGS, pair_palm_fold=fold)
    return skinned_human._GripHand(_ASSET, settings)


def test_a_posed_hand_folds_round_a_thumb_and_stops_on_what_is_under_it() -> None:
    angles = skinned_human._straight_angles()
    thumb = _thumb()

    def nothing(points: np.ndarray) -> np.ndarray:
        return np.full(len(points), 1.0)

    hand = _hand(_FOLD)
    hand.swing_sideways("index_prox", 0.05)
    out, skin = fold_palm(hand, _ASSET, angles, thumb, nothing)
    assert out is not hand
    assert np.array_equal(out.asset.rest_vertices, skin.rest_vertices)
    assert out.side == hand.side and out.placement is hand.placement
    line = _line()
    whole = folded(_ASSET, line, math.radians(_FOLD.max_deg), _FOLD)
    assert np.allclose(skin.rest_vertices, whole.rest_vertices)

    def floor(points: np.ndarray) -> np.ndarray:  # a plate 6 mm over the palm
        return 0.028 - points[:, 2]

    low, _ = fold_palm(_hand(_FOLD), _ASSET, angles, thumb, floor)
    rise = low.asset.rest_vertices[:, 2] - _ASSET.rest_vertices[:, 2]
    assert 0.0 < rise.max() < (skin.rest_vertices - _ASSET.rest_vertices)[:, 2].max()

    same = _hand(dataclasses.replace(_FOLD, max_deg=0.0))
    assert fold_palm(same, _ASSET, angles, thumb, nothing) == (same, _ASSET)
    assert fold_palm(hand, _ASSET, angles, None, nothing) == (hand, _ASSET)


@pytest.mark.parametrize(
    "change",
    [
        {"max_deg": -1.0},
        {"max_deg": 90.0},
        {"step_deg": 0.0},
        {"offset_m": -0.001},
        {"blend_m": 0.0},
        {"press_m": -0.001},
        {"below_thumb_m": math.nan},
        {"fade": (0.5, 0.5)},
        {"bones": ()},
        {"carried": ()},
        {"thumb_bones": ()},
    ],
)
def test_bad_fold_settings_are_refused(change: dict) -> None:
    with pytest.raises(ValueError, match="palm fold"):
        dataclasses.replace(_FOLD, **change)


def test_bad_fold_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="thumb points"):
        fold_line(_ASSET, np.zeros((2, 3)), _FOLD)
    upright = np.array([[0.0, 0.0, z] for z in (0.01, 0.02, 0.03, 0.04)])
    with pytest.raises(ValueError, match="no line to fold on"):
        fold_line(_ASSET, upright, _FOLD)
    with pytest.raises(ValueError, match="names no bone"):
        fold_weights(_ASSET, _line(), dataclasses.replace(_FOLD, bones=("palm",)))
    with pytest.raises(ValueError, match="finite"):
        folded(_ASSET, _line(), math.inf, _FOLD)
