"""Palm-on-thumb allocation plumbing in the bundle overlay provider (HRG-H1, #13189)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.mujoco_axial_loads import GripClosure
from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.shared.python.grip_contact.palm_thumb import allocate_from_closure
from tests.unit.force_overlay.test_axial_closure_provider import _bundle, _Kin

pytestmark = pytest.mark.unit

PATCH = type(
    "P",
    (),
    {
        "point_club_m": (0.0, 0.0, 0.1),
        "normal_club": (0.0, 1.0, 0.0),
        "trail_side": "R",
    },
)()


class _Source:
    closing_hand_side = "R"

    def __init__(self, available: bool = True) -> None:
        self.available = available

    def closure_wrench(self, coordinates, rates, primitive_efforts):
        if not self.available:
            return GripClosure.unavailable("synthetic: no solve")
        return GripClosure.from_multiplier(
            body_a="hand",
            body_b="club",
            point_a_m=(0, 0, 0),
            point_b_m=(0, 0, 0),
            multiplier=(0, 5, 0, 0, 0, 0),
        )

    def club_frame_world(self, coordinates):
        return np.eye(3), np.zeros(3)


def _provider(source) -> BundleOverlayProvider:
    kin = _Kin()
    return BundleOverlayProvider(_bundle(), kin, kin, engine="mujoco").with_palm_thumb(
        source, PATCH, static_friction=0.9
    )


def test_frame_metadata_carries_the_allocated_force_with_its_label() -> None:
    meta = _provider(_Source()).frame_at(2).metadata["palm_thumb"]
    assert meta["method"] == "allocation" and meta["available"] is True
    assert meta["normal_force_n"] == pytest.approx(5.0)
    assert meta["unloaded"] is False


def test_unavailable_closure_reports_none_and_the_reason() -> None:
    meta = _provider(_Source(available=False)).frame_at(1).metadata["palm_thumb"]
    assert meta["available"] is False and meta["reason"] == "synthetic: no solve"
    assert meta["normal_force_n"] is None and meta["force_on_lead_n"] is None


def test_closing_side_must_match_the_patch_trail_side() -> None:
    source = _Source()
    source.closing_hand_side = "L"
    with pytest.raises(ValueError, match="closing hand"):
        _provider(source)


def test_a_source_without_the_accessors_is_rejected() -> None:
    with pytest.raises(TypeError, match="closure_wrench"):
        kin = _Kin()
        BundleOverlayProvider(_bundle(), kin, kin, engine="mujoco").with_palm_thumb(
            object(), PATCH, static_friction=0.9
        )


def test_without_the_hook_the_frame_has_no_palm_thumb_entry() -> None:
    kin = _Kin()
    frame = BundleOverlayProvider(_bundle(), kin, kin, engine="mujoco").frame_at(1)
    assert "palm_thumb" not in frame.metadata


def test_allocate_from_closure_uses_the_club_side_wrench() -> None:
    closure = GripClosure.from_multiplier(
        body_a="hand",
        body_b="club",
        point_a_m=(0, 0, 0),
        point_b_m=(0, 0, 0),
        multiplier=(0, 5, 0, 0, 0, 0),
    )
    out = allocate_from_closure(
        closure,
        club_rotation=np.eye(3),
        club_origin=np.zeros(3),
        patch=PATCH,
        static_friction=0.9,
    )
    assert out.force_on_lead_n == pytest.approx((0.0, -5.0, 0.0))


def test_loaded_frame_carries_one_labelled_force_wrench_on_the_club() -> None:
    from src.shared.python.grip_contact.palm_thumb import PALM_THUMB_LABEL

    frame = _provider(_Source()).frame_at(2)
    wrench = next(w for w in frame.wrenches if w.label == PALM_THUMB_LABEL)
    assert wrench.body == "club" and wrench.torque_nm is None
    # The palm pushes the thumb along -normal with the normal force (5 N).
    assert wrench.force_n == pytest.approx((0.0, -5.0, 0.0))
    assert wrench.point_m == pytest.approx((0.0, 0.0, 0.1))
    assert "allocation" in wrench.source


def test_unavailable_closure_adds_no_wrench() -> None:
    from src.shared.python.grip_contact.palm_thumb import PALM_THUMB_LABEL

    frame = _provider(_Source(available=False)).frame_at(1)
    assert all(w.label != PALM_THUMB_LABEL for w in frame.wrenches)
