"""Ground reaction and force kinds by ZTCF/ZVCF case (#15150 Z4c).

The owner asked for the ground reaction split into its ZTCF and ZVCF parts
and drawn with each case, so a case's render shows only that case: its
shading, its hand forces and its ground reaction. ZTCF has ``tau = 0``, so
no actuator torques are drawn; ZVCF has no gravity, so no weight arrow.
Each foot's part reaction is drawn at its centre of pressure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.mujoco_axial_loads import BodyWrench
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.palette import get_kind_rgba, hex_to_rgba
from src.shared.python.force_overlay.tip_to_tail import TipToTailStyle
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.ztcf_axial import (
    FORCE_NOTES,
    centre_of_pressure,
    ground_arrows,
    case_force_key,
    ground_readout,
    net_ground,
    part_ground,
    with_part_wrenches,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def test_centre_of_pressure_moves_the_force_until_the_tilting_moment_vanishes() -> None:
    # F = 100 N up at the origin with M = (10, -20, 0) N m: the force acts
    # 0.2 m along x and 0.1 m along y (r x F = M).
    cop = centre_of_pressure((0.0, 0.0, 0.0), (0.0, 0.0, 100.0), (10.0, -20.0, 0.0))
    assert cop == pytest.approx((0.2, 0.1, 0.0))
    r = np.subtract(cop, (0.0, 0.0, 0.0))
    moment = np.cross(r, (0.0, 0.0, 100.0))
    assert moment[:2] == pytest.approx((10.0, -20.0))


def test_a_force_without_vertical_load_stays_at_its_point() -> None:
    cop = centre_of_pressure((1.0, 2.0, 0.0), (50.0, 0.0, 1e-6), (3.0, 4.0, 0.0))
    assert cop == pytest.approx((1.0, 2.0, 0.0))


def _bw(fz: float, mx: float = 0.0) -> BodyWrench:
    return BodyWrench("calcn_r", (0.0, 0.0, 0.0), (0.0, 0.0, fz), (mx, 0.0, 0.0))


def test_the_ground_parts_add_to_the_total() -> None:
    parts = part_ground((_bw(300.0, 3.0),), (_bw(500.0, 5.0),))
    assert set(parts) == {"ztcf", "zvcf", "total"}
    (total,) = parts["total"]
    assert total.force_n == pytest.approx((0.0, 0.0, 800.0))
    assert total.torque_nm == pytest.approx((8.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="point"):
        part_ground(
            (_bw(1.0),),
            (BodyWrench("calcn_r", (0.1, 0.0, 0.0), (0, 0, 1.0), (0, 0, 0)),),
        )


def test_the_net_ground_reaction_is_the_sum_at_the_combined_centre_of_pressure() -> (
    None
):
    """With both feet welded the per-foot split is one null-space solution;
    only the net wrench is unique, so it is what each case draws."""
    left = BodyWrench("calcn_l", (0.0, 0.2, 0.0), (0.0, 0.0, 100.0), (0, 0, 0))
    right = BodyWrench("calcn_r", (0.0, -0.2, 0.0), (0.0, 0.0, 300.0), (0, 0, 0))
    net = net_ground((left, right))
    assert net is not None
    assert net.force_n == pytest.approx((0.0, 0.0, 400.0))
    cop = centre_of_pressure(net.point_m, net.force_n, net.torque_nm)
    assert cop == pytest.approx((0.0, -0.1, 0.0))  # weighted toward the 300 N foot
    assert net_ground(()) is None


def test_the_readout_names_the_case_and_its_net_ground_force() -> None:
    net = BodyWrench("ground", (0, 0, 0), (30.0, 40.0, 116.0), (0, 0, 0))
    text = ground_readout(net, "ztcf")
    assert "ZTCF" in text and "116 N vertical" in text and "50 N horizontal" in text
    assert "not drawn" in ground_readout(None, "zvcf")


def _wrench(kind: WrenchKind, label: str) -> OverlayWrench:
    return OverlayWrench(kind, label, "body", (0, 0, 0), (0, 0, 10.0), source="t")


FRAME = ForceTorqueFrame(
    0.5,
    "mujoco",
    (
        _wrench(WrenchKind.CONTACT, "contact:sphere_1"),
        _wrench(WrenchKind.GRAVITY, "gravity:com"),
        _wrench(WrenchKind.JOINT_ACTUATOR, "joint_actuator:hip"),
    ),
)
GROUND = part_ground((_bw(300.0),), (_bw(500.0),))


def _kinds(frame: ForceTorqueFrame) -> set[WrenchKind]:
    return {w.kind for w in frame.wrenches}


@pytest.mark.parametrize(
    ("part", "kinds"),
    [
        ("ztcf", {WrenchKind.GRAVITY}),
        ("zvcf", {WrenchKind.JOINT_ACTUATOR}),
        ("total", {WrenchKind.GRAVITY, WrenchKind.JOINT_ACTUATOR}),
    ],
)
def test_each_case_keeps_only_its_own_force_kinds(part: str, kinds: set) -> None:
    """The recorded contact arrows go: the case's ground reaction is drawn
    as its own case-coloured arrow (:func:`ground_arrows`)."""
    at = with_part_wrenches(lambda _i: FRAME, part)
    assert _kinds(at(0)) == kinds
    assert FORCE_NOTES[part]


STYLE = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=3.0)


@pytest.mark.parametrize(
    ("part", "fz", "hex_field"),
    [
        ("ztcf", 300.0, "drift_hex"),
        ("zvcf", 500.0, "control_hex"),
        ("total", 800.0, "total_hex"),
    ],
)
def test_the_ground_arrow_wears_the_case_colour_at_its_centre_of_pressure(
    part: str, fz: float, hex_field: str
) -> None:
    (arrow,) = ground_arrows(lambda _i: GROUND, part, STYLE)(0)
    look = TipToTailStyle()
    assert arrow.rgba == pytest.approx(hex_to_rgba(getattr(look, hex_field)))
    assert arrow.label.startswith("ground")
    assert np.subtract(arrow.tip_m, arrow.tail_m) == pytest.approx((0, 0, fz * 0.001))
    assert arrow.tail_m == pytest.approx((0.0, 0.0, 0.0))


def test_the_case_force_key_row_names_the_case_in_its_colour() -> None:
    (text, rgb), torques = case_force_key("zvcf")
    assert "torque" in torques[0].lower()
    actuator = get_kind_rgba(WrenchKind.JOINT_ACTUATOR)
    assert torques[1] == tuple(round(255 * c) for c in actuator[:3])
    assert "ground" in text.lower() and "ZVCF" in text
    look = TipToTailStyle()
    assert rgb == tuple(round(255 * c) for c in hex_to_rgba(look.control_hex)[:3])
    assert case_force_key("recorded") == ()
    # the cases that draw the weight arrow name it in the key, in its colour
    (_ground, weight) = case_force_key("ztcf")
    assert "weight" in weight[0].lower()
    grey = get_kind_rgba(WrenchKind.GRAVITY)
    assert weight[1] == tuple(round(255 * c) for c in grey[:3])
    assert len(case_force_key("total")) == 3


def test_recorded_keeps_the_frame_and_an_unknown_part_is_refused() -> None:
    at = with_part_wrenches(lambda _i: FRAME, "recorded")
    assert at(0) is FRAME
    with pytest.raises(ValueError, match="part"):
        with_part_wrenches(lambda _i: FRAME, "sideways")


def test_the_force_part_flag_defaults_to_the_recorded_forces() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(BASE).force_part == "recorded"
    assert parser.parse_args([*BASE, "--force-part", "ztcf"]).force_part == "ztcf"
    with pytest.raises(SystemExit):
        parser.parse_args([*BASE, "--force-part", "sideways"])


SEEDS = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.mark.requires_pinocchio
def test_live_ground_parts_add_up_on_the_seed_swing() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.ztcf_axial import part_axial_source

    spec = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    order = tuple(json.loads(spec)["coordinate_order"])
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q, v, tau = (rec[n][300] for n in ("q", "v", "tau"))
    ground = part_axial_source(spec, order).split(q, v, tau, 1.25).ground
    assert len(ground["total"]) >= 2  # both feet
    for d, c, t in zip(ground["ztcf"], ground["zvcf"], ground["total"], strict=True):
        assert np.add(d.force_n, c.force_n) == pytest.approx(t.force_n)
        assert np.all(np.isfinite(t.force_n))


def test_extra_hud_lines_follow_the_clip_frames_and_hold_after_the_end() -> None:
    """The ground readout is by source state, like the extra arrows: a
    resampled clip reads it at the mapped state and a held tail repeats it."""
    from src.tools.native_viewer_export.overlay_feed import OverlayFeed

    feed = OverlayFeed(lambda _i: FRAME, extra_hud_lines=lambda i: (f"state {i}",))
    assert feed.hud_lines_at(3) == ("state 3",)
    resampled = feed.following(object(), lambda k: 2 * k)
    assert resampled.hud_lines_at(3) == ("state 6",)
    assert resampled.held_after(4).hud_lines_at(9) == ("state 8",)


def test_no_hand_case_colour_repeats_a_force_kind_colour() -> None:
    """A case render draws hand arrows beside weight and torque arrows; a
    shared colour would make two different forces read as one. Muscle
    arrows (musculoskeletal models only) still share the control vermillion:
    a tracked follow-up on the report-package epic."""
    from src.tools.native_viewer_export.ztcf_axial import PART_KINDS
    from src.tools.native_viewer_export.ztcf_feed import hand_styles

    drawn = set().union(*PART_KINDS.values()) - {WrenchKind.MUSCLE}
    kinds = {
        "#" + "".join(f"{round(255 * c):02X}" for c in get_kind_rgba(k)[:3])
        for k in drawn
    }
    for look in hand_styles().values():
        for field in ("drift_hex", "control_hex", "total_hex"):
            assert getattr(look, field).upper() not in kinds, field


def test_the_hud_says_why_the_shading_is_withheld_at_ball_contact() -> None:
    """Section loads are blanked within a step of contact (the impact impulse
    is not modelled); unexplained, the unshaded body reads as a glitch."""
    from types import SimpleNamespace

    from src.tools.native_viewer_export.overlay_feed import (
        CONTACT_BLACKOUT_NOTE,
        OverlayFeed,
    )

    def frame(value):
        loads = SimpleNamespace(values_n={"trunk": value})
        return SimpleNamespace(axial_loads=loads, wrenches=())

    blank = OverlayFeed(lambda _i: frame(None))
    assert CONTACT_BLACKOUT_NOTE in blank.hud_lines_at(0)
    assert "impulse" in CONTACT_BLACKOUT_NOTE
    assert CONTACT_BLACKOUT_NOTE not in OverlayFeed(lambda _i: frame(5.0)).hud_lines_at(
        0
    )
