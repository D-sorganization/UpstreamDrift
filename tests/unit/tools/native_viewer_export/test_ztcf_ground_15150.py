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
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.ztcf_axial import (
    FORCE_NOTES,
    centre_of_pressure,
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
    ("part", "kinds", "fz"),
    [
        ("ztcf", {WrenchKind.CONTACT, WrenchKind.GRAVITY}, 300.0),
        ("zvcf", {WrenchKind.CONTACT, WrenchKind.JOINT_ACTUATOR}, 500.0),
        (
            "total",
            {WrenchKind.CONTACT, WrenchKind.GRAVITY, WrenchKind.JOINT_ACTUATOR},
            800.0,
        ),
    ],
)
def test_each_case_draws_its_own_ground_reaction_and_kinds(
    part: str, kinds: set, fz: float
) -> None:
    at = with_part_wrenches(lambda _i: FRAME, lambda _i: GROUND, part)
    frame = at(0)
    assert _kinds(frame) == kinds
    (contact,) = (w for w in frame.wrenches if w.kind is WrenchKind.CONTACT)
    assert contact.force_n == pytest.approx((0.0, 0.0, fz))
    assert part.upper() in contact.source
    assert FORCE_NOTES[part]


def test_recorded_keeps_the_frame_and_an_unknown_part_is_refused() -> None:
    at = with_part_wrenches(lambda _i: FRAME, lambda _i: GROUND, "recorded")
    assert at(0) is FRAME
    with pytest.raises(ValueError, match="part"):
        with_part_wrenches(lambda _i: FRAME, lambda _i: GROUND, "sideways")


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
