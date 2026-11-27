"""Ground reaction and force kinds by ZTCF/ZVCF case (#15150 Track Z, Z4c)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.body_part_viz.mujoco_axial_loads import BodyWrench
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.palette import hex_to_rgba
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.overlay import with_force_part
from src.tools.native_viewer_export.overlay_feed import OverlayFeed
from src.tools.native_viewer_export.ztcf_feed import hand_styles
from src.tools.native_viewer_export.ztcf_ground import (
    centre_of_pressure,
    ground_arrows,
    load_ground_settings,
    net_ground,
    part_ground,
    with_part_wrenches,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
LOOK = load_ground_settings()
CASE = hand_styles()[LOOK["case_colours_hand"]]
STYLE = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=3.0)
KINDS = (WrenchKind.CONTACT, WrenchKind.GRAVITY, WrenchKind.JOINT_ACTUATOR)
FRAME = ForceTorqueFrame(
    0.5,
    "mujoco",
    tuple(
        OverlayWrench(k, f"{k.value}:a", "b", (0, 0, 0), (0, 0, 1), source="t")
        for k in KINDS
    ),
)


def _bw(fz: float, mx: float = 0.0, y: float = 0.0) -> BodyWrench:
    return BodyWrench("calcn_r", (0.0, y, 0.0), (0.0, 0.0, fz), (mx, 0.0, 0.0))


def _net(wrenches: tuple[BodyWrench, ...]) -> tuple[np.ndarray, ...]:
    net = net_ground(wrenches)
    assert net is not None and net.force_n and net.torque_nm
    cop = centre_of_pressure(net.point_m, net.force_n, net.torque_nm)
    assert cop is not None
    return tuple(np.asarray(x) for x in (net.point_m, net.force_n, net.torque_nm, cop))


GROUND = part_ground((_bw(300.0, 3.0),), (_bw(500.0, 5.0),))


def test_centre_of_pressure_is_where_the_tilting_moment_vanishes() -> None:
    cop = centre_of_pressure((0, 0, 0), (0, 0, 100.0), (10.0, -20.0, 0.0))
    assert cop is not None and cop == pytest.approx((0.2, 0.1, 0.0))
    assert np.cross(np.asarray(cop), (0.0, 0.0, 100.0))[:2] == pytest.approx(
        (10.0, -20.0)
    )
    assert centre_of_pressure((1, 2, 0), (50.0, 0.0, 1e-6), (3, 4, 0)) is None
    with pytest.raises(ValueError, match="finite"):
        centre_of_pressure((0, 0, 0), (0, 0, np.nan), (0, 0, 0))


def test_the_ground_parts_add_and_the_net_sits_between_the_feet() -> None:
    (total,) = GROUND["total"]
    assert total.force_n == pytest.approx((0, 0, 800.0))
    assert total.torque_nm == pytest.approx((8.0, 0, 0))
    with pytest.raises(ValueError, match="point"):
        part_ground((_bw(1.0),), (_bw(1.0, y=0.1),))
    *_, cop = _net((_bw(100.0, y=0.2), _bw(300.0, y=-0.2)))
    assert cop == pytest.approx((0.0, -0.1, 0.0))  # toward the 300 N foot
    assert net_ground(()) is None


@pytest.mark.parametrize(
    ("part", "kinds", "fz", "field"),
    [
        ("ztcf", {WrenchKind.GRAVITY}, 300.0, "drift_hex"),
        ("zvcf", {WrenchKind.JOINT_ACTUATOR}, 500.0, "control_hex"),
    ],
)
def test_a_case_keeps_its_kinds_and_wears_the_hand_case_colour(
    part: str, kinds: set, fz: float, field: str
) -> None:
    hand = SimpleNamespace(label="hand")
    feed = OverlayFeed(
        lambda _i: FRAME,
        STYLE,
        extra_arrows=lambda _i: (hand,),
        colour_key=(("h", (1, 2, 3)),),
    )
    out = with_force_part(feed, lambda _i: SimpleNamespace(ground=GROUND), STYLE, part)
    assert {w.kind for w in out.frame_at(0).wrenches} == kinds
    assert out.extra_arrows is not None
    first, arrow = out.extra_arrows(0)
    assert first is hand and arrow.label.startswith(LOOK["arrow_label"])
    assert arrow.rgba == pytest.approx(hex_to_rgba(getattr(CASE, field)))
    assert np.subtract(arrow.tip_m, arrow.tail_m) == pytest.approx((0, 0, fz * 1e-3))
    ground_row = (
        LOOK["key_ground"].format(part=part.upper()),
        tuple(round(255 * c) for c in arrow.rgba[:3]),
    )
    assert out.colour_key == (("h", (1, 2, 3)), ground_row)


def test_no_arrow_where_the_pressure_centre_is_undefined_and_bad_parts_fail() -> None:
    flat = part_ground((_bw(0.0),), (_bw(0.0),))
    assert ground_arrows(lambda _i: flat, "ztcf", STYLE, LOOK)(0) == ()  # no NaN
    assert with_part_wrenches(lambda _i: FRAME, "recorded")(0) is FRAME
    with pytest.raises(ValueError, match="part"):
        with_part_wrenches(lambda _i: FRAME, "sideways")
    with pytest.raises(ValueError, match="case"):
        ground_arrows(lambda _i: GROUND, "recorded", STYLE, LOOK)


def test_the_force_part_flag_defaults_to_the_recorded_forces() -> None:
    base = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    parser = cli.build_parser()
    assert parser.parse_args(base).force_part == "recorded"
    assert parser.parse_args([*base, "--force-part", "zvcf"]).force_part == "zvcf"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--force-part", "sideways"])


SEEDS = Path(__file__).resolve().parents[4] / (
    "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.fixture(scope="module")
def seed() -> SimpleNamespace:
    pytest.importorskip("pinocchio")
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.tools.native_viewer_export.ztcf_axial import part_axial_source
    from src.tools.native_viewer_export.ztcf_feed import split_provider

    spec = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    order = tuple(json.loads(spec)["coordinate_order"])
    adapter = PinocchioForceAdapter(spec)
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        states = {n: rec[n] for n in ("q", "v", "tau")}
    return SimpleNamespace(
        source=part_axial_source(spec, order),
        provider=split_provider(spec, order).provider,
        adapter=adapter,
        take=np.array([order.index(n) for n in adapter.coordinate_order]),
        pad_m=max(s["radius_m"] for s in json.loads(spec)["contact"]["spheres"]),
        **states,
    )


def _ground(seed: SimpleNamespace, k: int) -> dict:
    return seed.source.split(seed.q[k], seed.v[k], seed.tau[k], 0.0).ground


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize("k", [0, 150, 300, 400])
def test_live_parts_add_and_each_pressure_centre_lies_within_the_feet(
    seed: SimpleNamespace, k: int
) -> None:
    ground = _ground(seed, k)
    for d, c, t in zip(ground["ztcf"], ground["zvcf"], ground["total"], strict=True):
        assert np.add(d.force_n, c.force_n) == pytest.approx(t.force_n, abs=1e-9)
        assert np.add(d.torque_nm, c.torque_nm) == pytest.approx(t.torque_nm, abs=1e-9)
    spheres = seed.adapter.foot_contact_candidates(seed.q[k][seed.take])
    xy = np.array([s.contact.position[:2] for s in spheres])
    lo, hi = xy.min(axis=0) - seed.pad_m, xy.max(axis=0) + seed.pad_m
    moment = {}
    for part in ("ztcf", "zvcf", "total"):
        point, force, _torque, cop = _net(ground[part])
        assert force[2] > LOOK["min_vertical_n"]
        assert np.all(lo <= cop[:2]) and np.all(cop[:2] <= hi), part
        moment[part] = force[2] * (cop - point)[:2]
    assert moment["ztcf"] + moment["zvcf"] == pytest.approx(moment["total"], abs=1e-6)


@pytest.mark.requires_pinocchio
def test_live_address_ground_obeys_newton_for_each_case(seed: SimpleNamespace) -> None:
    """Address total = body weight; ZTCF (tau = 0, body sags) = m (g + a_com)
    of the drift solve, not the weight; ZVCF = m a_com (no gravity)."""
    import pinocchio as pin

    from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
        drift_and_control_states,
    )

    k = int(np.argmin(np.linalg.norm(seed.v[: len(seed.v) // 4], axis=1)))
    q, v, take = seed.q[k], seed.v[k], seed.take
    fz = {p: _net(w)[1][2] for p, w in _ground(seed, k).items()}
    model = seed.adapter.native_plant.model
    data = model.createData()
    mass = sum(inertia.mass for inertia in model.inertias)
    weight = -mass * float(seed.adapter.gravity_m_s2()[2])
    assert fz["total"] == pytest.approx(weight, rel=0.02)
    (a_d, _), (a_c, _) = drift_and_control_states(seed.provider, q, v, seed.tau[k])
    for part, a, rates, g in (("ztcf", a_d, v, weight), ("zvcf", a_c, 0 * v, 0.0)):
        com = pin.centerOfMass  # type: ignore[attr-defined]
        com(model, data, q[take], rates[take], np.asarray(a)[take])
        assert fz[part] == pytest.approx(g + mass * data.acom[0][2], rel=0.01)
