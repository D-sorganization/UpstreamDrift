"""Tests for the shared ground-reaction analysis core (GCV-1, #11707)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics import ground_reaction as gr
from src.shared.python.biomechanics.ground_reaction import (
    COP_MIN_FZ_N,
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
    foot_reaction,
    grf_overlay_wrench,
    to_contact_reaction,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.motion_matching.force_torque import ContactReaction

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665


def _vertical(points, fz):
    pts = np.asarray(points, float)
    return np.tile([0.0, 0.0, fz], (len(pts), 1)), pts


def test_threshold_is_named_ten_newtons() -> None:
    assert COP_MIN_FZ_N == 10.0


def test_single_point_load_cop_is_the_point_with_zero_free_moment() -> None:
    f, p = _vertical([[0.1, 0.2, 0.0]], 400.0)
    r = foot_reaction("left", f, p)
    np.testing.assert_allclose(r.cop_m, (0.1, 0.2, 0.0), atol=1e-12)
    np.testing.assert_allclose(r.free_moment_nm, (0, 0, 0), atol=1e-12)
    assert r.in_contact


def test_symmetric_two_foot_stance_net_is_midpoint_and_sum() -> None:
    w = 80.0 * G
    cl = ContactSet(*_vertical([[0.0, 0.15, 0.0]], w / 2))
    cr = ContactSet(*_vertical([[0.0, -0.15, 0.0]], w / 2))
    b = analyze_ground_reaction({"left": cl, "right": cr}, (0.0, 0.0, 0.95))
    np.testing.assert_allclose(b.net.force_n, (0, 0, w))
    np.testing.assert_allclose(b.net.cop_m, (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(b.per_foot["left"].force_n, (0, 0, w / 2))
    np.testing.assert_allclose(b.per_foot["right"].force_n, (0, 0, w / 2))


def test_pure_torsion_on_plate() -> None:
    # four equal tangential forces around the centre (0.3, -0.1), plus normal loads
    c = np.array([0.3, -0.1, 0.0])
    r = 0.1
    pts = c + np.array([[r, 0, 0], [0, r, 0], [-r, 0, 0], [0, -r, 0]])
    tang = np.array([[0, 5, 0], [-5, 0, 0], [0, -5, 0], [5, 0, 0]], float)
    normal = np.tile([0, 0, 100.0], (4, 1))
    res = foot_reaction("f", tang + normal, pts)
    np.testing.assert_allclose(res.force_n, (0, 0, 400.0), atol=1e-12)
    np.testing.assert_allclose(res.cop_m, c, atol=1e-12)
    np.testing.assert_allclose(res.free_moment_nm, (0, 0, 4 * 5 * r), atol=1e-12)


def test_contact_torques_enter_the_moment() -> None:
    f, p = _vertical([[0.0, 0.0, 0.0]], 100.0)
    res = foot_reaction("f", f, p, torques=[[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(res.free_moment_nm, (0, 0, 3.0))
    np.testing.assert_allclose(res.moment_about_origin_nm, (0, 0, 3.0))


def test_free_moment_independent_of_reference_origin() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.1, 0.1, (5, 3)) * [1, 1, 0]
    forces = rng.uniform(-10, 10, (5, 3))
    forces[:, 2] = rng.uniform(50, 150, 5)
    base = foot_reaction("f", forces, pts)
    shift = np.array([1.7, -2.3, 0.0])
    moved = foot_reaction("f", forces, pts + shift)
    np.testing.assert_allclose(moved.free_moment_nm, base.free_moment_nm, atol=1e-9)
    np.testing.assert_allclose(moved.cop_m, base.cop_m + shift, atol=1e-9)


def test_net_free_moment_differs_from_sum_and_matches_when_cops_coincide() -> None:
    # shear at offset foot CoPs makes the net free moment differ from the sum
    sheared = {
        "left": ContactSet(np.array([[20.0, 0.0, 300.0]]), np.array([[0, 0.2, 0.0]])),
        "right": ContactSet(np.array([[0.0, 0.0, 100.0]]), np.array([[0, -0.2, 0.0]])),
    }
    bs = analyze_ground_reaction(sheared, (0, 0, 1))
    assert bs.net.free_moment_nm[2] != pytest.approx(
        sum(r.free_moment_nm[2] for r in bs.per_foot.values())
    )
    # coincident CoPs: net free moment equals the sum
    same = {
        "left": ContactSet(
            np.array([[5.0, 2.0, 300.0]]),
            np.array([[0.1, 0.1, 0.0]]),
            np.array([[0, 0, 1.5]]),
        ),
        "right": ContactSet(
            np.array([[-1.0, 3.0, 100.0]]),
            np.array([[0.1, 0.1, 0.0]]),
            np.array([[0, 0, 0.5]]),
        ),
    }
    bc = analyze_ground_reaction(same, (0, 0, 1))
    np.testing.assert_allclose(bc.net.free_moment_nm[2], 1.5 + 0.5 + 0.0, atol=1e-9)


def test_net_com_moment_is_exact_sum_random() -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        contacts = {}
        for name in ("left", "right"):
            n = int(rng.integers(1, 5))
            contacts[name] = ContactSet(
                rng.uniform(-50, 200, (n, 3)),
                rng.uniform(-0.5, 0.5, (n, 3)) * [1, 1, 0],
                rng.uniform(-3, 3, (n, 3)),
            )
        com = rng.uniform(-0.3, 0.3, 3) + [0, 0, 1.0]
        b = analyze_ground_reaction(contacts, com)
        m = b.moment_about_com_nm
        np.testing.assert_allclose(
            m["net"], m["left"] + m["right"], rtol=1e-12, atol=1e-9
        )


def test_com_moment_equals_arm_cross_force_plus_free_moment() -> None:
    f, p = _vertical([[0.1, 0.0, 0.0]], 500.0)
    f = f + [20.0, -10.0, 0.0]
    com = np.array([0.0, 0.05, 1.0])
    b = analyze_ground_reaction(
        {"left": ContactSet(f, p, np.array([[0, 0, 2.0]]))}, com
    )
    r = b.per_foot["left"]
    expected = np.cross(r.cop_m - com, r.force_n) + r.free_moment_nm
    np.testing.assert_allclose(b.moment_about_com_nm["left"], expected, atol=1e-9)
    np.testing.assert_allclose(
        b.force_moment_about_com_nm["left"], np.cross(r.cop_m - com, r.force_n)
    )


def test_below_threshold_cop_and_free_moment_none_force_reported() -> None:
    f, p = _vertical([[0.0, 0.0, 0.0]], COP_MIN_FZ_N - 0.5)
    r = foot_reaction("f", f, p)
    assert r.cop_m is None and r.free_moment_nm is None
    np.testing.assert_allclose(r.force_n, (0, 0, COP_MIN_FZ_N - 0.5))
    assert r.in_contact
    b = analyze_ground_reaction({"f": ContactSet(f, p)}, (0, 0, 1))
    assert b.force_moment_about_com_nm["f"] is None
    assert b.moment_about_com_nm["f"] is not None


def test_threshold_is_configurable_and_inclusive() -> None:
    f, p = _vertical([[0.0, 0.0, 0.0]], 5.0)
    assert foot_reaction("f", f, p).cop_m is None
    assert foot_reaction("f", f, p, cop_min_fz_n=5.0).cop_m is not None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_inputs_raise(bad: float) -> None:
    f, p = _vertical([[0.0, 0.0, 0.0]], 100.0)
    f[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        foot_reaction("f", f, p)
    f, p = _vertical([[0.0, 0.0, 0.0]], 100.0)
    p[0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        foot_reaction("f", f, p)
    with pytest.raises(ValueError, match="finite"):
        analyze_ground_reaction({}, (0, bad, 0))
    with pytest.raises(ValueError, match="ground_height_m"):
        foot_reaction("f", *_vertical([[0, 0, 0]], 100.0), ground_height_m=bad)


def test_shape_and_label_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        foot_reaction("f", np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        foot_reaction("f", np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="label"):
        foot_reaction("", np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="cop_min_fz_n"):
        foot_reaction("f", np.zeros((0, 3)), np.zeros((0, 3)), cop_min_fz_n=-1.0)
    with pytest.raises(TypeError):
        analyze_ground_reaction({"f": "nope"}, (0, 0, 1))  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="reserved"):
        analyze_ground_reaction({"net": ContactSet.empty()}, (0, 0, 1))


@pytest.mark.parametrize("zg", [0.0, 0.37, -0.2])
def test_ground_height_cop_lies_on_plane(zg: float) -> None:
    pts = np.array([[0.1, 0.0, zg], [0.3, 0.2, zg]])
    forces = np.array([[10.0, -5.0, 200.0], [-3.0, 8.0, 100.0]])
    r = foot_reaction("f", forces, pts, ground_height_m=zg)
    assert r.cop_m[2] == pytest.approx(zg)
    # contact points on the plane: CoP equals the normal-weighted point mean
    expected = (pts * forces[:, 2:3]).sum(axis=0) / forces[:, 2].sum()
    np.testing.assert_allclose(r.cop_m[:2], expected[:2], atol=1e-12)


def test_ground_height_formula_uses_zg_terms() -> None:
    # force off the plane: x_cop = (zg*Fx - My)/Fz, y_cop = (Mx + zg*Fy)/Fz
    f = np.array([[30.0, 20.0, 100.0]])
    p = np.array([[0.2, 0.1, 0.5]])
    zg = 0.1
    r = foot_reaction("f", f, p, ground_height_m=zg)
    m = np.cross(p[0], f[0])
    assert r.cop_m[0] == pytest.approx((zg * 30.0 - m[1]) / 100.0)
    assert r.cop_m[1] == pytest.approx((m[0] + zg * 20.0) / 100.0)


def test_no_contact_foot_reports_zero_force_and_no_cop() -> None:
    b = analyze_ground_reaction(
        {
            "left": ContactSet.empty(),
            "right": ContactSet(*_vertical([[0, 0, 0]], 300.0)),
        },
        (0, 0, 1),
    )
    left = b.per_foot["left"]
    assert not left.in_contact and left.cop_m is None and left.free_moment_nm is None
    np.testing.assert_array_equal(left.force_n, (0, 0, 0))
    assert b.net.in_contact
    np.testing.assert_allclose(b.net.cop_m, (0, 0, 0))


def test_to_overlay_wrenches_labels_and_kinds() -> None:
    f = np.array([[10.0, 0.0, 200.0]])
    p = np.array([[0.1, 0.0, 0.0]])
    contacts = {
        "left": ContactSet(f, p, np.array([[0, 0, 1.0]])),
        "right": ContactSet(f, -p),
        "idle": ContactSet.empty(),
    }
    b = analyze_ground_reaction(contacts, (0.0, 0.0, 1.0))
    ws = {w.label: w for w in to_overlay_wrenches(b)}
    expected = {
        "contact:grf_left", "contact:grf_right", "contact:grf_net",
        "contact:free_moment_left", "contact:free_moment_right",
        "contact:free_moment_net",
        "contact:moment_com_left", "contact:moment_com_right",
        "contact:moment_com_net",
    }  # fmt: skip
    assert set(ws) == expected
    assert all(w.kind is WrenchKind.CONTACT for w in ws.values())
    np.testing.assert_allclose(ws["contact:grf_left"].point_m, b.per_foot["left"].cop_m)
    np.testing.assert_allclose(ws["contact:grf_net"].point_m, b.net.cop_m)
    assert ws["contact:grf_left"].torque_nm is None
    assert ws["contact:free_moment_left"].force_n is None
    np.testing.assert_allclose(ws["contact:free_moment_left"].torque_nm, (0, 0, 1.0))
    np.testing.assert_allclose(ws["contact:moment_com_net"].point_m, (0, 0, 1.0))
    np.testing.assert_allclose(
        ws["contact:moment_com_left"].torque_nm, b.moment_about_com_nm["left"]
    )


def test_overlay_omits_unavailable_cop_quantities_and_sanitises_labels() -> None:
    f, p = _vertical([[0.4, 0.0, 0.0]], 3.0)
    b = analyze_ground_reaction({"L foot": ContactSet(f, p)}, (0, 0, 1))
    labels = {w.label for w in to_overlay_wrenches(b)}
    assert "contact:grf_L_foot" in labels
    assert "contact:free_moment_L_foot" not in labels
    grf = next(w for w in to_overlay_wrenches(b) if w.label == "contact:grf_L_foot")
    np.testing.assert_allclose(grf.point_m, (0.4, 0.0, 0.0))  # contact centroid


def test_to_overlay_wrenches_rejects_wrong_type_and_blank_source() -> None:
    with pytest.raises(TypeError):
        to_overlay_wrenches("x")  # type: ignore[arg-type]
    b = analyze_ground_reaction({}, (0, 0, 1))
    with pytest.raises(ValueError, match="source"):
        to_overlay_wrenches(b, source=" ")
    assert to_overlay_wrenches(b) == []


def test_series_stacks_with_nan_for_unavailable_and_dataframe() -> None:
    on = analyze_ground_reaction(
        {"left": ContactSet(*_vertical([[0, 0, 0]], 300.0))}, (0, 0, 1)
    )
    off = analyze_ground_reaction({"left": ContactSet.empty()}, (0, 0, 1))
    s = GroundReactionSeries.from_breakdowns([0.0, 0.1], [on, off])
    assert s.force_n["left"].shape == (2, 3)
    assert np.isnan(s.cop_m["left"][1]).all()
    assert not np.isnan(s.cop_m["left"][0]).any()
    assert np.isnan(s.free_moment_nm["net"][1]).all()
    assert s.in_contact["left"].tolist() == [True, False]
    np.testing.assert_allclose(s.moment_about_com_nm["left"][1], 0.0)
    pd = pytest.importorskip("pandas")
    df = s.to_dataframe()
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    assert {"time_s", "left_force_x_n", "net_cop_z_m"} <= set(df.columns)


def test_series_validation() -> None:
    b = analyze_ground_reaction({"left": ContactSet.empty()}, (0, 0, 1))
    c = analyze_ground_reaction({"right": ContactSet.empty()}, (0, 0, 1))
    with pytest.raises(ValueError, match="length"):
        GroundReactionSeries.from_breakdowns([0.0], [b, b])
    with pytest.raises(ValueError, match="empty"):
        GroundReactionSeries.from_breakdowns([], [])
    with pytest.raises(ValueError, match="same feet"):
        GroundReactionSeries.from_breakdowns([0.0, 1.0], [b, c])
    with pytest.raises(ValueError, match="increasing"):
        GroundReactionSeries.from_breakdowns([1.0, 0.0], [b, b])


def test_to_contact_reaction_populates_existing_dataclass() -> None:
    f, p = _vertical([[0.1, 0.2, 0.0]], 300.0)
    b = analyze_ground_reaction(
        {
            "left": ContactSet(f, p),
            "right": ContactSet.empty(),
        },
        (0, 0, 1),
    )
    cr = to_contact_reaction(b, time_s=0.25)
    assert isinstance(cr, ContactReaction)
    assert cr.time_s == 0.25
    assert cr.net_grf_n == pytest.approx((0, 0, 300.0))
    assert cr.left_foot_grf_n == pytest.approx((0, 0, 300.0))
    assert cr.right_foot_grf_n == (0.0, 0.0, 0.0)
    assert cr.left_foot_cop_m == pytest.approx((0.1, 0.2))
    assert cr.right_foot_cop_m is None
    assert cr.net_cop_m == pytest.approx((0.1, 0.2))
    assert cr.contact_status == {"left": True, "right": False}


def test_to_contact_reaction_custom_foot_names_and_missing_foot() -> None:
    b = analyze_ground_reaction({"l": ContactSet.empty()}, (0, 0, 1))
    cr = to_contact_reaction(b, time_s=0.0, left_label="l", right_label="r")
    assert cr.left_foot_grf_n == (0.0, 0.0, 0.0)
    assert cr.right_foot_grf_n is None  # not supplied: unavailable, not zero


def test_module_has_no_stray_public_names() -> None:
    assert {"foot_reaction", "analyze_ground_reaction"} <= set(gr.__all__)


def test_remaining_validation_and_grf_helper_paths() -> None:
    with pytest.raises(ValueError, match="com_m"):
        analyze_ground_reaction({}, (0, 0))
    with pytest.raises(ValueError, match="shape"):
        foot_reaction("f", np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="foot labels"):
        analyze_ground_reaction({" ": ContactSet.empty()}, (0, 0, 1))
    with pytest.raises(TypeError):
        grf_overlay_wrench("x")  # type: ignore[arg-type]
    idle = foot_reaction("f", np.zeros((0, 3)), np.zeros((0, 3)))
    assert grf_overlay_wrench(idle) is None
    loaded = foot_reaction("calcn r", *_vertical([[0.2, 0, 0]], 50.0))
    w = grf_overlay_wrench(loaded, label_part="calcn_r", body="calcn r")
    assert w.label == "contact:grf_calcn_r" and w.body == "calcn r"
