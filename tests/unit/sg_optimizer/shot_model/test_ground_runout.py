"""SHOW-B3b (#14610): bounce/roll rest position on the B6 slope bins."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model import ground_runout as gr
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    _impacts,
    standard_normals,
    TableGrid,
    build_grid_table,
)
from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    GROUND_MODEL_ID,
    LANDING_COLUMNS,
    SlopeAwareGround,
)
from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    PipelineLandingSimulator,
)
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import (
    SLOPE_BINS,
    make_slope_modifier,
)

pytestmark = pytest.mark.unit
GROUND = SlopeAwareGround()


def _landing(
    carry: float = 120.0, speed: float = 35.0, drop: float = 25.0, spin: float = 400.0
):
    """One landing row: x, y (yd), vx, vy, vz (m/s), backspin (rad/s); vz < 0 descends."""
    return np.array([[carry, 2.0, speed, 0.0, -drop, spin]])


def _rest(
    slope: str, lie: str = "fairway", skill: str = "mid_iron", **kw: float
) -> np.ndarray:
    """Rest offsets for one default landing, with per-shot keyword overrides."""
    return GROUND(_landing(**kw), lie, slope, skill)


def test_there_is_one_slope_vocabulary_and_it_is_the_b6_one() -> None:
    """The slope bins are the single shared B6 vocabulary object."""
    assert gr.SLOPE_BINS is SLOPE_BINS  # the same object: no second table to drift
    for slope in SLOPE_BINS:
        assert _rest(slope).shape == (1, 2)


def test_uphill_stops_shorter_and_downhill_runs_further_than_flat() -> None:
    """Uphill shortens and downhill lengthens the runout relative to flat."""
    flat, up, down = (_rest(s)[0, 0] for s in ("flat", "up_5", "down_5"))
    assert up < flat < down


def test_ball_above_or_below_feet_bins_do_not_change_runout() -> None:
    """Ball-above/below-feet slope bins leave the runout unchanged."""
    flat = _rest("flat")
    np.testing.assert_allclose(_rest("above_5"), flat)
    np.testing.assert_allclose(_rest("below_5"), flat)


def test_rest_is_never_behind_carry_and_roll_follows_the_landing_direction() -> None:
    """Rest never falls behind carry and rolls along the landing direction."""
    land = np.array([[100.0, 0.0, 0.0, 30.0, -20.0, 400.0]])  # landing sideways (+y)
    rest = GROUND(land, "fairway", "flat", "mid_iron")
    assert rest[0, 0] == pytest.approx(100.0) and rest[0, 1] > 0.0
    assert _rest("up_5", speed=1.0, spin=0.0)[0, 0] >= 120.0
    # Penner Sec. 3.2: enough backspin on a steep landing bounces the ball back
    assert _rest("up_5", speed=1.0, spin=900.0)[0, 0] < 120.0


def test_steeper_descent_and_rougher_lie_shorten_the_roll() -> None:
    """Steeper descents and rougher lies shorten the roll."""
    assert _rest("flat", drop=45.0)[0, 0] < _rest("flat", drop=10.0)[0, 0]
    fairway, rough, sand = (
        _rest("flat", lie)[0, 0] for lie in ("fairway", "rough", "sand")
    )
    assert sand < rough < fairway


def test_missing_landing_inputs_and_unknown_names_fail_closed() -> None:
    """Missing columns or unknown lie/slope/skill names raise contract violations."""
    with pytest.raises(ContractViolationError):
        GROUND(
            np.array([[100.0, 0.0]]), "fairway", "flat", "mid_iron"
        )  # no velocity columns
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "swamp", "flat", "mid_iron")
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "fairway", "uphill", "mid_iron")
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "fairway", "flat", "putter")


def test_every_coefficient_is_cited_or_a_labelled_assumption() -> None:
    """Every ground coefficient is either cited or a labelled assumption."""
    assert GROUND.model_id == GROUND_MODEL_ID
    assert gr.GROUND_PARAMETERS
    for name, p in gr.GROUND_PARAMETERS.items():
        assert p.source and p.basis == ("cited" if "bounce." in name else "assumption")
    assert LANDING_COLUMNS[:2] == ("x_yd", "y_yd")
    assert LANDING_COLUMNS[-1] == "spin_rads"


def test_bounce_parameters_come_from_the_cited_settings_file() -> None:
    """Bounce coefficients match the cited ``runout_bounce.json`` source."""
    raw = gr.load_bounce_settings()
    assert raw["source"]["doi"] == "10.1139/p02-035"
    for name, rec in raw["parameters"].items():
        record = gr.GROUND_PARAMETERS[f"bounce.{name}"]
        assert record.value == rec["value"] and "Penner" in record.source
        assert "eq" in record.rationale or "Section" in record.rationale


def test_penner_rebound_reproduces_his_haake_green_check() -> None:
    """Penner 2002 Sec. 2: 18.6 m/s at 44.4 deg from vertical, 484 rad/s backspin.

    eq. (8) gives his fitted theta_c = 15.4 deg there, and mu_c and e match the
    0.29 and 0.14 he quotes. His quoted rebound (4.34 m/s at 34.1 deg, 154 rad/s
    topspin) is NOT reproduced by eqs. (3), (6), (7) as printed: they give
    4.16 m/s at 41.7 deg and 163 rad/s, pinned here so a change is seen.
    """
    v, phi = 18.6, np.radians(44.4)
    out = gr.penner_rebound(v * np.sin(phi), v * np.cos(phi), 484.0)
    assert out.mu_c == pytest.approx(0.29, abs=0.01)
    assert out.e == pytest.approx(0.14, abs=0.005)
    assert out.rolling
    assert np.hypot(out.vx, out.vz) == pytest.approx(4.16, abs=0.01)
    assert np.degrees(np.arctan2(out.vx, out.vz)) == pytest.approx(41.7, abs=0.1)


#: Penner 2002, Sec. 3: landing speed (m/s), angle from vertical (deg) and
#: backspin (rad/s) of his drives and 9-iron, with the mu_c and first-bounce
#: height (m, None where he quotes none) he reports for them
PENNER_QUOTED = [
    pytest.param(26.8, 68.0, 224.0, 0.23, 2.82, id="drive_35_m_s_club"),
    pytest.param(26.2, 46.2, 315.0, 0.18, 1.17, id="drive_55_m_s_club"),
    pytest.param(23.8, 34.0, 809.0, 0.28, None, id="nine_iron_809"),
    pytest.param(23.8, 34.0, 1000.0, 0.33, None, id="nine_iron_1000"),
    pytest.param(23.8, 34.0, 1200.0, 0.38, None, id="nine_iron_1200"),
]


@pytest.mark.parametrize(("v", "phi_deg", "spin", "mu_c", "height_m"), PENNER_QUOTED)
def test_penner_rebound_reproduces_his_quoted_drive_and_iron_bounces(
    v: float, phi_deg: float, spin: float, mu_c: float, height_m: float | None
) -> None:
    """The primary validation: the printed equations reproduce the mu_c and
    first-bounce height Penner quotes for his drives and 9-iron (to the
    precision he quotes), which the Haake case above cannot."""
    phi = np.radians(phi_deg)
    out = gr.penner_rebound(v * np.sin(phi), v * np.cos(phi), spin)
    assert out.mu_c == pytest.approx(mu_c, abs=0.005)
    if height_m is not None:
        assert out.vz**2 / (2.0 * GRAVITY_M_S2) == pytest.approx(height_m, abs=0.01)


def test_more_backspin_bounces_shorter_until_the_ball_slides() -> None:
    """Rolling bounce (eq. 3a) loses 2/7 r w; once mu < mu_c (eq. 2) spin is moot."""
    spins = np.linspace(0.0, 2000.0, 41)
    hop = gr.first_hop_m(np.full(41, 11.9), np.full(41, 14.5), spins)
    assert np.all(np.diff(hop) <= 1e-12)
    rolling = gr.penner_rebound(11.9, 14.5, spins).rolling
    assert rolling[0] and not rolling[-1]
    assert np.all(np.diff(hop[rolling]) < 0.0)
    np.testing.assert_allclose(hop[~rolling], hop[-1])


def test_penner_rebound_low_liner_vz_positive() -> None:
    """#15861 bug 1: theta_c must be clamped so a low-liner landing gives vz > 0.

    At h=55, d=2 m/s (phi ~88 deg from vertical) eq. 8 extrapolates tc past pi/2,
    producing cos(tc) < 0 and vz ~-4.7 m/s (into the ground) without the clamp.
    """
    out = gr.penner_rebound(55.0, 2.0, 500.0)
    assert float(out.vz) > 0.0, f"vz={float(out.vz):.3f}: ball bounced into ground"


def test_penner_rebound_topspin_slip_classified_and_directed_correctly() -> None:
    """#15861 bug 2: when topspin gives mu_c < 0 with |mu_c| > mu, must classify as slip.

    At h=15, d=25, w=-3000 (topspin) mu_c ~-0.51, |mu_c| > friction_mu=0.40.
    Without the fix rolling=True (wrong); with fix rolling=False (correct).
    """
    out = gr.penner_rebound(15.0, 25.0, -3000.0)
    assert float(out.mu_c) < 0.0, (
        f"expected topspin mu_c < 0, got {float(out.mu_c):.3f}"
    )
    p = {k[7:]: v.value for k, v in gr.GROUND_PARAMETERS.items() if k[:7] == "bounce."}
    assert abs(float(out.mu_c)) > p["friction_mu"], (
        f"|mu_c|={abs(float(out.mu_c)):.3f} not > friction_mu={p['friction_mu']:.3f}"
    )
    assert not bool(out.rolling), (
        f"topspin |mu_c| > friction_mu must slip, got rolling=True; mu_c={float(out.mu_c):.3f}"
    )


def _eq8_tilt_deg(horizontal: float, down: float) -> float:
    """Penner eq. (8) tilt (deg), recomputed straight from the cited coefficients."""
    p = {k[7:]: v.value for k, v in gr.GROUND_PARAMETERS.items() if k[:7] == "bounce."}
    phi = np.degrees(np.arctan2(horizontal, down))
    return (
        p["theta_c_ref_deg"]
        * (np.hypot(horizontal, down) / p["theta_c_ref_speed_mps"])
        * (phi / p["theta_c_ref_angle_deg"])
    )


def _tilt_domain_edge_mps() -> float:
    """Landing speed at Penner's reference angle where eq. (8) reaches the 45 deg edge."""
    p = {k[7:]: v.value for k, v in gr.GROUND_PARAMETERS.items() if k[:7] == "bounce."}
    return p["theta_c_ref_speed_mps"] * 45.0 / p["theta_c_ref_deg"]


def test_penner_rebound_flags_extrapolated_tilt() -> None:
    """#15861: the 45 deg theta_c clamp is an explicit flag, not a silent bound.

    Penner fitted eq. (8) to a single green impact (15.4 deg at 18.6 m/s and
    44.4 deg from vertical), so every tilt past 45 deg is extrapolation. Clamping
    keeps the rebound physical, but a caller reading only ``vx``/``vz`` cannot tell
    a fitted bounce from an invented one, so the out-of-domain case must be marked.
    """
    assert _eq8_tilt_deg(55.0, 2.0) > 45.0, "input must really leave the domain"
    out = gr.penner_rebound(55.0, 2.0, 500.0)
    assert bool(out.tilt_clamped), (
        "eq. (8) past 45 deg must report an extrapolated tilt"
    )


def test_penner_rebound_does_not_flag_tilt_inside_the_calibrated_domain() -> None:
    """Penner's own reference impact stays unflagged: the flag is not always true."""
    v, phi = 18.6, np.radians(44.4)
    assert _eq8_tilt_deg(v * np.sin(phi), v * np.cos(phi)) < 45.0
    out = gr.penner_rebound(v * np.sin(phi), v * np.cos(phi), 484.0)
    assert not bool(out.tilt_clamped), "in-domain bounce must not be flagged"


def test_tilt_clamp_flag_is_per_element_and_flips_at_the_domain_edge() -> None:
    """Mixed landings flag elementwise, and the flag turns on exactly at 45 deg."""
    edge = _tilt_domain_edge_mps()
    inside = 0.999 * edge
    outside = 1.001 * edge
    phi = np.radians(44.4)
    speeds = np.array([inside, outside])
    out = gr.penner_rebound(speeds * np.sin(phi), speeds * np.cos(phi), np.zeros(2))
    assert _eq8_tilt_deg(inside * np.sin(phi), inside * np.cos(phi)) < 45.0
    assert _eq8_tilt_deg(outside * np.sin(phi), outside * np.cos(phi)) > 45.0
    assert list(np.asarray(out.tilt_clamped, bool)) == [False, True]
    assert np.shape(out.tilt_clamped) == np.shape(out.rolling)


def test_landing_backspin_decays_with_the_flight_kernel_rate() -> None:
    """Landing backspin decays by ``exp(-k t)`` at the flight kernel rate."""
    w0 = 6000.0 * 2.0 * np.pi / 60.0
    assert gr.landing_backspin_rads(6000.0, 0.0) == pytest.approx(w0)
    assert gr.landing_backspin_rads(6000.0, 6.0, 0.5) == pytest.approx(
        0.5 * w0 * np.exp(-0.05 * 6.0)
    )


def _stub_sim(impacts, skill, wind):
    """Landing state from the impact rows: speed drives carry and descent is fixed."""
    n = len(impacts)
    carry = impacts[:, 0] * 2.0
    return np.column_stack(
        [
            carry,
            *(np.zeros(n), np.full(n, 30.0), np.zeros(n), np.full(n, -20.0)),
            np.full(n, 400.0),
        ]
    )


def test_table_cells_report_rest_not_carry_and_slopes_change_them() -> None:
    """Table cells report rest positions and slope bins move the runout."""
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",), ("flat", "up_5", "down_5"))
    kw = {"n": 64, "seed": 3, "slope_modifier": make_slope_modifier}
    carry = build_grid_table(
        family,
        lambda i, s, w: _stub_sim(i, s, w)[:, :2],
        grid,
        **kw,
    )
    rest = build_grid_table(family, _stub_sim, grid, ground=GROUND, **kw)
    cell = lambda t, slope: t.lookup("mid_iron", "mid_iron", "fairway", "calm", slope)  # noqa: E731
    run = {k: cell(rest, k).mean[0] - cell(carry, k).mean[0] for k in grid.slopes}
    assert run["flat"] > 0.0
    assert (
        run["up_5"] < run["flat"] < run["down_5"]
    )  # the slope bins alone move the runout
    assert rest.provenance["ground_model"] == GROUND_MODEL_ID
    assert (
        "ground_model" not in carry.provenance
    )  # carry-only hashes are unchanged (#14708, #14816)


def test_ground_without_landing_state_columns_is_rejected() -> None:
    """A ground model that needs landing columns rejects rows without them."""
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    with pytest.raises(ContractViolationError):
        build_grid_table(
            family,
            lambda i, s, w: _stub_sim(i, s, w)[:, :2],
            grid,
            n=8,
            ground=GROUND,
        )


def _prov_table(ground):
    """Build a small provenance table for one ground model."""
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    return build_grid_table(family, _stub_sim, grid, n=16, seed=1, ground=ground)


def test_changing_a_coefficient_changes_the_table_hash(monkeypatch) -> None:
    """Changing a coefficient changes the provenance hash, not the table cells."""
    base = _prov_table(GROUND)
    rec = gr.GROUND_PARAMETERS[
        "roll.sand"
    ]  # unused on a fairway grid: cells are unchanged
    monkeypatch.setitem(
        gr.GROUND_PARAMETERS, "roll.sand", replace(rec, value=rec.value + 0.1)
    )
    changed = _prov_table(GROUND)
    assert changed.cells == base.cells  # same cells, different physics: hash must move
    assert changed.table_hash != base.table_hash
    prov = base.provenance
    assert prov["ground_parameters"]["roll.sand"] == rec.value
    assert "stance" in prov["ground_slope_source"]
    # one lie bin is the impact lie AND the landing surface: declared and hashed
    assert "landing surface" in prov["ground_lie_source"]


_FULL_SWING_SKILLS = (
    "driver",
    "fairway_wood",
    "long_iron",
    "mid_iron",
    "short_iron",
    "wedge",
)


@lru_cache(maxsize=1)
def _pipeline_run_fractions() -> dict[str, float]:
    """Mean (rest - carry) / carry per full-swing skill on the real pipeline."""
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for flight pipeline")
    family, sim = get_skill_family("tour"), PipelineLandingSimulator(landing_state=True)
    z = standard_normals(0, 12)
    frac = {}
    for skill in _FULL_SWING_SKILLS:
        land = sim(_impacts(family.skills[skill].distribution, z), skill, (0.0, 0.0))
        rest = GROUND(land, "fairway", "flat", skill)
        frac[skill] = float((rest[:, 0] - land[:, 0]).mean() / land[:, 0].mean())
    return frac


def test_roll_fraction_falls_from_driver_to_short_iron_on_the_pipeline() -> None:
    """Roll fraction falls monotonically from driver to short iron."""
    frac = _pipeline_run_fractions()
    chain = [frac[s] for s in _FULL_SWING_SKILLS[:-1]]
    assert all(a >= b for a, b in zip(chain, chain[1:], strict=False)), frac


def test_roll_fraction_falls_from_short_iron_to_wedge_on_the_pipeline() -> None:
    """#15443: with each skill's own club spin cap the wedge lands with more
    backspin than the short iron, so it checks up and runs a smaller fraction."""
    frac = _pipeline_run_fractions()
    assert frac["short_iron"] >= frac["wedge"], frac


def test_green_side_skills_keep_carry_and_the_list_is_hashed() -> None:
    """Green-side skills keep carry and their list is hashed in provenance."""
    for skill in ("pitch", "chip", "bunker"):
        rest = GROUND(_landing(), "fairway", "flat", skill)
        np.testing.assert_array_equal(rest, _landing()[:, :2])
    assert set(GROUND.provenance["ground_carry_only_skills"]) == {
        "pitch",
        "chip",
        "bunker",
    }
    assert _prov_table(GROUND).provenance["ground_carry_only_skills"]


def test_ground_provenance_cannot_override_table_keys() -> None:
    """Ground provenance cannot override the table's own provenance keys."""

    class _Clash:
        provenance = {"seed": 99}

        def __call__(self, landing, lie, slope, skill):
            """Forward to the real ground model while clashing on provenance."""
            return GROUND(landing, lie, slope, skill)

    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    with pytest.raises(ContractViolationError):
        build_grid_table(family, _stub_sim, grid, n=8, ground=_Clash())
