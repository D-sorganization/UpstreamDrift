"""TDD tests for WP-2 signed F0 launch handoff trace and dt convergence gate (#15805)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d import ball as bs_ball
from bunkershot3d import solvers as bs_solvers
from bunkershot3d.geometry import get_preset
from bunkershot3d.metrics import divot_metrics
from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers import launch_handoff_trace as lht
from src.tools.bunker_shot_gui.bridge import build_head, entry_kinematics, strike_trace
from src.tools.bunker_shot_gui.design import SwingSetup
from src.tools.bunker_shot_gui.model import _sand_delivery, _strike_scene

pytestmark = pytest.mark.unit


def _pulse(
    force_n: tuple[float, float, float], dur_s: float, n: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a padded force pulse over a known contact window."""
    dt = dur_s / float(n)
    t = np.arange(n + 4, dtype=np.float64) * dt
    f = np.zeros((n + 4, 3), dtype=np.float64)
    m = np.zeros(n + 4, dtype=bool)
    m[2 : 2 + n], f[2 : 2 + n] = True, np.asarray(force_n, dtype=np.float64)
    return t, f, m


def _run_acushnet(
    dt_s: float,
) -> tuple[bs_solvers.ShotResult, bs_ball.BallLaunchResult]:
    """Run F0 acushnet_example_1 at ``dt_s`` and compute splash launch."""
    g = get_preset("acushnet_example_1").geometry
    sand, swing = playing_condition(PlayingCondition.FIRM), SwingSetup(25.0)
    b = build_head(g, 36, 15)
    solver = bs_solvers.DRFTSolver(
        material=bs_solvers.MaterialResponse.from_sand_state(sand)
    )
    kin, cfg = entry_kinematics(b, swing), bs_solvers.ShotSettings(time_step_s=dt_s)
    shot = bs_solvers.simulate_shot(
        solver,
        b.elements_body,
        head_mass_kg=g.head_mass_kg,
        kinematics=kin,
        settings=cfg,
        sole_reference_body_m=b.sole_reference_body_m,
    )
    tr = strike_trace(shot)
    assert tr is not None
    div = divot_metrics(
        tr,
        b.head_model,
        _strike_scene(swing.ball_depth_m),
        width_m=g.sole_width_m,
        bulk_density_kg_m3=sand.bulk_density_kg_m3,
        friction_angle_deg=sand.friction_angle_deg,
    )
    lie = bs_ball.BallLie(depth_m=swing.ball_depth_m)
    st = bs_ball.BunkerShotState(
        56.0, lie, _sand_delivery(shot, div, sand), g.head_mass_kg
    )
    return shot, bs_ball.compute_bunker_launch(st)


def test_trace_f0_launch_handoff_valid_si_transfer() -> None:
    """Verify signed impulse integration and healthy momentum transfer pass."""
    times, forces, mask = _pulse((200.0, 0.0, 100.0), 0.010)
    tr = lht.trace_f0_launch_handoff(
        times, forces, (10.0, 0.0, 10.0), contact_mask=mask
    )
    assert isinstance(tr, lht.LaunchHandoffTrace) and tr.passed and tr.faults == ()
    assert not tr.sign_inverted and not tr.collapsed_transfer
    assert not tr.over_unity_transfer
    assert math.isclose(tr.ball_mass_kg, lht.DEFAULT_BALL_MASS_KG, rel_tol=1e-12)
    assert math.isclose(tr.impulse_club_n_s[0], 2.0, rel_tol=1e-9)
    assert math.isclose(tr.impulse_club_n_s[2], -1.0, rel_tol=1e-9)
    assert tr.horizontal_alignment > 0.99 and 0.20 < tr.transfer_ratio < 0.40


def test_trace_f0_launch_handoff_flags_baseline_collapsed_transfer() -> None:
    """Flag the 0.118 N*s vs 2.22 N*s collapsed momentum transfer defect."""
    j_mag, target_p = 2.22, 0.118
    t, f = np.linspace(0.0, 0.010, 11), np.zeros((11, 3))
    f[:, 0], f[:, 2] = (j_mag * 0.8) / 0.010, (j_mag * 0.6) / 0.010
    vm = target_p / lht.DEFAULT_BALL_MASS_KG
    v = (vm * 0.56, 0.0, vm * math.sqrt(1.0 - 0.56**2))
    tr = lht.trace_f0_launch_handoff(t, f, v)
    assert not tr.passed and tr.collapsed_transfer and not tr.sign_inverted
    assert math.isclose(tr.impulse_club_mag_n_s, j_mag, rel_tol=1e-9)
    assert math.isclose(tr.ball_momentum_mag_n_s, target_p, rel_tol=1e-9)
    assert any("collapsed" in x.lower() for x in tr.faults)


def test_trace_f0_launch_handoff_flags_sign_and_conservation_faults() -> None:
    """Detect sign inversions, over-unity momentum gain, and missing transfer."""
    t, f = np.linspace(0.0, 0.010, 11), np.zeros((11, 3))
    f[:, 0], f[:, 2] = 200.0, 100.0
    ih = lht.trace_f0_launch_handoff(
        t, f, (10.0, 0.0, 10.0), head_to_flight_frame=False
    )
    assert not ih.passed and ih.sign_inverted and ih.horizontal_alignment < 0.0
    iv = lht.trace_f0_launch_handoff(t, f, (10.0, 0.0, -5.0))
    assert not iv.passed and iv.sign_inverted
    assert any("vertical" in x.lower() for x in iv.faults)
    f[:, 0], f[:, 2] = 50.0, 0.0
    over = lht.trace_f0_launch_handoff(t, f, (20.0, 0.0, 20.0))
    assert not over.passed and over.over_unity_transfer
    missing = lht.trace_f0_launch_handoff(t, f, (0.0, 0.0, 0.0))
    assert not missing.passed and missing.collapsed_transfer


def test_trace_shot_launch_handoff_real_f0_acushnet_example_1() -> None:
    """Run F0 DRFT acushnet_example_1 and trace signed handoff vs splash launch."""
    shot, launch = _run_acushnet(2.5e-4)
    h = lht.trace_f0_launch_handoff(
        shot.times_s,
        shot.forces_n,
        launch.ball_velocity,
        contact_mask=shot.active_areas_m2 > 0.0,
    )
    assert not h.sign_inverted and h.horizontal_alignment > 0.9
    assert h.impulse_club_mag_n_s > 1.0 and h.ball_momentum_mag_n_s < 0.25
    assert h.collapsed_transfer and not h.passed
    entry = lht.verify_launch_carry_reproduction(
        launch,
        0.578,
        flight_carry_fn=lambda r: 0.578 if r.ball_speed_m_s > 0.0 else 0.0,
    )
    assert entry.passed and entry.flight_model == "flight_entry"


def test_verify_launch_carry_reproduction_ballistic_and_flight_entry() -> None:
    """Verify launch carry reproduction passes on match and fails on mismatch."""
    v = (1.44123, 0.0, 2.13671)
    exp = math.hypot(v[0], v[1]) * (2.0 * v[2]) / lht.DEFAULT_GRAVITY_M_S2
    ok = lht.verify_launch_carry_reproduction(v, exp)
    assert isinstance(ok, lht.LaunchCarryCheck) and ok.passed and ok.fault is None
    assert math.isclose(ok.replayed_carry_m, exp, rel_tol=1e-9)
    bad = lht.verify_launch_carry_reproduction(v, 12.0)
    assert not bad.passed and bad.abs_error_m > 10.0 and bad.fault is not None


def test_verify_timestep_halving_convergence_gate() -> None:
    """Pass within 2% relative speed change and fail closed above 2%."""
    ok = lht.verify_timestep_halving_convergence(lambda dt: 10.0 + 50.0 * dt, 2.5e-4)
    assert isinstance(ok, lht.TimestepConvergenceReport) and ok.passed
    assert ok.rel_diff <= 0.02
    bad = lht.verify_timestep_halving_convergence(lambda dt: 10.0 + 2000.0 * dt, 2.5e-4)
    assert not bad.passed and bad.rel_diff > 0.02 and bad.fault is not None
    with pytest.raises(bs_solvers.SolverError):
        lht.verify_timestep_halving_convergence(
            lambda dt: 10.0 + 2000.0 * dt, 2.5e-4, strict=True
        )
    nan_rep = lht.verify_timestep_halving_convergence(
        lambda dt: float("nan") if dt < 2e-4 else 10.0, 2.5e-4
    )
    assert not nan_rep.passed and nan_rep.fault is not None
    with pytest.raises(bs_solvers.SolverInputError):
        lht.verify_timestep_halving_convergence(lambda dt: 10.0, dt_s=0.0)


def test_verify_timestep_halving_convergence_real_f0_shot() -> None:
    """Confirm F0 acushnet_example_1 launch speed satisfies 2% dt-halving gate."""
    rep = lht.verify_timestep_halving_convergence(
        lambda dt: _run_acushnet(dt)[1], 2.5e-4
    )
    assert rep.passed and rep.rel_diff <= 0.02
    assert bs_ball.BallProperties().mass_kg == lht.DEFAULT_BALL_MASS_KG
