"""Tests of the noise-aware allocation (NSX-92, #12721).

Oracles are independent of the allocator: the exact NSX-80 golden minimisers
(``z* = -1315/348`` for M1 and ``1180/1669`` for M1 + M2), closed-form clipping
for an active bound, a dense grid for the non-convex case, and the flat negative
control (additive noise, no preload).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import minimize

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation_spec import (  # noqa: E402
    NotStrictlyConvexError,
)
from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
)
from src.shared.python.loop_nullspace.noise_aware_allocation import (  # noqa: E402
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.noise_aware_spec import (  # noqa: E402
    NoiseAwareSettings,
    NoiseAwareStatus,
)
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx  # noqa: E402
from tests.unit.loop_nullspace._tangent_fixtures import golden  # noqa: E402

pytestmark = pytest.mark.unit

TIMES = fx.times()


def _solve(model, decl, cap=50.0, settings=None, frames=None):
    frames = frames or [fx.frame(cap)]
    return minimise_face_variance(
        frames, fx.times(len(frames)), fx.SCALE, model, decl, settings
    )


def _m1():
    sp = fx.golden_spec()
    return fx.m1_model(sp), fx.declaration(True, False, sp)


def _m2():
    sp = fx.golden_spec()
    return fx.m2_model(sp), fx.declaration(True, True, sp)


def test_m1_planar_golden_minimiser_to_1e_9() -> None:
    res = _solve(*_m1())
    gold = golden()["min_variance"]
    z_star = float(Fraction(gold["z_star"]))
    assert res.status is NoiseAwareStatus.OPTIMAL
    assert res.z[0][0] == pytest.approx(z_star, abs=1e-9)
    assert res.variance == pytest.approx(
        float(Fraction(gold["var_at_zstar"])), rel=1e-12
    )
    assert res.convex and res.global_optimum_certified
    assert res.kkt_residuals is not None and res.kkt_residuals.max() < 1e-8
    assert res.variance <= res.reference_variance
    assert res.wrap_probability_bound < 1e-12


def test_m1_plus_m2_planar_golden_is_a_certified_local_stationary_point() -> None:
    res = _solve(*_m2())
    z_star = float(Fraction(golden()["min_variance"]["z_star_closed_form"]))
    # the M1-only golden z* differs, the preload moves the optimum to 1180/1669
    assert res.status is NoiseAwareStatus.LOCAL_STATIONARY
    assert res.z[0][0] == pytest.approx(1180 / 1669, abs=1e-9)
    assert z_star == pytest.approx(-1315 / 348)
    assert not res.global_optimum_certified and not res.convex
    assert res.step_norm is not None and res.step_norm < 1e-8
    assert res.kkt_residuals.max() < 1e-6
    # interior optimum: the exact gradient vanishes
    assert res.gradient_norm < 1e-9
    assert res.variance <= res.reference_variance


def test_negative_control_additive_noise_is_flat_and_invariant() -> None:
    sp = fx.spec(None)
    model, decl = fx.m1_model(sp), fx.declaration(False, False, sp)
    res = _solve(model, decl)
    flat = golden()["additive_noise_flat"]
    assert flat["gradient_no_preload_is_zero"] is True
    assert res.status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT
    assert res.gradient_norm < 1e-12
    assert res.record.allocation_dependence is AllocationDependence.INVARIANT
    assert res.record.hypotheses == ()
    assert "H-NOISE-ADDITIVE" in res.record.conditioning
    assert res.variance == pytest.approx(res.reference_variance, rel=0, abs=0)
    assert "no variance claim" in res.reason
    # every feasible allocation has the same variance (the control is real)
    for z in (-8.0, 0.0, 6.0):
        tau = (fx.U_P - fx.Q * z)[None]
        assert model.variance(np.array([z]), tau) == pytest.approx(
            res.variance, rel=1e-14
        )


def test_additive_noise_with_declared_preload_is_not_flat() -> None:
    """V1: M2 alone (additive noise) couples the allocation to the variance."""
    sp = fx.spec(None)
    model, decl = fx.m2_model(sp), fx.declaration(False, True, sp)
    res = _solve(model, decl)
    assert res.status is NoiseAwareStatus.LOCAL_STATIONARY
    assert res.record.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert "H-GEOMETRIC-PRELOAD" in res.record.hypotheses
    assert res.variance < res.reference_variance
    # independent oracle: V is an exact quadratic in z; minimise by np.polyfit
    zs = np.linspace(-5.0, 5.0, 11)
    vs = [model.variance(np.array([z]), (fx.U_P - fx.Q * z)[None]) for z in zs]
    c2, c1, _ = np.polyfit(zs, vs, 2)
    assert res.z[0][0] == pytest.approx(-c1 / (2 * c2), abs=1e-7)


def test_active_capacity_bound_matches_the_clipped_unconstrained_optimum() -> None:
    res = _solve(*_m1(), cap=8.0)
    # u1 = 7 - 0.6 z >= -8  ->  z <= 25 ; u2 = 5 + 0.8 z >= -8 -> z >= -16.25;
    # u1 <= 8 -> z >= -5/3 ; u2 <= 8 -> z <= 3.75.   z* = -3.78 clips to -5/3.
    assert res.status is NoiseAwareStatus.OPTIMAL
    assert res.z[0][0] == pytest.approx(-5.0 / 3.0, abs=1e-9)
    assert res.kkt_residuals.max() < 1e-8
    assert res.gradient_norm > 1e-6  # active bound: the free gradient does not vanish


def test_infeasible_family_is_propagated_not_relaxed() -> None:
    res = _solve(*_m1(), cap=1.0)
    assert res.status is NoiseAwareStatus.INFEASIBLE
    assert res.z == () and res.variance is None and res.record is None


def test_effort_cap_gives_matched_effort_and_a_monotone_variance_curve() -> None:
    model, decl = _m1()
    free = _solve(model, decl)
    ref_effort = free.reference_effort
    prev = float("inf")
    for cap in (
        0.5 * free.effort,
        0.9 * free.effort,
    ):  # below the free optimum's effort
        res = _solve(model, decl, settings=NoiseAwareSettings(effort_cap=cap))
        assert res.effort <= cap * (1 + 1e-6)
        assert (
            free.variance - 1e-12 <= res.variance <= prev
        )  # more effort, less variance
        prev = res.variance
    tight = _solve(
        model, decl, settings=NoiseAwareSettings(effort_cap=ref_effort * 1.001)
    )
    assert tight.effort <= ref_effort * 1.001 * (1 + 1e-6)
    assert free.variance <= tight.variance <= tight.reference_variance + 1e-12
    # at the unconstrained optimum's effort the constraint is inactive
    loose = _solve(
        model, decl, settings=NoiseAwareSettings(effort_cap=free.effort * 1.5)
    )
    assert loose.variance == pytest.approx(free.variance, rel=1e-9)


def test_effort_cap_on_the_nonconvex_path_matches_the_convex_cone() -> None:
    model, decl = _m2()
    free = _solve(model, decl)
    cap = 0.5 * (free.effort + free.reference_effort)
    res = _solve(model, decl, settings=NoiseAwareSettings(effort_cap=cap))
    assert res.status is NoiseAwareStatus.LOCAL_STATIONARY
    assert res.effort <= cap * (1 + 1e-6)
    # independent oracle: 1-D box on z from the effort cone, minimise the quadratic
    zs = np.linspace(-20.0, 20.0, 400001)
    taus = fx.U_P[None, :] - zs[:, None] * fx.Q[None, :]
    eff = np.sum((taus / fx.SCALE) ** 2, axis=1)  # single frame, trapezoid weight 1
    ok = eff <= cap
    vs = np.array(
        [
            model.variance(np.array([z]), t[None])
            for z, t in zip(zs[ok][::40], taus[ok][::40], strict=True)
        ]
    )
    assert res.variance <= vs.min() + 1e-9


def test_non_strictly_convex_variance_is_refused_then_tie_broken_explicitly() -> None:
    # k acts on channel 0 only: no curvature along the u2 direction of a 2-D family
    sp = fx.spec((0.4, 0.0))
    model = fx.m1_model_2d(sp)
    decl = fx.declaration(True, False, sp)
    frames = [fx.frame_2d()]
    with pytest.raises(NotStrictlyConvexError, match="tie_break_effort_weight"):
        minimise_face_variance(frames, TIMES, fx.SCALE, model, decl)
    res = minimise_face_variance(
        frames,
        TIMES,
        fx.SCALE,
        model,
        decl,
        NoiseAwareSettings(tie_break_effort_weight=1e-3),
    )
    assert res.status is NoiseAwareStatus.OPTIMAL
    assert res.tie_break_applied and not res.global_optimum_certified
    # the u2 direction carries no variance: the tie-break picks its minimum-effort value 0
    assert res.torque[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert res.variance <= res.reference_variance


def test_m1_optimum_beats_every_feasible_perturbation_on_the_two_dim_family() -> None:
    sp = fx.spec((0.4, 0.7))
    model = fx.m1_model_2d(sp)
    decl = fx.declaration(True, False, sp)
    res = minimise_face_variance([fx.frame_2d(30.0)], TIMES, fx.SCALE, model, decl)
    assert res.status is NoiseAwareStatus.OPTIMAL

    def objective(z: np.ndarray) -> float:
        return model.variance(z, (fx.U_P + z)[None])

    ref = minimize(
        objective, np.array([1.0, -2.0]), method="SLSQP", options={"ftol": 1e-14}
    )
    assert res.variance <= ref.fun + 1e-12


def test_nonconvex_two_frame_case_reports_local_optima_never_global() -> None:
    model, sp = fx.two_frame_nonconvex()
    decl = fx.declaration(True, True, sp)
    cap = 12.0  # compact box z in [-8.33, 8.75]^2
    frames = [fx.frame(cap), fx.frame(cap)]
    lo, hi = (7 - cap) / 0.6, (cap - 5) / 0.8

    def v(z: np.ndarray) -> float:
        return model.variance(np.asarray(z, float), fx.torque_of(z))

    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(2000, 2, 2))
    assert any(v(0.5 * (p + q)) > 0.5 * (v(p) + v(q)) + 1e-12 for p, q in pts), (
        "the fixture must be non-convex"
    )
    grid = np.linspace(lo, hi, 161)
    grid_min = min(v([a, b]) for a in grid for b in grid)

    starts = tuple(np.array(s, float) for s in ([lo, hi], [hi, lo], [hi, hi], [lo, lo]))
    res = minimise_face_variance(
        frames, fx.times(2), fx.SCALE, model, decl, NoiseAwareSettings(starts=starts)
    )
    assert res.status is NoiseAwareStatus.LOCAL_STATIONARY
    assert not res.global_optimum_certified and not res.convex
    assert len(set(np.round(res.local_optima, 8))) >= 2, (
        "starts must expose >= 2 optima"
    )
    assert res.variance == pytest.approx(min(res.local_optima))
    assert (
        res.variance <= grid_min + 1e-6
    )  # best certified local optimum is grid-competitive
    assert res.variance <= res.reference_variance
    ref = minimize(
        v,
        np.concatenate(res.z),
        method="SLSQP",
        bounds=[(lo, hi)] * 2,
        options={"ftol": 1e-14},
    )
    assert (
        res.variance <= ref.fun + 1e-9
    )  # a first-order stationary point: no SLSQP descent


def test_declaration_must_name_the_model_mechanisms() -> None:
    sp = fx.golden_spec()
    model1 = fx.m1_model(sp)
    with pytest.raises(ValueError, match="M1 signal law"):
        _solve(model1, fx.declaration(False, False, fx.spec(None)))
    with pytest.raises(ValueError, match="M2 preload"):
        _solve(fx.m2_model(sp), fx.declaration(True, False, sp))
    with pytest.raises(TypeError):
        _solve(model1, "declaration")


def test_shape_and_type_preconditions() -> None:
    model, decl = _m1()
    with pytest.raises(TypeError):
        minimise_face_variance([fx.frame()], TIMES, fx.SCALE, "model", decl)
    with pytest.raises(ValueError, match="steps"):
        minimise_face_variance(
            [fx.frame(), fx.frame()], fx.times(2), fx.SCALE, model, decl
        )
    sp = fx.golden_spec()
    wrong_nz = fx.m1_model(sp, n_steps=2)
    with pytest.raises(ValueError, match="steps|n_z"):
        _solve(wrong_nz, decl)
    with pytest.raises(ValueError):
        NoiseAwareSettings(tie_break_effort_weight=-1.0)
    with pytest.raises(ValueError):
        NoiseAwareSettings(effort_cap=0.0)
    with pytest.raises(ValueError):
        NoiseAwareSettings(kkt_tol=0.0)


def test_record_is_typed_and_names_unverified_hypotheses_honestly() -> None:
    res = _solve(*_m1())
    rec = res.record
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert rec.units == "(rad)^2"
    assert "H-NOISE-SIGNAL-DEPENDENT" in rec.hypotheses
    assert "H-NOISE-SIGNAL-DEPENDENT" in rec.conditioning
    assert set(res.mechanisms) == {"H-NOISE-SIGNAL-DEPENDENT"}
    assert "hypothesis-conditioned" in res.non_claim
    assert isinstance(res.unverified_hypothesis_ids, tuple)
