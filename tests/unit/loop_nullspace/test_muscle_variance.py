"""NSX-94 (#12723): the variance-versus-co-contraction U-curve.

Analytic fixture (antagonist pair, one joint ``I xi'' + B xi' + K xi = tau``):
``F = 1000 N``, ``r = 0.04 m``, ``rho = 50 1/m``, ``kappa = 0.05 sqrt(s)``,
external torque PSD ``S_ext = 1 N^2 m^2 s``, ``I = 0.1``, ``B = 4``, net torque
``8 N m`` so the antagonist offset is ``d = tau / (r F) = 0.2``.  With
``a_{1,2} = (c +- d) / 2``:

* ``K(c) = r^2 rho F c = 80 c`` (short-range stiffness, ``c in [d, 2 - d]``);
* muscle-noise torque PSD ``r^2 (kappa F)^2 (a_1^2 + a_2^2) = 2 (c^2 + d^2)``;
* stationary position variance of a damped oscillator under white torque noise of
  PSD ``S`` is ``S / (2 B K)`` (independent of ``I``), so

      Var(c) = (S_ext + 2 (c^2 + d^2)) / (2 B * 80 c),   c* = sqrt((S_ext/2 + d^2)).

Tolerances.  (a) The discrete-time stationary variance of the explicit-Euler loop
equals ``scipy.linalg.solve_discrete_lyapunov`` to ``1e-9`` relative (both are
exact linear algebra; the margin covers the finite horizon, ``exp(-2 * 4.5 * 3)``
~ 1e-16 of the slowest mode).  (b) The Euler scheme differs from the continuous
formula by the leading Lyapunov correction ``dt * F P F^T`` whose relative size
is ``dt B / I`` (= 0.08 here), so the continuous formula is checked to that
bound, not tighter.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.muscle_cocontraction import SweepConfig
from src.shared.python.loop_nullspace.muscle_fiber import total_activation
from src.shared.python.loop_nullspace.muscle_noise import muscle_noise_spec
from src.shared.python.loop_nullspace.muscle_stiffness import StiffnessModel
from src.shared.python.loop_nullspace.muscle_variance import (
    VarianceModel,
    cocontraction_variance_curve,
)
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace._muscle_fixtures import (
    DAMPING,
    EXTERNAL_PSD,
    INERTIA,
    OneJoint,
    PAIR_ARM,
    PAIR_FORCE,
    PAIR_KAPPA,
    PAIR_RHO,
    elbow_frame,
    pair_frame,
)

pytestmark = pytest.mark.unit

DT = 2e-3
STEPS = 1500
TAU = 8.0
OFFSET = TAU / (PAIR_ARM * PAIR_FORCE)  # d = 0.2


def _external() -> DisturbanceSpec:
    return DisturbanceSpec(
        name="external-torque",
        stage=InjectionStage.TORQUE,
        channels=("joint",),
        units="N m",
        covariance=np.array([[EXTERNAL_PSD]]),
        spectrum=Spectrum.WHITE,
    )


def _model() -> VarianceModel:
    manifold = tl.EuclideanManifold(1)
    routine = tl.FiniteDifferenceDerivatives(OneJoint(), manifold)
    q, v, tau = np.array([0.4]), np.array([0.0]), np.array([1.0])
    return VarianceModel(
        derivs=routine.constrained_acceleration_derivatives(q, v, tau),
        q=q,
        manifold=manifold,
        basis_field=lambda _q: np.array([[1.0]]),
        coordinate_indices=(0,),
        dt=DT,
        n_steps=STEPS,
        outcome_rows=np.array([[1.0, 0.0]]),
        external=_external(),
        initial_covariance=np.zeros((2, 2)),
        outcome_units="rad",
    )


def _config(frame, *, kappa: float, stiffness: bool, n_levels: int = 13):
    return SweepConfig(
        noise=muscle_noise_spec(frame, kappa=kappa),
        dt=DT,
        stiffness_model=StiffnessModel.FORCE_PROPORTIONAL if stiffness else None,
        force_gain=PAIR_RHO if stiffness else None,
        n_levels=n_levels,
    )


def _closed_form(level: np.ndarray) -> np.ndarray:
    noise_psd = (
        PAIR_ARM**2 * (PAIR_KAPPA * PAIR_FORCE) ** 2 * (level**2 + OFFSET**2) / 2.0
    )
    stiffness = PAIR_ARM**2 * PAIR_RHO * PAIR_FORCE * level
    return (EXTERNAL_PSD + noise_psd) / (2.0 * DAMPING * stiffness)


@pytest.fixture(scope="module")
def curve():
    frame = pair_frame()
    return cocontraction_variance_curve(
        frame,
        np.array([TAU]),
        total_activation(2),
        _config(frame, kappa=PAIR_KAPPA, stiffness=True),
        _model(),
    )


def test_variance_curve_is_u_shaped_with_the_analytic_minimum(curve) -> None:
    lo, hi = curve.sweep.level_range
    assert lo == pytest.approx(OFFSET, abs=1e-9)
    assert hi == pytest.approx(2.0 - OFFSET, abs=1e-9)
    var = curve.variance.values[:, 0]
    k_min = int(np.argmin(var))
    assert 0 < k_min < var.size - 1
    assert np.all(np.diff(var[: k_min + 1]) < 0.0)
    assert np.all(np.diff(var[k_min:]) > 0.0)
    c_star = np.sqrt(EXTERNAL_PSD / 2.0 + OFFSET**2)
    step = float(np.diff(curve.sweep.levels).max())
    assert abs(curve.sweep.levels[k_min] - c_star) <= step


def test_discrete_stationary_variance_matches_the_lyapunov_solution(curve) -> None:
    level = curve.sweep.levels
    k_joint = curve.sweep.stiffness_trace.values
    expected = []
    for c, k in zip(level, k_joint, strict=True):
        a = np.eye(2) + DT * np.array([[0.0, 1.0], [-k / INERTIA, -DAMPING / INERTIA]])
        b = DT * np.array([[0.0], [1.0 / INERTIA]])
        psd = (
            EXTERNAL_PSD
            + PAIR_ARM**2 * (PAIR_KAPPA * PAIR_FORCE) ** 2 * (c**2 + OFFSET**2) / 2.0
        )
        expected.append(solve_discrete_lyapunov(a, b @ b.T * psd / DT)[0, 0])
    np.testing.assert_allclose(curve.variance.values[:, 0], expected, rtol=1e-9)


def test_continuous_formula_holds_to_the_euler_error_bound(curve) -> None:
    ratio = curve.variance.values[:, 0] / _closed_form(curve.sweep.levels)
    assert np.max(np.abs(ratio - 1.0)) <= DT * DAMPING / INERTIA


def test_record_is_family_dependent_with_both_mechanisms(curve) -> None:
    rec = curve.variance
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert set(rec.hypotheses) == {
        "H-NOISE-SIGNAL-DEPENDENT",
        "H-MUSCLE-SHORT-RANGE-STIFFNESS",
    }
    assert curve.argmin_level == pytest.approx(
        curve.sweep.levels[int(np.argmin(rec.values[:, 0]))]
    )
    budget = curve.assessments[0].budget
    assert {"muscle_noise", "external"} <= set(budget.contributions)


def test_negative_control_without_a_mechanism_is_flat_and_invariant() -> None:
    frame = pair_frame()
    flat = cocontraction_variance_curve(
        frame,
        np.array([TAU]),
        total_activation(2),
        _config(frame, kappa=0.0, stiffness=False, n_levels=5),
        _model(),
    )
    var = flat.variance.values[:, 0]
    np.testing.assert_allclose(var, var[0], rtol=1e-12)
    assert flat.variance.allocation_dependence is AllocationDependence.INVARIANT
    assert "H-NOISE-ADDITIVE" in flat.variance.conditioning
    assert flat.argmin_level is not None


def test_not_qualified_frame_gives_unavailable_variance() -> None:
    frame = elbow_frame()
    cfg = _config(frame, kappa=PAIR_KAPPA, stiffness=True)
    ext = DisturbanceSpec(
        name="e",
        stage=InjectionStage.TORQUE,
        channels=("elbow_flexion",),
        units="N m",
        covariance=np.array([[1.0]]),
        spectrum=Spectrum.WHITE,
    )
    model = _model()
    model = VarianceModel(**{**model.__dict__, "external": ext})
    out = cocontraction_variance_curve(
        frame, np.array([80.0]), total_activation(3), cfg, model
    )
    assert out.variance.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert "NOT_QUALIFIED" in (out.variance.reason or "")
    assert out.argmin_level is None and out.assessments == ()


def test_declared_but_unavailable_stiffness_fails_closed() -> None:
    frame = pair_frame()
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=PAIR_KAPPA),
        dt=DT,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=3,
    )
    out = cocontraction_variance_curve(
        frame, np.array([TAU]), total_activation(2), cfg, _model()
    )
    assert out.variance.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert "stiffness" in (out.variance.reason or "")


def test_inconsistent_inputs_are_rejected() -> None:
    frame = pair_frame()
    cfg = _config(frame, kappa=PAIR_KAPPA, stiffness=True, n_levels=3)
    bad_dt = VarianceModel(**{**_model().__dict__, "dt": 2 * DT})
    with pytest.raises(ValueError, match="dt"):
        cocontraction_variance_curve(
            frame, np.array([TAU]), total_activation(2), cfg, bad_dt
        )
    with pytest.raises(ValueError, match="n_steps"):
        VarianceModel(**{**_model().__dict__, "n_steps": 0})
