"""Shared pendulum models for the NSX-92 noise-aware allocation tests.

The redundantly actuated planar pendulum of the NSX-80 goldens: two world-frame
actuator forces, one closure multiplier, allocation ``u(z) = u_p - q z``.  The
tangent chain is the real NSX-84 linearisation (reaction re-solved, geometric
preload included) turned into the explicit-Euler ``(A, B)`` of NSX-90.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.shared.python.loop_nullspace.decomposition import (
    CharacteristicScales,
    Decomposition,
    DimensionAccounting,
)
from src.shared.python.loop_nullspace.evidence import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.noise_aware_model import (
    NoiseAwareModel,
    fit_affine_dynamics,
)
from src.shared.python.loop_nullspace.variance_records import (
    VarianceDeclaration,
    declared_mechanisms,
    noise_hypotheses,
)
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum
from tests.unit.loop_nullspace._variance_fixtures import noise, pendulum_dynamics

DT = 0.1
Q = Pendulum().state()[0]
U_P = Pendulum.u_particular
LAM_P = -91.0 / 5.0
SCALE = np.array([20.0, 35.0])
CHANNELS = ("u1", "u2")


def decomposition() -> Decomposition:
    """``x = (u1, u2, lambda) = x_p + N z`` with ``N = (-q, 1)`` (NSX-80)."""
    n_vec = np.array([-Q[0], -Q[1], 1.0]).reshape(3, 1)
    counts = DimensionAccounting(
        n=2, n_a=2, n_u=0, m=1, n_g=0, dim_x=3, rank_A=2, nullity=1,
        rank_J_c=1, rank_J_cu_T=0, multiplier_gauge=0,
        attainable_torque_variation=1, rank_stacked=2, rank_stacked_u=2,
        multiplier_gauge_fg_free=0, attainable_torque_variation_fg_free=1,
    )  # fmt: skip
    return Decomposition(
        x_p=np.array([U_P[0], U_P[1], LAM_P]),
        N=n_vec,
        counts=counts,
        singular_values_scaled=np.ones(1),
        condition_number=1.0,
        equality_residual=0.0,
        metric_label="identity",
        scales=CharacteristicScales(),
    )


def frame(cap: float = 50.0) -> FeasibleSet:
    """Torque capacity ``|u_i| <= cap`` on the one-dimensional family."""
    dec = decomposition()
    rows = []
    for i in range(2):
        up = FeasibleRow(
            dec.N[i], cap - dec.x_p[i], f"T:{i}:up", "H-TORQUE-CAPACITY", "torque"
        )
        lo = FeasibleRow(
            -dec.N[i], cap + dec.x_p[i], f"T:{i}:lo", "H-TORQUE-CAPACITY", "torque"
        )
        rows += [up, lo]
    return dataclasses.replace(FeasibleSet.from_rows(1, rows), decomposition=dec)


def times(n: int = 1) -> np.ndarray:
    return DT * np.arange(n, dtype=float)


def spec(
    k: tuple[float, float] | None = (0.1, 0.25),
    sigma0: tuple[float, float] = (0.2, 0.3),
    *,
    golden_units: bool = True,
) -> DisturbanceSpec:
    """White torque noise.  ``golden_units``: the NSX-80 per-step variance
    ``sigma0^2 + k^2 u^2`` is what the golden states, i.e. ``S = dt sigma0^2``
    and ``k' = k sqrt(dt)`` after the ``S / dt`` of V6."""
    root = np.sqrt(DT) if golden_units else 1.0
    return DisturbanceSpec(
        name="torque",
        stage=InjectionStage.TORQUE,
        channels=CHANNELS,
        units="N",
        covariance=np.diag(np.asarray(sigma0) ** 2) * (DT if golden_units else 1.0),
        spectrum=Spectrum.WHITE,
        signal_gain=None if k is None else tuple(float(x) * root for x in k),
        signal_bounds=None if k is None else ((-80.0, 80.0),) * 2,
    )


def golden_spec() -> DisturbanceSpec:
    nz = noise()
    return spec(tuple(nz["k"]), tuple(nz["sigma0"]))


def declaration(sdn: bool, m2: bool, sp: DisturbanceSpec) -> VarianceDeclaration:
    return VarianceDeclaration(
        mechanisms=declared_mechanisms([sp], geometric_preload=m2),
        noise_hypotheses=noise_hypotheses([sp]),
        model_provenance=ModelProvenance("pendulum-nsx80", "1"),
        input_provenance=InputProvenance("fixture", "pendulum", "NSX-92 test"),
    )


def b_matrix(n_steps: int = 1) -> np.ndarray:
    """``B = dt G`` with the NSX-80 nominal ``G`` row (state ``(theta, omega)``)."""
    b = np.zeros((2, 2))
    b[1] = DT * np.array([0.4, 0.3])
    return np.repeat(b[None], n_steps, axis=0)


def frozen_dynamics(n_steps: int = 1) -> np.ndarray:
    """``A`` with ``F10 = 1`` (no preload), the golden M1-only chain."""
    a = np.array([[1.0, DT], [DT * 1.0, 1.0]])
    return np.repeat(a[None], n_steps, axis=0)


def m1_model(sp: DisturbanceSpec | None = None, n_steps: int = 1) -> NoiseAwareModel:
    nz = noise()
    return NoiseAwareModel(
        a0=frozen_dynamics(n_steps),
        b=b_matrix(n_steps),
        outcome_row=nz["c"],
        p0=nz["p0"],
        spec=sp or golden_spec(),
        dt=DT,
        n_z=n_steps,
    )


def real_a_of_z(z: np.ndarray) -> np.ndarray:
    """Real NSX-84 ``A(z)`` of a single step for the feed-forward ``u(z)``."""
    return np.asarray(pendulum_dynamics(float(z[0]), DT).a)


def m2_model(sp: DisturbanceSpec | None = None) -> NoiseAwareModel:
    """Single-step chain with the real preload ``A(z)`` (affine, fitted+checked)."""
    nz = noise()
    a0, slope = fit_affine_dynamics(real_a_of_z, 1)
    return NoiseAwareModel(
        a0=a0,
        b=b_matrix(1),
        outcome_row=nz["c"],
        p0=nz["p0"],
        spec=sp or golden_spec(),
        dt=DT,
        n_z=1,
        a_slope=slope,
    )


def two_frame_nonconvex() -> tuple[NoiseAwareModel, DisturbanceSpec]:
    """Two-step synthetic chain whose variance is NOT convex in ``(z0, z1)``.

    ``A_t(z_t)`` is fully affine in its own frame's ``z_t`` (seed 13, found by a
    scan that counts midpoint-convexity violations and grid local minima).
    """
    rng = np.random.default_rng(13)
    a0 = np.repeat(np.eye(2)[None], 2, axis=0) + 0.1 * rng.standard_normal((2, 2, 2))
    slope = np.zeros((2, 2, 2, 2))
    slope[0, ..., 0] = 0.3 * rng.standard_normal((2, 2))
    slope[1, ..., 1] = 0.3 * rng.standard_normal((2, 2))
    sp = spec((0.3, 0.5), (0.2, 0.3))
    model = NoiseAwareModel(
        a0, b_matrix(2), noise()["c"], noise()["p0"], sp, DT, 2, a_slope=slope
    )
    return model, sp


def torque_of(z: np.ndarray) -> np.ndarray:
    """Torque history of stacked ``z`` for identical pendulum frames."""
    return np.array([U_P - Q * zk for zk in np.atleast_1d(z)])


def decomposition_2d() -> Decomposition:
    """Two torque-visible null directions: ``x = (u1, u2, lambda1, lambda2)``."""
    n_vec = np.zeros((4, 2))
    n_vec[0, 0] = n_vec[1, 1] = 1.0
    counts = DimensionAccounting(
        n=2, n_a=2, n_u=0, m=2, n_g=0, dim_x=4, rank_A=2, nullity=2,
        rank_J_c=2, rank_J_cu_T=0, multiplier_gauge=0,
        attainable_torque_variation=2, rank_stacked=2, rank_stacked_u=2,
        multiplier_gauge_fg_free=0, attainable_torque_variation_fg_free=2,
    )  # fmt: skip
    return Decomposition(
        x_p=np.array([U_P[0], U_P[1], 0.0, 0.0]),
        N=n_vec,
        counts=counts,
        singular_values_scaled=np.ones(2),
        condition_number=1.0,
        equality_residual=0.0,
        metric_label="identity",
        scales=CharacteristicScales(),
    )


def frame_2d(cap: float = 50.0) -> FeasibleSet:
    """Box ``|u_i| <= cap`` on the free torques (rank-2 retained space)."""
    dec = decomposition_2d()
    rows = []
    for i in range(2):
        rows.append(
            FeasibleRow(
                dec.N[i], cap - dec.x_p[i], f"T:{i}:up", "H-TORQUE-CAPACITY", "torque"
            )
        )
        rows.append(
            FeasibleRow(
                -dec.N[i], cap + dec.x_p[i], f"T:{i}:lo", "H-TORQUE-CAPACITY", "torque"
            )
        )
    return dataclasses.replace(FeasibleSet.from_rows(2, rows), decomposition=dec)


def m1_model_2d(sp: DisturbanceSpec) -> NoiseAwareModel:
    """Frozen-kernel M1 chain with a two-dimensional null space (n_z = 2)."""
    return NoiseAwareModel(
        frozen_dynamics(), b_matrix(), noise()["c"], noise()["p0"], sp, DT, 2
    )


# ----------------------------------------------------- independent MC forward model
def strong_spec() -> DisturbanceSpec:
    """Larger declared noise so the M1/M2 effect is resolvable by a campaign."""
    return spec((0.5, 1.0), (0.1, 0.1), golden_units=False)


def mc_campaign_spec(sp: DisturbanceSpec, n_trials: int, n_holdout: int):
    """One torque step plus a persistent initial-state disturbance."""
    from src.shared.python.loop_nullspace.paired_mc_spec import (
        CampaignSpec,
        OutcomeChannel,
        OutcomeKind,
    )

    x0 = DisturbanceSpec(
        name="x0",
        stage=InjectionStage.INITIAL_STATE,
        channels=("theta", "omega"),
        units="rad",
        covariance=noise()["p0"],
        persistent=True,
    )
    return CampaignSpec(
        disturbances=(sp, x0),
        n_steps=1,
        dt=DT,
        n_trials=n_trials,
        n_holdout=n_holdout,
        paired_seed=101,
        holdout_seed=202,
        outcomes=(OutcomeChannel("y", "rad", OutcomeKind.ANGLE, tolerance=0.1),),
    )


def mc_runner_factory(sp: DisturbanceSpec):
    """``(torque, z) -> runner`` using the NONLINEAR reduced pendulum step.

    The linear model never enters: the outcome is ``c.(x1 - x1_nominal)`` with
    ``x1`` from the exact reduced-coordinate step of ``_variance_fixtures``
    driven by ``u(z)`` (so the geometric preload is physical, not assumed).
    """
    from tests.unit.loop_nullspace._variance_fixtures import (
        nominal_angle,
        pendulum_nonlinear_step,
    )

    c = noise()["c"]
    x_nom = np.array([nominal_angle(), Pendulum.omega])

    def factory(torque, z):
        u_ff = np.asarray(torque)[0]
        nominal = pendulum_nonlinear_step(x_nom, np.zeros(2), u_ff, DT)

        def runner(draw):
            d0 = draw.realise("x0")
            signal = np.asarray(torque) if sp.has_signal_law else None
            w = draw.realise("torque", signal=signal)
            x1 = pendulum_nonlinear_step(x_nom + d0, w[0], u_ff, DT)
            return [float(c @ (x1 - nominal))]

        return runner

    return factory
