"""Shared toy contexts for the NSX-95 strategy-catalogue tests (#12724).

Three small representations of the same kind of question ("what does a strategy
do to the face-angle variance"), each built from the fixtures the dependencies
already use:

* the redundantly actuated planar pendulum of NSX-80/92 (allocation strategies),
* the double integrator of NSX-93 (impedance and delayed feedback),
* the antagonistic pair of NSX-94 (co-contraction).

The numbers are invented fixture values that exercise the mechanisms; none is a
golfer measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.shared.python.loop_nullspace.evidence import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_linear import LayerLinearization
from src.shared.python.loop_nullspace.muscle_cocontraction import SweepConfig
from src.shared.python.loop_nullspace.muscle_fiber import total_activation
from src.shared.python.loop_nullspace.muscle_noise import muscle_noise_spec
from src.shared.python.loop_nullspace.muscle_stiffness import StiffnessModel
from src.shared.python.loop_nullspace.noise_aware_front import ChsCase
from src.shared.python.loop_nullspace.strategy_arms import (
    AllocationContext,
    ComparisonContext,
    LayerContext,
    MuscleContext,
)
from src.shared.python.loop_nullspace.variance_records import VarianceDeclaration
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx
from tests.unit.loop_nullspace._muscle_fixtures import (
    PAIR_KAPPA,
    PAIR_RHO,
    pair_frame,
)

NOMINAL_CHS = 40.0
SLOW_CHS = 30.0
EFFORT_CAP = 0.16  # between the minimum effort (0.106) and the M1 optimum (0.19)

# ---------------------------------------------------------------- allocation


def allocation(
    *,
    m1: bool = True,
    m2: bool = False,
    k: tuple[float, float] = (0.5, 1.0),
    sigma0: tuple[float, float] = (0.1, 0.1),
) -> AllocationContext:
    """Pendulum allocation context; ``m1``/``m2`` pick the declared mechanisms."""
    sp = fx.spec(k if m1 else None, sigma0, golden_units=False)
    model = fx.m2_model(sp) if m2 else fx.m1_model(sp)
    return AllocationContext(
        feasible_sets=(fx.frame(),),
        times_s=fx.times(),
        torque_scale=fx.SCALE,
        model=model,
        declaration=fx.declaration(m1, m2, sp),
    )


def slow_case(ctx: AllocationContext, label: str = "slow") -> ChsCase:
    """A slower swing: smaller torque response, lower clubhead speed (30 m/s)."""
    slow_model = dataclasses.replace(ctx.model, b=0.6 * ctx.model.b)
    return ChsCase(label, SLOW_CHS, ctx.feasible_sets, slow_model)


def nominal_case(ctx: AllocationContext, label: str = "nominal") -> ChsCase:
    return ChsCase(label, NOMINAL_CHS, ctx.feasible_sets, ctx.model)


# --------------------------------------------------------------------- layer
M, KP, KD = 2.0, 200.0, 20.0
LAYER_DT = 5e-3
LAYER_STEPS = 40
_F = np.array([[0.0, 1.0], [0.0, 0.0]])
_G = np.array([[0.0], [1.0 / M]])


def layer(*, psd: float = 0.8, kappa: float | None = None) -> LayerContext:
    """Double integrator driven by white force noise (``psd`` is the PSD)."""
    n, dt = LAYER_STEPS, LAYER_DT
    lin = LayerLinearization(
        f=np.repeat(_F[None], n, axis=0),
        g_w=np.repeat(_G[None], n, axis=0),
        g_u=np.repeat(_G[None], n, axis=0),
        dt=dt,
    )
    decl = VarianceDeclaration(
        mechanisms=frozenset(),
        noise_hypotheses=("H-NOISE-ADDITIVE",),
        model_provenance=ModelProvenance("double-integrator", "1"),
        input_provenance=InputProvenance("fixture", "nsx95-tests"),
    )
    return LayerContext(
        linearization=lin,
        times_s=dt * np.arange(n),
        selection=np.array([[1.0]]),
        basis=np.array([[1.0]]),
        offset=np.array([[0.0]]),
        p0=np.diag([1e-4, 1e-3]),
        sigma=np.full((n, 1, 1), psd / dt),
        outcome_rows=np.array([[1.0, 0.0]]),
        declaration=decl,
        outcome_units="rad",
    )


def impedance(
    t_start: float = 0.0,
    t_end: float = LAYER_STEPS * LAYER_DT,
    *,
    k: float = KP,
    d: float = KD,
    kappa: float | None = None,
) -> JointImpedance:
    window = ImpedanceWindow(t_start, t_end, np.array([k]), np.array([d]))
    return JointImpedance(
        (window,), None if kappa is None else np.array([kappa]), "fixture impedance"
    )


def feedback_gain(kp: float = KP, kd: float = KD) -> np.ndarray:
    return np.repeat(np.array([[[kp, kd]]]), LAYER_STEPS, axis=0)


# -------------------------------------------------------------------- muscle
NET_TORQUE = 8.0


def muscle() -> MuscleContext:
    """NSX-94 antagonistic pair with signal-dependent noise and stiffness."""
    from tests.unit.loop_nullspace.test_muscle_variance import DT, _model

    frame = pair_frame()
    config = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=PAIR_KAPPA),
        dt=DT,
        stiffness_model=StiffnessModel.FORCE_PROPORTIONAL,
        force_gain=PAIR_RHO,
        n_levels=9,
    )
    return MuscleContext(
        frame=frame,
        torque=np.array([NET_TORQUE]),
        index=total_activation(2),
        config=config,
        model=_model(),
    )


def comparison(
    *, chs: float = NOMINAL_CHS, with_muscle: bool = False, **alloc_kw
) -> ComparisonContext:
    return ComparisonContext(
        label="fixture",
        chs_mps=chs,
        allocation=allocation(**alloc_kw),
        layer=layer(),
        muscle=muscle() if with_muscle else None,
    )
