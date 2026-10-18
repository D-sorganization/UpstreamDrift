"""Fixture sensitivity provider for the NSX-99 tests (#12728).

Wires the NSX-90 pendulum variance problem and the NSX-95 toy comparison context
into a :class:`SensitivityProvider` for the planar report fixture.  The numbers
are invented demonstrations of the mechanisms, not golfer measurements: the
pendulum is a one-degree-of-redundancy test plant and the comparison contexts are
the NSX-92/93 toy models.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.shared.python.loop_nullspace.evidence import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.facade import EngineContext
from src.shared.python.loop_nullspace.sensitivity_stage import (
    SensitivityInputs,
    SensitivityRequest,
)
from src.shared.python.loop_nullspace.strategy_arms import ArmSpec
from src.shared.python.loop_nullspace.strategy_campaign import (
    ComparisonMode,
    MatchSettings,
)
from src.shared.python.loop_nullspace.strategy_spec import StrategyId
from src.shared.python.loop_nullspace.variance_propagation import (
    InputClass,
    VarianceProblem,
    step_covariances,
)
from src.shared.python.loop_nullspace.variance_records import (
    Mechanism,
    VarianceDeclaration,
    declared_mechanisms,
    noise_hypotheses,
)
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace import _strategy_fixtures as sf
from tests.unit.loop_nullspace._variance_fixtures import noise, pendulum_dynamics
from tests.unit.loop_nullspace.report_support import planar_engine

N_STEPS = 6
CHANNELS = ("shoulder", "elbow")
MODEL = ModelProvenance("pendulum-fixture", "1")
SOURCE = InputProvenance("fixture", "nsx99-tests")


def disturbance(*, sdn: bool = False, scale: float = 1.0) -> DisturbanceSpec:
    """White torque noise on two channels; ``sdn`` adds the M1 signal law."""
    dt = float(noise()["dt"])
    var = np.asarray(noise()["sigma0"]) ** 2 * scale
    return DisturbanceSpec(
        name="torque_sdn" if sdn else "torque_add",
        stage=InjectionStage.TORQUE,
        channels=CHANNELS,
        units="N m",
        covariance=np.diag(var) * dt,
        spectrum=Spectrum.WHITE,
        signal_gain=(0.1, 0.25) if sdn else None,
        signal_bounds=((-50.0, 50.0),) * 2 if sdn else None,
    )


def arm_specs() -> tuple[ArmSpec, ...]:
    """S1 (free and capped), S2 and S7; the rest of the catalogue stays dormant."""
    return (
        ArmSpec(StrategyId.S1, "S1 free"),
        ArmSpec(StrategyId.S2, "S2 impedance", impedance=sf.impedance()),
        ArmSpec(
            StrategyId.S7,
            "S7 feedback",
            feedback_gain=sf.feedback_gain(),
            delay_steps=2,
        ),
    )


def inputs(
    request: SensitivityRequest,
    *,
    z: float = 0.0,
    with_contexts: bool = True,
    outcome: str = "face_angle",
) -> SensitivityInputs:
    """The pendulum variance problem for the declared disturbances."""
    spec = request.spec
    nz = noise()
    dt = float(nz["dt"])
    first = spec.disturbances[0]
    signals = None
    if first.has_signal_law:
        signals = np.tile(np.array([1.0, 2.0]), (N_STEPS, 1))
    problem = VarianceProblem(
        dynamics=pendulum_dynamics(z, dt, n_steps=N_STEPS),
        initial_covariance=np.asarray(nz["p0"]),
        step_covariance=step_covariances(first, dt, N_STEPS, signals),
        outcome_rows=np.asarray(nz["c"])[None, :],
        noise_classes=(InputClass("torque", (0, 1)),),
        initial_classes=(InputClass("state", (0, 1)),),
    )
    mechanisms = declared_mechanisms(
        spec.disturbances,
        geometric_preload=Mechanism.GEOMETRIC_PRELOAD in spec.mechanisms,
        muscle_impedance=Mechanism.MUSCLE_IMPEDANCE in spec.mechanisms,
    )
    declaration = VarianceDeclaration(
        mechanisms=mechanisms,
        noise_hypotheses=noise_hypotheses(spec.disturbances),
        model_provenance=MODEL,
        input_provenance=SOURCE,
    )
    context = {"contexts": (sf.comparison(),)} if with_contexts else {}
    return SensitivityInputs(
        problem=problem,
        declaration=declaration,
        outcome_names=(outcome,),
        outcome_units="rad",
        input_labels=CHANNELS,
        input_units="N m",
        step_times_s=tuple(dt * k for k in range(N_STEPS)),
        provenance={"fixture": "pendulum", "z": z, "steps": N_STEPS},
        arm_specs=arm_specs(),
        match=MatchSettings(reference_chs_mps=sf.NOMINAL_CHS),
        mode=ComparisonMode.UNCONSTRAINED,
        **context,
    )


def provider(**kwargs):
    """A ``SensitivityProvider`` closing over the fixture options."""

    def build(request: SensitivityRequest) -> SensitivityInputs:
        return inputs(request, **kwargs)

    return build


def sensitivity_engine(**kwargs) -> EngineContext:
    """The planar report engine with the fixture provider attached."""
    return dataclasses.replace(planar_engine(), sensitivity_provider=provider(**kwargs))
