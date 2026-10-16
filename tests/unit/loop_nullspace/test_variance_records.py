"""NSX-90 (#12719): evidence typing of kernels and variance (NSX-81 conditioning)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import variance_propagation as vp
from src.shared.python.loop_nullspace import variance_records as vr
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import LinearValidity
from src.shared.python.loop_nullspace.registry import load_registry
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)
from tests.unit.loop_nullspace._variance_fixtures import noise, pendulum_dynamics

pytestmark = pytest.mark.unit

MODEL = ModelProvenance("pendulum-fixture", "1")
INPUT = InputProvenance("fixture", "nsx80-golden")


def _spec(*, sdn: bool) -> DisturbanceSpec:
    return DisturbanceSpec(
        name="torque_sdn" if sdn else "torque_add",
        stage=InjectionStage.TORQUE,
        channels=("u1", "u2"),
        units="N",
        covariance=np.diag([0.04, 0.09]) * 0.1,
        spectrum=Spectrum.WHITE,
        signal_gain=(0.1, 0.25) if sdn else None,
        signal_bounds=((-50.0, 50.0),) * 2 if sdn else None,
    )


def _decl(*, sdn=False, m2=False, m3=False) -> vr.VarianceDeclaration:
    mech = vr.declared_mechanisms(
        [_spec(sdn=sdn)], geometric_preload=m2, muscle_impedance=m3
    )
    return vr.VarianceDeclaration(
        mechanisms=mech,
        noise_hypotheses=vr.noise_hypotheses([_spec(sdn=sdn)]),
        model_provenance=MODEL,
        input_provenance=INPUT,
    )


def _problem(z=0.0):
    nz = noise()
    dyn = pendulum_dynamics(z, nz["dt"])
    return vp.VarianceProblem(
        dynamics=dyn,
        initial_covariance=nz["p0"],
        step_covariance=np.diag([0.04, 0.09])[None],
        outcome_rows=nz["c"][None, :],
        noise_classes=(vp.InputClass("torque", (0, 1)),),
        initial_classes=(vp.InputClass("state", (0, 1)),),
    )


def test_additive_noise_variance_is_invariant_with_conditioning() -> None:
    rec = vr.assess_variance(_problem(), _decl(), outcome_units="rad").variance
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.hypotheses == ()
    assert "H-NOISE-ADDITIVE" in rec.conditioning
    assert rec.units == "(rad)^2"
    assert rec.validation_status is ValidationStatus.UNVALIDATED


def test_signal_dependent_noise_variance_is_family_dependent() -> None:
    rec = vr.assess_variance(_problem(), _decl(sdn=True), outcome_units="rad").variance
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert "H-NOISE-SIGNAL-DEPENDENT" in rec.hypotheses


def test_preload_makes_additive_variance_family_dependent() -> None:
    rec = vr.assess_variance(_problem(), _decl(m2=True), outcome_units="rad").variance
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert rec.hypotheses == ("H-GEOMETRIC-PRELOAD",)
    assert "H-NOISE-ADDITIVE" in rec.conditioning


def test_muscle_impedance_declares_its_hypothesis() -> None:
    rec = vr.assess_variance(_problem(), _decl(m3=True), outcome_units="rad").variance
    assert "H-MUSCLE-SHORT-RANGE-STIFFNESS" in rec.hypotheses


def test_kernels_are_invariant_given_model_unless_a_mechanism_moves_them() -> None:
    plain = vr.assess_variance(_problem(), _decl(sdn=True), outcome_units="rad")
    assert plain.kernel_input.allocation_dependence is AllocationDependence.INVARIANT
    assert plain.kernel_x0.allocation_dependence is AllocationDependence.INVARIANT
    assert plain.kernel_x0.hypotheses == ()
    m2 = vr.assess_variance(_problem(), _decl(m2=True), outcome_units="rad")
    assert (
        m2.kernel_input.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    )
    assert m2.kernel_input.hypotheses == ("H-GEOMETRIC-PRELOAD",)


def test_every_emitted_hypothesis_id_exists_in_the_registry() -> None:
    reg = load_registry()
    for decl in (_decl(), _decl(sdn=True), _decl(m2=True, m3=True)):
        res = vr.assess_variance(_problem(), decl, outcome_units="rad")
        for rec in (res.variance, res.kernel_x0, res.kernel_input):
            for hid in (*rec.hypotheses, *rec.conditioning):
                reg.hypothesis(hid)


def test_without_a_gate_the_result_is_marked_not_assessed() -> None:
    res = vr.assess_variance(_problem(), _decl(), outcome_units="rad")
    assert res.linear_validity is LinearValidity.NOT_ASSESSED
    assert res.nonlinearity is None
    assert "saturation" in res.non_claims and "missed impacts" in res.non_claims


def test_records_above_the_gate_are_flagged_and_keep_their_values() -> None:
    prob = _problem()
    kernels = vp.adjoint_kernels(prob.dynamics, prob.outcome_rows)

    def curved(theta: np.ndarray) -> np.ndarray:
        lin = vp.stacked_sensitivity(kernels) @ theta
        return lin + 50.0 * lin**2

    gate = vr.GateSpec(rollout=curved, amplitude=3.0, gate_threshold=0.05)
    flagged = vr.assess_variance(_problem(), _decl(), outcome_units="rad", gate=gate)
    assert flagged.linear_validity is LinearValidity.NOT_LINEAR_VALID
    assert flagged.variance.values[0] > 0.0
    assert flagged.variance.validation_status is ValidationStatus.UNVALIDATED
    ok = vr.assess_variance(
        _problem(),
        _decl(),
        outcome_units="rad",
        gate=vr.GateSpec(
            rollout=lambda th: vp.stacked_sensitivity(kernels) @ th,
            amplitude=3.0,
            gate_threshold=0.05,
        ),
    )
    assert ok.linear_validity is LinearValidity.LINEAR_VALID


def test_assessment_variance_equals_forward_and_budget_total() -> None:
    prob = _problem(z=0.5)
    res = vr.assess_variance(prob, _decl(m2=True), outcome_units="rad")
    forward = prob.forward_outcome_covariance()
    assert res.variance.values[0] == pytest.approx(forward[0, 0], rel=1e-12)
    assert res.budget.total[0] == pytest.approx(forward[0, 0], rel=1e-12)


def test_declaration_rejects_unknown_mechanism_inputs() -> None:
    with pytest.raises(TypeError):
        vr.noise_hypotheses("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        vr.noise_hypotheses([])
