"""Native equilibrium alone cannot admit a source's passive-load readiness."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
    MusclePassiveLimits,
    PassiveReadinessError,
    PassiveReadinessPolicy,
    audit_native_passive_readiness,
    require_native_passive_readiness,
)

pytestmark = pytest.mark.unit


def _model(length: float = 0.32, law: str = "Thelen2003Muscle") -> tuple[Any, Any, Any]:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    body = osim.Body("body", 1, osim.Vec3(0), osim.Inertia(0.1))
    joint = osim.SliderJoint("joint", model.getGround(), body)
    joint.updCoordinate().setDefaultValue(length)
    model.addBody(body)
    model.addJoint(joint)
    muscle = getattr(osim, law)("muscle", 1000, 0.1, 0.1, 0)
    muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)
    model.finalizeConnections()
    state = model.initSystem()
    muscle.setActivation(state, 0.05)
    model.equilibrateMuscles(state)  # Explicit preparation, before the audit.
    return model, state, muscle


def _policy(model: Any, *, passive_limit: float = 0.1) -> PassiveReadinessPolicy:
    return PassiveReadinessPolicy(
        loaded_model_sha256=hashlib.sha256(model.dump().encode()).hexdigest(),
        preparation_scope="synthetic initial SliderJoint fixture; not physiological",
        limits=(
            MusclePassiveLimits(
                path="/forceset/muscle",
                normalized_fiber_range=(0.1, 4.0),
                max_abs_passive_elastic_force_ratio=passive_limit,
                max_abs_tendon_force_ratio=100.0,
                expected_ignore_tendon_compliance=False,
                expected_ignore_activation_dynamics=False,
                provenance="test_native_passive_readiness.py synthetic threshold",
            ),
        ),
    )


@pytest.mark.parametrize("law", ["Thelen2003Muscle", "Millard2012EquilibriumMuscle"])
def test_native_equilibrium_can_balance_excessive_passive_force(law: str) -> None:
    model, state, muscle = _model(law=law)
    model.realizeDynamics(state)
    residual = muscle.getFiberForceAlongTendon(state) - muscle.getTendonForce(state)
    assert abs(residual) < 1e-3
    assert muscle.getPassiveFiberForce(state) / muscle.getMaxIsometricForce() > 0.1
    policy = _policy(model)
    observed = audit_native_passive_readiness(model, state, policy)
    assert not observed.within_declared_limits
    assert "passive-elastic-force-limit:/forceset/muscle" in observed.blockers
    with pytest.raises(PassiveReadinessError, match="passive-elastic-force-limit"):
        require_native_passive_readiness(model, state, policy)


def test_missing_policy_is_unavailable_despite_successful_equilibrium() -> None:
    model, state, _ = _model()
    observed = audit_native_passive_readiness(model, state)
    assert observed.blockers == ("passive-policy-unavailable",)
    assert not observed.within_declared_limits
    with pytest.raises(PassiveReadinessError, match="unavailable"):
        require_native_passive_readiness(model, state)


def test_native_damping_cannot_hide_large_passive_elastic_force() -> None:
    model, state, muscle = _model(0.25, "Millard2012EquilibriumMuscle")
    # Explicit non-steady native state: force equilibrium chooses fiber velocity.
    muscle.setFiberLength(state, 0.2)
    model.realizeDynamics(state)
    assert abs(muscle.getPassiveFiberForce(state)) < 1e-8
    assert muscle.getPassiveFiberElasticForce(state) > 1000
    assert muscle.getPassiveFiberDampingForce(state) < -1000
    audit = audit_native_passive_readiness(model, state, _policy(model))
    observed = audit.muscles[0]
    assert observed.passive_elastic_fiber_force_n == muscle.getPassiveFiberElasticForce(
        state
    )
    assert observed.passive_damping_fiber_force_n == muscle.getPassiveFiberDampingForce(
        state
    )
    assert observed.passive_force_cancellation_n > 1000
    assert observed.fiber_velocity_m_per_s == muscle.getFiberVelocity(state)
    assert audit.qualification == "not-qualified-for-muscle-matching"
    with pytest.raises(PassiveReadinessError, match="passive.*force-limit"):
        require_native_passive_readiness(model, state, _policy(model))


def test_within_synthetic_limits_is_never_full_matching_qualification() -> None:
    model, state, _ = _model(0.21)
    policy = _policy(model, passive_limit=100.0)
    before = model.getStateVariableValues(state)
    observed = require_native_passive_readiness(model, state, policy)
    assert observed.within_declared_limits
    assert observed.qualification == "not-qualified-for-muscle-matching"
    assert observed.policy_sha256 == policy.identity_sha256
    assert observed.native_state.qualification == "not-qualified-for-native-restart"
    assert len(observed.observation_sha256) == 64
    after = model.getStateVariableValues(state)
    assert [before.get(i) for i in range(before.size())] == [
        after.get(i) for i in range(after.size())
    ]
    with pytest.raises(ValueError, match="init=False"):
        replace(observed, qualification="qualified")


@pytest.mark.parametrize("mutation", ["missing", "extra", "foreign"])
def test_policy_must_cover_exact_native_model(mutation: str) -> None:
    model, state, _ = _model(0.21)
    policy = _policy(model, passive_limit=100.0)
    if mutation == "missing":
        policy = replace(policy, limits=())
    elif mutation == "extra":
        policy = replace(
            policy,
            limits=policy.limits + (replace(policy.limits[0], path="/forceset/extra"),),
        )
    else:
        policy = replace(policy, loaded_model_sha256="a" * 64)
    with pytest.raises(PassiveReadinessError, match="policy"):
        require_native_passive_readiness(model, state, policy)


@pytest.mark.parametrize("mode", ["disabled", "overridden"])
def test_hidden_or_disabled_muscle_actuation_cannot_pass(mode: str) -> None:
    model, state, muscle = _model(0.21)
    if mode == "disabled":
        muscle.setAppliesForce(state, False)
    else:
        muscle.overrideActuation(state, True)
        muscle.setOverrideActuation(state, 0)
    policy = _policy(model, passive_limit=100.0)
    with pytest.raises(PassiveReadinessError, match=mode):
        require_native_passive_readiness(model, state, policy)


@pytest.mark.parametrize("option", ["activation", "tendon"])
def test_runtime_muscle_options_must_match_policy(option: str) -> None:
    model, state, muscle = _model(0.21)
    policy = _policy(model, passive_limit=100.0)
    if option == "activation":
        muscle.setIgnoreActivationDynamics(state, True)
    else:
        muscle.setIgnoreTendonCompliance(state, True)
    assert (
        hashlib.sha256(model.dump().encode()).hexdigest() == policy.loaded_model_sha256
    )
    with pytest.raises(PassiveReadinessError, match="muscle-policy-options"):
        require_native_passive_readiness(model, state, policy)


@pytest.mark.parametrize("quantity", ["fiber", "tendon"])
def test_each_declared_metric_is_enforced(quantity: str) -> None:
    model, state, _ = _model(0.32)
    policy = _policy(model, passive_limit=100.0)
    limit = policy.limits[0]
    if quantity == "fiber":
        limit = replace(limit, normalized_fiber_range=(0.1, 0.2))
    else:
        limit = replace(limit, max_abs_tendon_force_ratio=0.0)
    policy = replace(policy, limits=(limit,))
    with pytest.raises(PassiveReadinessError, match=f"{quantity}.*limit"):
        require_native_passive_readiness(model, state, policy)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_policy_rejects_invalid_limits(value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        MusclePassiveLimits(
            "/forceset/m",
            (0.1, 2.0),
            value,
            2.0,
            False,
            False,
            "explicit synthetic source",
        )


def test_policy_requires_provenance_and_immutable_complete_limits() -> None:
    limit = MusclePassiveLimits(
        "/forceset/m", (0.1, 2.0), 1.0, 2.0, False, False, "synthetic"
    )
    with pytest.raises(ValueError, match="provenance"):
        replace(limit, provenance="")
    with pytest.raises(TypeError, match="tuple"):
        PassiveReadinessPolicy("a" * 64, "synthetic", [limit])
    with pytest.raises(ValueError, match="duplicate"):
        PassiveReadinessPolicy("a" * 64, "synthetic", (limit, limit))


def test_native_boundary_rejects_fabricated_model_and_state() -> None:
    pytest.importorskip("opensim")
    with pytest.raises(TypeError, match="native"):
        require_native_passive_readiness(object(), object())


def test_nonfinite_native_state_time_cannot_produce_a_readiness_receipt() -> None:
    model, state, _ = _model(0.21)
    state.setTime(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        require_native_passive_readiness(model, state, _policy(model))


def test_policy_identity_binds_source_limits_and_runtime_options() -> None:
    limit = MusclePassiveLimits(
        "/forceset/m", (0.1, 2.0), 1.0, 2.0, False, False, "synthetic"
    )
    policy = PassiveReadinessPolicy("a" * 64, "synthetic scope", (limit,))
    for changed in (
        replace(limit, provenance="a different source"),
        replace(limit, max_abs_passive_elastic_force_ratio=0.5),
        replace(limit, expected_ignore_tendon_compliance=True),
    ):
        assert (
            replace(policy, limits=(changed,)).identity_sha256 != policy.identity_sha256
        )
    with pytest.raises(TypeError, match="boolean"):
        replace(limit, expected_ignore_activation_dynamics=1)
