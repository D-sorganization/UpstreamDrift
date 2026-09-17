"""Exercise the pinned Tools impact-interval energy audit through the UD gate.

Issue #9548 (epic #9546): the provider must integrate every ledger term from
contact state and the declared law, keep the residual signed and unfudged, and
UpstreamDrift must block qualified output when that audit fails. Provider fix:
Tools #5079 (#4130); termination: Tools #5088 (#9547).
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.shared.python.physics.impact_interval_audit import (
    IMPACT_INTERVAL_CONTRACT_MODULE,
    ImpactAuditError,
    ImpactAuditTolerances,
    audit_impact_interval,
    qualified_post_impact_state,
    recompute_energy_residual,
)
from tests.shared_contracts.test_tools_provider_contracts import (
    _assert_from_tools,
    _fresh_provider_import,
)

pytestmark = pytest.mark.integration

_CLUB_MASS_KG = 0.2
_STIFFNESS_N_PER_M = 5.0e7
_TOE_OFFSET_M = np.array([0.0, 0.0, 0.02])


@pytest.fixture(scope="module")
def provider() -> ModuleType:
    with _fresh_provider_import("swing_sim"):
        module = importlib.import_module(IMPACT_INTERVAL_CONTRACT_MODULE)
        _assert_from_tools(Path(module.__file__))
        yield module


def _club(provider: ModuleType, offset: np.ndarray | None = None) -> object:
    return provider.ClubRigidBody(
        mass_kg=_CLUB_MASS_KG,
        inertia_body_kg_m2=np.diag([4.5e-4, 4.5e-4, 4.5e-4]),
        cg_to_contact_body_m=np.zeros(3) if offset is None else offset,
        cg_to_attachment_body_m=np.array([0.0, 0.90, 0.0]),
        face_normal_body=np.array([1.0, 0.0, 0.0]),
    )


def _initial(
    provider: ModuleType, offset: np.ndarray | None = None, *, pinned: bool = False
) -> object:
    impact = importlib.import_module("shared.python.swing_sim.impact")
    ball = (np.zeros(3) if offset is None else offset) + np.array(
        [impact.GOLF_BALL_RADIUS_M, 0.0, 0.0]
    )
    return provider.ImpactIntervalInitialState(
        club_position_m=np.zeros(3),
        club_orientation=np.eye(3),
        club_velocity_mps=np.array([50.0, 0.0, 0.0]),
        club_angular_velocity_rad_s=(
            np.array([0.0, 0.0, 50.0 / 0.90]) if pinned else np.zeros(3)
        ),
        ball_position_m=ball,
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )


def _restitution_law(provider: ModuleType) -> object:
    impact = importlib.import_module("shared.python.swing_sim.impact")
    reduced = (
        impact.GOLF_BALL_MASS_KG
        * _CLUB_MASS_KG
        / (impact.GOLF_BALL_MASS_KG + _CLUB_MASS_KG)
    )
    return provider.KelvinVoigtContactLaw.from_restitution(
        stiffness_n_per_m=_STIFFNESS_N_PER_M,
        restitution=0.83,
        effective_mass_kg=reduced,
    )


def _solve(
    provider: ModuleType,
    *,
    law: object,
    offset: np.ndarray | None = None,
    pinned: bool = False,
    **config: object,
) -> object:
    return provider.solve_impact_interval(
        _initial(provider, offset, pinned=pinned),
        _club(provider, offset),
        provider.ImpactIntervalConfig(contact_law=law, **config),
    )


def test_interrupted_compression_stores_energy_and_is_blocked(
    provider: ModuleType,
) -> None:
    result = _solve(
        provider,
        law=_restitution_law(provider),
        time_step_s=1.0e-7,
        maximum_time_s=3.0e-5,
        friction_coefficient=0.4,
    )
    assert result.termination is provider.ImpactTermination.TIME_LIMIT
    audit = result.audit
    assert audit.stored_contact_energy_final_j > 1.0
    assert audit.unilateral_release_energy_j == 0.0
    assert audit.dashpot_and_friction_dissipation_j > 0.0
    assert audit.energy_residual_j != 0.0
    assert audit.energy_residual_j == pytest.approx(recompute_energy_residual(audit))
    with pytest.raises(ImpactAuditError, match="did not separate") as info:
        qualified_post_impact_state(result)
    report = info.value.verdict.to_report()
    assert report["termination"] == "TIME_LIMIT"
    assert (
        report["stored_contact_energy_final_j"] == audit.stored_contact_energy_final_j
    )
    assert report["limitations"]


def test_elastic_frictionless_completion_closes_without_dissipation(
    provider: ModuleType,
) -> None:
    result = _solve(
        provider,
        law=provider.KelvinVoigtContactLaw(
            stiffness_n_per_m=_STIFFNESS_N_PER_M, damping_n_s_per_m=0.0
        ),
        friction_coefficient=0.0,
    )
    audit = result.audit
    assert audit.dissipated_energy_j == 0.0
    assert audit.unilateral_release_energy_j == 0.0
    assert audit.stored_contact_energy_final_j == 0.0
    verdict = audit_impact_interval(
        result, ImpactAuditTolerances(energy_residual_j=0.05)
    )
    assert verdict.qualified, verdict.failures


def test_unilateral_clipping_release_is_tracked_from_contact_state(
    provider: ModuleType,
) -> None:
    result = _solve(
        provider,
        law=provider.KelvinVoigtContactLaw(
            stiffness_n_per_m=_STIFFNESS_N_PER_M, damping_n_s_per_m=400.0
        ),
        friction_coefficient=0.0,
    )
    clipped = (result.compression_m > 0.0) & (result.normal_force_n == 0.0)
    assert np.any(clipped)
    stored_at_clip = (
        0.5 * _STIFFNESS_N_PER_M * float(result.compression_m[np.argmax(clipped)]) ** 2
    )
    audit = result.audit
    assert audit.unilateral_release_energy_j == pytest.approx(stored_at_clip, rel=0.2)
    assert audit.energy_residual_j == pytest.approx(recompute_energy_residual(audit))
    assert audit_impact_interval(result).qualified


@pytest.mark.parametrize("boundary", ["FREE", "TORSIONAL_GRIP"])
def test_off_center_friction_and_damped_grip_ledgers_qualify(
    provider: ModuleType, boundary: str
) -> None:
    supported = boundary == "TORSIONAL_GRIP"
    result = _solve(
        provider,
        law=_restitution_law(provider),
        offset=_TOE_OFFSET_M,
        pinned=supported,
        friction_coefficient=0.4,
        boundary=provider.BoundaryKind[boundary],
        torsional_stiffness_n_m_per_rad=2_000.0 if supported else 0.0,
        torsional_damping_n_m_s_per_rad=0.02 if supported else 0.0,
    )
    audit = result.audit
    assert audit.dashpot_and_friction_dissipation_j > 0.0
    if supported:
        assert audit.torsional_damping_dissipation_j > 0.0
        assert audit.boundary_stored_energy_j > 0.0
    verdict = audit_impact_interval(result)
    assert verdict.qualified, verdict.failures
    assert math.isnan(verdict.supported_momentum_residual_n_m_s) is not supported


def test_perturbed_force_law_stays_visible_and_blocks_output(
    provider: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    law_type = provider.KelvinVoigtContactLaw
    original = law_type.normal_force
    monkeypatch.setattr(
        law_type,
        "normal_force",
        lambda self, c, r: float(1.05 * original(self, c, r)),
    )
    result = _solve(provider, law=_restitution_law(provider))
    assert result.contact_completed
    assert abs(result.audit.energy_residual_j) > 0.5
    with pytest.raises(ImpactAuditError, match="energy residual"):
        qualified_post_impact_state(result)


def test_halving_dt_converges_free_and_supported_residuals(
    provider: ModuleType,
) -> None:
    def free(dt: float) -> object:
        return _solve(
            provider,
            law=provider.KelvinVoigtContactLaw(
                stiffness_n_per_m=_STIFFNESS_N_PER_M, damping_n_s_per_m=0.0
            ),
            time_step_s=dt,
            friction_coefficient=0.0,
        ).audit

    def supported(dt: float) -> object:
        return _solve(
            provider,
            law=_restitution_law(provider),
            offset=_TOE_OFFSET_M,
            pinned=True,
            time_step_s=dt,
            friction_coefficient=0.4,
            boundary=provider.BoundaryKind.TORSIONAL_GRIP,
            torsional_stiffness_n_m_per_rad=2_000.0,
            torsional_damping_n_m_s_per_rad=0.02,
        ).audit

    coarse, fine = free(2.0e-7), free(1.0e-7)
    assert abs(coarse.energy_residual_j) > abs(fine.energy_residual_j)
    assert abs(fine.energy_residual_j) < ImpactAuditTolerances().energy_residual_j
    assert fine.linear_momentum_residual_n_s < 1.0e-9
    coarse_s, fine_s = supported(2.0e-7), supported(1.0e-7)
    assert (
        coarse_s.supported_momentum_residual_n_m_s
        > fine_s.supported_momentum_residual_n_m_s
    )
    assert (
        fine_s.supported_momentum_residual_n_m_s
        < ImpactAuditTolerances().supported_momentum_residual_n_m_s
    )
