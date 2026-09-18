"""Consume the I1/I2 impact-interval fixes through the pinned Tools provider.

Epic #9546 orders the two contact-interval defects fixed in Tools first and
their reviewed pin consumed here. The provider fixes landed as Tools #5088
(#9547: report contact completion, refuse to export unfinished contact) and
Tools #5079 (#9548: audit contact energy independently instead of zeroing
the residual). This module is the UpstreamDrift half of that consumption: it
drives the vendored solver through the exact probe the audit reproduced and
pins the behaviour a UD consumer relies on. Against the audit pin
``3d93bb2c`` every test here fails at import (``ImpactTermination`` and
``IncompleteContactError`` did not exist).

The fixture below is the audit probe, not a Tools test import: a 0.2 kg
rigid head at 50 m/s, Kelvin-Voigt contact tuned for e = 0.83, dt = 1e-7 s,
cut off by a 1e-5 s cap while the ball is still compressed under tens of
kilonewtons. Depending on the provider's own test module would tie this
contract to Tools' test layout rather than to its public façade.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.shared_contracts.test_tools_provider_contracts import (
    _assert_from_tools,
    _fresh_provider_import,
)

pytestmark = pytest.mark.integration

_HEAD_MASS_KG = 0.2
_TRUNCATING_CAP_S = 1.0e-5


def _provider() -> ModuleType:
    module = importlib.import_module("shared.python.swing_sim.impact_interval")
    _assert_from_tools(Path(module.__file__))
    return module


def _audit_probe(
    provider: ModuleType, *, maximum_time_s: float, club_speed_mps: float = 50.0
) -> Any:
    """Solve the #9547 audit probe; only the time cap and speed vary."""
    impact = importlib.import_module("shared.python.swing_sim.impact")
    ball_mass = impact.GOLF_BALL_MASS_KG
    reduced_mass = ball_mass * _HEAD_MASS_KG / (ball_mass + _HEAD_MASS_KG)
    club = provider.ClubRigidBody(
        mass_kg=_HEAD_MASS_KG,
        inertia_body_kg_m2=np.diag([4.5e-4, 4.5e-4, 4.5e-4]),
        cg_to_contact_body_m=np.zeros(3),
        cg_to_attachment_body_m=np.array([0.0, 0.90, 0.0]),
        face_normal_body=np.array([1.0, 0.0, 0.0]),
    )
    initial = provider.ImpactIntervalInitialState(
        club_position_m=np.zeros(3),
        club_orientation=np.eye(3),
        club_velocity_mps=np.array([club_speed_mps, 0.0, 0.0]),
        club_angular_velocity_rad_s=np.zeros(3),
        ball_position_m=np.array([impact.GOLF_BALL_RADIUS_M, 0.0, 0.0]),
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )
    config = provider.ImpactIntervalConfig(
        contact_law=provider.KelvinVoigtContactLaw.from_restitution(
            stiffness_n_per_m=5.0e7,
            restitution=0.83,
            effective_mass_kg=reduced_mass,
        ),
        time_step_s=1.0e-7,
        maximum_time_s=maximum_time_s,
        friction_coefficient=0.4,
    )
    return provider.solve_impact_interval(initial=initial, club=club, config=config)


def test_truncated_contact_is_reported_and_refuses_to_export() -> None:
    """I1 (#9547): still-compressed contact is not a post-impact state."""
    with _fresh_provider_import("swing_sim"):
        provider = _provider()
        result = _audit_probe(provider, maximum_time_s=_TRUNCATING_CAP_S)
        assert result.did_contact is True
        assert result.compression_m[-1] > 0.0, "probe must still be compressed"
        assert result.normal_force_n[-1] > 1.0e4, "probe must still be loaded"
        assert result.termination is provider.ImpactTermination.TIME_LIMIT
        assert result.contact_completed is False
        # The cap is a hard limit: no silent extra step past it (audit saw
        # 1.01e-5 s under a 1e-5 s cap).
        assert result.time_s[-1] <= _TRUNCATING_CAP_S + 1.0e-12
        with pytest.raises(provider.IncompleteContactError) as excinfo:
            result.to_post_impact_state()
        # The partial trace stays inspectable; only the conclusion is refused.
        assert excinfo.value.result is result
        assert excinfo.value.termination is provider.ImpactTermination.TIME_LIMIT


def test_completed_contact_and_a_miss_are_distinct_from_a_timeout() -> None:
    """No-hit, unfinished contact and completed contact are three states."""
    with _fresh_provider_import("swing_sim"):
        provider = _provider()
        completed = _audit_probe(provider, maximum_time_s=2.0e-3)
        assert completed.termination is provider.ImpactTermination.SEPARATED
        assert completed.contact_completed is True
        post = completed.to_post_impact_state()
        assert post.ball_velocity[0] > completed.club_velocity_mps[-1, 0] > 0.0

        miss = _audit_probe(
            provider, maximum_time_s=_TRUNCATING_CAP_S, club_speed_mps=0.0
        )
        assert miss.did_contact is False
        assert miss.termination is provider.ImpactTermination.NO_CONTACT
        assert miss.contact_completed is False


def test_energy_audit_keeps_stored_energy_and_a_signed_residual() -> None:
    """I2 (#9548): a deficit is not relabelled as unilateral release.

    The audit reproduced 5.777 J of "unilateral release" and a 0.0 J residual
    on this exact truncated probe. Under the fix the still-compressed spring
    energy is reported as stored, no release is claimed because no tensile
    clip happened, and the residual is the signed remainder of the declared
    ledger rather than a term defined to make it vanish.
    """
    with _fresh_provider_import("swing_sim"):
        provider = _provider()
        audit = _audit_probe(provider, maximum_time_s=_TRUNCATING_CAP_S).audit
        assert audit.unilateral_release_energy_j == 0.0
        assert audit.stored_contact_energy_initial_j == 0.0
        assert audit.stored_contact_energy_final_j > 1.0
        ledger_remainder = (
            audit.initial_kinetic_energy_j
            - audit.final_kinetic_energy_j
            + (
                audit.stored_contact_energy_initial_j
                - audit.stored_contact_energy_final_j
            )
            - audit.unilateral_release_energy_j
            - audit.dissipated_energy_j
            - audit.boundary_stored_energy_j
        )
        assert audit.energy_residual_j == pytest.approx(ledger_remainder, abs=1e-9)
        # Declared O(dt) semi-implicit tolerance; a zeroed residual would also
        # pass this bound, which is why the identity above is asserted first.
        assert abs(audit.energy_residual_j) < 0.15
        assert audit.dissipated_energy_j == pytest.approx(
            audit.dashpot_and_friction_dissipation_j
            + audit.torsional_damping_dissipation_j
        )
