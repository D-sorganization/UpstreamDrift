"""Exercise the public contact-interval façade and energy audit through the pinned Tools provider.

Combines contract coverage for:
- Issue #9548 (epic #9546): energy audit consumer gate, signed residuals, and
  qualified post-impact state validation.
- Issue #9549 (epic #9546): six-DOF contact-interval solver integration into
  live runs, playback, and export.
"""

from __future__ import annotations

import dataclasses
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

PINNED_PROVIDER_REVISION = "62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1"

# Channels the through-contact inspection must expose, in acceptance order:
# ball/club position, orientation, velocity, spin, force, compression and
# face/loft/twist. Each is a per-sample history aligned with ``time_s``.
_TIME_RESOLVED_CHANNELS = (
    "ball_position_m",
    "club_position_m",
    "club_orientation",
    "ball_velocity_mps",
    "club_velocity_mps",
    "ball_angular_velocity_rad_s",
    "club_angular_velocity_rad_s",
    "normal_force_n",
    "friction_force_n",
    "compression_m",
    "face_angle_deg",
    "dynamic_loft_deg",
    "twist_angle_rad",
)

_CLUB_MASS_KG = 0.2
_STIFFNESS_N_PER_M = 5.0e7
_TOE_OFFSET_M = np.array([0.0, 0.0, 0.02])
_SHAFT_LENGTH_M = 0.90
_CLUB_SPEED_MPS = 50.0
_LOFT_RAD = math.radians(10.0)


# ---------------------------------------------------------------------------
# Helpers for #9548: Energy audit contracts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers for #9549: Interval facade and integration contracts
# ---------------------------------------------------------------------------


def _interval_module() -> ModuleType:
    module = importlib.import_module("shared.python.swing_sim.impact_interval")
    _assert_from_tools(Path(module.__file__))
    return module


def _facade_club(interval: ModuleType, toe_offset_m: float = 0.0):  # type: ignore[no-untyped-def]
    """Rigid driver head with a vertical shaft; +y is up in the app frame."""
    contact = np.array([0.0, 0.0, toe_offset_m])
    return interval.ClubRigidBody(
        mass_kg=_CLUB_MASS_KG,
        inertia_body_kg_m2=np.diag([4.5e-4, 4.5e-4, 4.5e-4]),
        cg_to_contact_body_m=contact,
        cg_to_attachment_body_m=np.array([0.0, _SHAFT_LENGTH_M, 0.0]),
        face_normal_body=np.array([math.cos(_LOFT_RAD), math.sin(_LOFT_RAD), 0.0]),
    )


def _facade_initial(
    interval: ModuleType, impact: ModuleType, toe_offset_m: float = 0.0
):  # type: ignore[no-untyped-def]
    """Ball resting on the lofted face at first touch, club moving downrange."""
    club = _facade_club(interval, toe_offset_m)
    contact = np.asarray(club.cg_to_contact_body_m, dtype=float)
    normal = np.asarray(club.face_normal_body, dtype=float)
    return interval.ImpactIntervalInitialState(
        club_position_m=np.zeros(3),
        club_orientation=np.eye(3),
        club_velocity_mps=np.array([_CLUB_SPEED_MPS, 0.0, 0.0]),
        club_angular_velocity_rad_s=np.zeros(3),
        ball_position_m=contact + impact.GOLF_BALL_RADIUS_M * normal,
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )


def _facade_config(interval: ModuleType, impact: ModuleType, **overrides: object):  # type: ignore[no-untyped-def]
    reduced_mass = (
        impact.GOLF_BALL_MASS_KG
        * _CLUB_MASS_KG
        / (impact.GOLF_BALL_MASS_KG + _CLUB_MASS_KG)
    )
    parameters: dict[str, object] = {
        "contact_law": interval.KelvinVoigtContactLaw.from_restitution(
            stiffness_n_per_m=5.0e7,
            restitution=0.83,
            effective_mass_kg=reduced_mass,
        ),
        "time_step_s": 1.0e-7,
        "maximum_time_s": 2.0e-3,
        "friction_coefficient": 0.4,
    }
    parameters.update(overrides)
    return interval.ImpactIntervalConfig(**parameters)


# ---------------------------------------------------------------------------
# Test suite for #9548: Energy audit contracts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test suite for #9549: Interval facade and integration contracts
# ---------------------------------------------------------------------------


def test_interval_facade_produces_time_resolved_contact_record() -> None:
    """A single façade call yields every inspection channel plus the audit."""
    with _fresh_provider_import("swing_sim"):
        interval = _interval_module()
        impact = importlib.import_module("shared.python.swing_sim.impact")
        config = _facade_config(interval, impact)
        result = interval.solve_impact_interval(
            _facade_initial(interval, impact), _facade_club(interval), config
        )

        assert result.did_contact is True
        assert result.contact_completed is True
        assert result.termination is interval.ImpactTermination.SEPARATED

        sample_count = len(result.time_s)
        assert sample_count > 100
        assert (
            0.0
            < result.contact_duration_s
            <= (float(result.time_s[-1]) + config.time_step_s)
        )
        for name in _TIME_RESOLVED_CHANNELS:
            channel = result.channel(name)
            assert channel.shape[0] == sample_count, name
            assert np.all(np.isfinite(channel)), name
        assert result.channel("club_orientation").shape == (sample_count, 3, 3)
        with pytest.raises(ValueError, match="Unknown impact-interval channel"):
            result.channel("not_a_channel")

        # Physical shape of the record: loaded then released, never adhesive.
        assert np.max(result.compression_m) > 0.0
        assert result.compression_m[-1] <= 0.0
        assert np.min(result.normal_force_n) >= 0.0
        assert np.max(result.normal_force_n) > 1.0e3
        assert result.ball_velocity_mps[-1, 0] > _CLUB_SPEED_MPS
        assert result.ball_velocity_mps[-1, 1] > 0.0, "lofted face launches upward"

        mid = result.at_time(0.5 * result.contact_duration_s)
        assert isinstance(mid, interval.ImpactIntervalSample)
        assert mid.compression_m > 0.0
        assert mid.normal_force_n > 0.0
        assert mid.club_orientation.shape == (3, 3)

        audit = result.audit
        assert isinstance(audit, interval.ImpactIntervalAudit)
        for field in dataclasses.fields(audit):
            value = getattr(audit, field.name)
            if field.name == "supported_momentum_residual_n_m_s":
                assert math.isnan(value), "FREE boundary has no support balance"
            else:
                assert math.isfinite(value), field.name
        assert audit.integrated_normal_impulse_n_s > 0.0
        assert audit.dissipated_energy_j > 0.0
        assert abs(audit.linear_momentum_residual_n_s) < 1.0e-6


def test_boundary_selection_changes_the_solver_configuration() -> None:
    """Grip idealizations are real solver inputs, and mismatches are refused."""
    with _fresh_provider_import("swing_sim"):
        interval = _interval_module()
        impact = importlib.import_module("shared.python.swing_sim.impact")
        kinds = interval.BoundaryKind
        assert {kind.value for kind in kinds} == {"free", "pinned", "torsional_grip"}

        toe_offset_m = 0.02
        club = _facade_club(interval, toe_offset_m)
        free_initial = _facade_initial(interval, impact, toe_offset_m)
        supported_initial = dataclasses.replace(
            free_initial,
            club_angular_velocity_rad_s=np.array(
                [0.0, 0.0, _CLUB_SPEED_MPS / _SHAFT_LENGTH_M]
            ),
        )
        free = interval.solve_impact_interval(
            free_initial, club, _facade_config(interval, impact, boundary=kinds.FREE)
        )
        pinned = interval.solve_impact_interval(
            supported_initial,
            club,
            _facade_config(interval, impact, boundary=kinds.PINNED),
        )
        sprung = interval.solve_impact_interval(
            supported_initial,
            club,
            _facade_config(
                interval,
                impact,
                boundary=kinds.TORSIONAL_GRIP,
                torsional_stiffness_n_m_per_rad=2_000.0,
                torsional_damping_n_m_s_per_rad=0.02,
            ),
        )
        anchor = pinned.attachment_position_m[0]
        assert (
            np.max(np.linalg.norm(pinned.attachment_position_m - anchor, axis=1))
            < 1.0e-9
        )
        assert np.max(np.linalg.norm(free.attachment_position_m - anchor, axis=1)) > (
            1.0e-3
        )
        assert abs(sprung.twist_angle_rad[-1]) < abs(pinned.twist_angle_rad[-1])
        assert sprung.audit.boundary_stored_energy_j >= 0.0
        assert math.isnan(free.audit.supported_momentum_residual_n_m_s)
        assert math.isfinite(pinned.audit.supported_momentum_residual_n_m_s)

        with pytest.raises(
            ValueError, match="torsional parameters require TORSIONAL_GRIP"
        ):
            _facade_config(
                interval,
                impact,
                boundary=kinds.FREE,
                torsional_stiffness_n_m_per_rad=2_000.0,
            )
        with pytest.raises(TypeError, match="boundary must be a BoundaryKind"):
            _facade_config(interval, impact, boundary="pinned")


def test_completed_terminal_state_feeds_canonical_flight_once() -> None:
    """Only a separated contact may become a launch; unfinished contact refuses."""
    with _fresh_provider_import("swing_sim"):
        interval = _interval_module()
        impact = importlib.import_module("shared.python.swing_sim.impact")
        flight = importlib.import_module("shared.python.swing_sim.flight")
        _assert_from_tools(Path(flight.__file__))
        initial = _facade_initial(interval, impact)
        club = _facade_club(interval)

        completed = interval.solve_impact_interval(
            initial, club, _facade_config(interval, impact)
        )
        post = completed.to_post_impact_state()
        assert isinstance(post, impact.PostImpactState)
        np.testing.assert_array_equal(
            post.ball_velocity, completed.ball_velocity_mps[-1]
        )
        np.testing.assert_array_equal(
            post.ball_angular_velocity, completed.ball_angular_velocity_rad_s[-1]
        )
        assert post.contact_duration == completed.contact_duration_s

        # The same handoff the canonical rate_of_closure pipeline performs.
        launch = flight.derive_launch_conditions(
            flight.to_flight_frame(post.ball_velocity),
            flight.to_flight_frame(post.ball_angular_velocity),
        )
        assert launch.ball_speed == pytest.approx(
            float(np.linalg.norm(post.ball_velocity))
        )
        assert 0.0 < launch.launch_angle < math.radians(45.0)
        trajectory = flight.simulate(launch, model_name="waterloo_penner")
        assert trajectory.carry_distance > 50.0
        assert trajectory.flight_time > 1.0

        truncated = interval.solve_impact_interval(
            initial, club, _facade_config(interval, impact, maximum_time_s=1.0e-5)
        )
        assert truncated.termination is interval.ImpactTermination.TIME_LIMIT
        assert truncated.compression_m[-1] > 0.0
        with pytest.raises(interval.IncompleteContactError) as excinfo:
            truncated.to_post_impact_state()
        assert excinfo.value.result is truncated

        miss = interval.solve_impact_interval(
            dataclasses.replace(initial, club_velocity_mps=np.zeros(3)),
            club,
            _facade_config(interval, impact, maximum_time_s=1.0e-5),
        )
        assert miss.did_contact is False
        assert miss.termination is interval.ImpactTermination.NO_CONTACT
        with pytest.raises(interval.IncompleteContactError):
            miss.to_post_impact_state()


def test_provider_honours_interval_tab_ruling() -> None:
    """Owner ruling guard: interval tab dropped, ImpactModelType unchanged.

    Tools #4946 closed with an owner-ratified ruling (Tools PR #5289,
    `docs/development/IMPACT_INTERVAL_TAB_RULING.md` in the Tools repo) and UD #9549:
    the standalone contact-interval tab is DROPPED, ImpactModelType shall NOT gain an
    INTERVAL member, and swing_sim/impact_interval/ stays a headless engine. The
    interval solver is consumed headless via the facade tested earlier in the file.
    """
    with _fresh_provider_import("rate_of_closure"):
        impact = importlib.import_module("shared.python.swing_sim.impact")
        _assert_from_tools(Path(impact.__file__))
        model_members = {member.name for member in impact.ImpactModelType}
        assert model_members == {"RIGID_BODY", "SPRING_DAMPER", "FINITE_TIME"}
        assert not any("INTERVAL" in name for name in model_members), (
            "Tools #4946 / Tools PR #5289 (IMPACT_INTERVAL_TAB_RULING.md) and UD #9549 "
            "ruling: ImpactModelType shall NOT gain an INTERVAL member; interval "
            "solver remains headless."
        )
