"""Exercise the public contact-interval façade through the pinned Tools provider.

UpstreamDrift #9549 (epic #9546) integrates the six-DOF contact-interval
solver into live runs, playback and export. Tools owns that runtime
(Tools #4130 / #4946) and UpstreamDrift consumes an immutable reviewed pin,
so this module records two things from the consumer side:

* what the launcher, parity and flight paths already rely on at the current
  pin -- a time-resolved contact record, boundary selection that changes the
  solver configuration, and a completed terminal state that feeds the
  canonical flight solver exactly once while unfinished contact refuses; and
* the workbench record integration the provider has not shipped yet, as a
  strict expected failure. When a pin bump lands the interval phase in the
  canonical ``rate_of_closure`` run record, that xfail turns into a failure
  and the UpstreamDrift adoption (launcher/parity/session paths) must follow.

Audit revisions: UpstreamDrift ``5347cba0f4378cd72a6e8afea9fb27c8bfe5db75``;
pinned Tools ``62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`` (``vendor/ud-tools``).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

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
_SHAFT_LENGTH_M = 0.9
_CLUB_SPEED_MPS = 50.0
_LOFT_RAD = math.radians(10.0)


def _interval_module() -> ModuleType:
    module = importlib.import_module("shared.python.swing_sim.impact_interval")
    _assert_from_tools(Path(module.__file__))
    return module


def _club(interval: ModuleType, toe_offset_m: float = 0.0):  # type: ignore[no-untyped-def]
    """Rigid driver head with a vertical shaft; +y is up in the app frame."""
    contact = np.array([0.0, 0.0, toe_offset_m])
    return interval.ClubRigidBody(
        mass_kg=_CLUB_MASS_KG,
        inertia_body_kg_m2=np.diag([4.5e-4, 4.5e-4, 4.5e-4]),
        cg_to_contact_body_m=contact,
        cg_to_attachment_body_m=np.array([0.0, _SHAFT_LENGTH_M, 0.0]),
        face_normal_body=np.array([math.cos(_LOFT_RAD), math.sin(_LOFT_RAD), 0.0]),
    )


def _initial(interval: ModuleType, impact: ModuleType, toe_offset_m: float = 0.0):  # type: ignore[no-untyped-def]
    """Ball resting on the lofted face at first touch, club moving downrange."""
    club = _club(interval, toe_offset_m)
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


def _config(interval: ModuleType, impact: ModuleType, **overrides: object):  # type: ignore[no-untyped-def]
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


def test_interval_facade_produces_time_resolved_contact_record() -> None:
    """A single façade call yields every inspection channel plus the audit."""
    with _fresh_provider_import("swing_sim"):
        interval = _interval_module()
        impact = importlib.import_module("shared.python.swing_sim.impact")
        config = _config(interval, impact)
        result = interval.solve_impact_interval(
            _initial(interval, impact), _club(interval), config
        )

        assert result.did_contact is True
        assert result.termination is interval.ImpactTermination.SEPARATED
        assert result.contact_completed is True
        sample_count = len(result.time_s)
        assert sample_count > 10
        assert result.time_s[0] == 0.0
        assert np.all(np.diff(result.time_s) > 0.0)
        # The loaded span ends at the last positive normal force; the history
        # runs on until compression is released, so it may outlast it slightly.
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
        club = _club(interval, toe_offset_m)
        free_initial = _initial(interval, impact, toe_offset_m)
        supported_initial = dataclasses.replace(
            free_initial,
            club_angular_velocity_rad_s=np.array(
                [0.0, 0.0, _CLUB_SPEED_MPS / _SHAFT_LENGTH_M]
            ),
        )
        free = interval.solve_impact_interval(
            free_initial, club, _config(interval, impact, boundary=kinds.FREE)
        )
        pinned = interval.solve_impact_interval(
            supported_initial, club, _config(interval, impact, boundary=kinds.PINNED)
        )
        sprung = interval.solve_impact_interval(
            supported_initial,
            club,
            _config(
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
            _config(
                interval,
                impact,
                boundary=kinds.FREE,
                torsional_stiffness_n_m_per_rad=2_000.0,
            )
        with pytest.raises(TypeError, match="boundary must be a BoundaryKind"):
            _config(interval, impact, boundary="pinned")


def test_completed_terminal_state_feeds_canonical_flight_once() -> None:
    """Only a separated contact may become a launch; unfinished contact refuses."""
    with _fresh_provider_import("swing_sim"):
        interval = _interval_module()
        impact = importlib.import_module("shared.python.swing_sim.impact")
        flight = importlib.import_module("shared.python.swing_sim.flight")
        _assert_from_tools(Path(flight.__file__))
        initial = _initial(interval, impact)
        club = _club(interval)

        completed = interval.solve_impact_interval(
            initial, club, _config(interval, impact)
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
            initial, club, _config(interval, impact, maximum_time_s=1.0e-5)
        )
        assert truncated.termination is interval.ImpactTermination.TIME_LIMIT
        assert truncated.compression_m[-1] > 0.0
        with pytest.raises(interval.IncompleteContactError) as excinfo:
            truncated.to_post_impact_state()
        assert excinfo.value.result is truncated

        miss = interval.solve_impact_interval(
            dataclasses.replace(initial, club_velocity_mps=np.zeros(3)),
            club,
            _config(interval, impact, maximum_time_s=1.0e-5),
        )
        assert miss.did_contact is False
        assert miss.termination is interval.ImpactTermination.NO_CONTACT
        with pytest.raises(interval.IncompleteContactError):
            miss.to_post_impact_state()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Tools #4946 has not shipped the contact-interval phase in the canonical "
        "rate_of_closure run record at provider pin "
        f"{PINNED_PROVIDER_REVISION[:9]}: SimulationRun carries no interval "
        "field and ImpactModelType has only RIGID_BODY, SPRING_DAMPER and "
        "FINITE_TIME. UpstreamDrift #9549 adopts the workbench integration "
        "through a reviewed pin bump, never through a local copy."
    ),
)
def test_pinned_provider_run_record_carries_a_contact_interval_phase() -> None:
    """Pin gate: fails (and must be adopted) once the provider ships the phase."""
    with _fresh_provider_import("rate_of_closure"):
        records = importlib.import_module("rate_of_closure.simulation.records")
        _assert_from_tools(Path(records.__file__))
        impact = importlib.import_module("shared.python.swing_sim.impact")
        _assert_from_tools(Path(impact.__file__))
        run_fields = {field.name for field in dataclasses.fields(records.SimulationRun)}
        model_members = {member.name for member in impact.ImpactModelType}
        assert model_members >= {"RIGID_BODY", "SPRING_DAMPER", "FINITE_TIME"}
        assert any("interval" in name for name in run_fields), sorted(run_fields)
        assert any("INTERVAL" in name for name in model_members), sorted(model_members)
