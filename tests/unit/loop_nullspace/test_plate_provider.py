"""OF-12 (#13070): plate trial -> per-frame ground observation -> screen.

A static plate trial is built from the spatial fixture's own contact loads,
seen through a vertical-cell plate with a 2 cm force-sensing depth, so the
true candidate must survive and a 5 %-light model must be eliminated through
the full trial -> reading -> observation -> feasible-set path.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.force_plates import (
    PlacedTrial,
    PlatePlacement,
    PlateProvenance,
    PlateTimeAlignment,
    PlateUncertainty,
    ReadingOptions,
    VerticalPlateTrial,
    loaded_net_cop,
)
from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.loop_nullspace.fixtures import spatial_two_arm_club_snapshot
from src.shared.python.loop_nullspace.ground_contact import GRF_ID, GroundObservation
from src.shared.python.loop_nullspace.plate_provider import (
    PlateObservationProvider,
    screen_with_plates,
)
from src.shared.python.loop_nullspace.plate_screen import (
    FamilyWiseRule,
    ScreenFrame,
    ScreenSetup,
)
from tests.unit.loop_nullspace._ground_fixtures import (
    hypothesis_set,
    make_consistent,
    scaled_mass_model,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

DEPTH_M = 0.02
TIMES_S = np.arange(11) * 0.005
GROUND_GROUPS = {"left": "foot_0", "right": "foot_1"}
MODEL = "spatial-fixture-v1"


@pytest.fixture(scope="module")
def truth():
    pytest.importorskip("highspy")
    snap, _ = spatial_two_arm_club_snapshot(
        floating_root=True, feet=True, zero_root_pose=True, foot_points=1
    )
    snap = make_consistent(snap)
    return snap, decompose(snap, weights=np.eye(snap.topology.n_actuated))


def _foot_truth(snap, dec, member: int):
    """Hand oracle: load and stance CoP of a point foot over a sensing plane."""
    lead = dec.n_actuated + dec.m
    f = np.asarray(dec.x_p[lead:]).reshape(-1, 3)[member]
    p = snap.foot_contacts[member].position
    return float(f[2]), np.array(
        [p[0] - DEPTH_M * f[0] / f[2], p[1] - DEPTH_M * f[1] / f[2]]
    )


def _uncertainty(load: float) -> PlateUncertainty:
    return PlateUncertainty(
        force_floor_n=0.005 * load,
        force_fraction=0.0,
        cop_m=0.002,
        sensing_depth_sigma_m=0.0,
        source="declared hypothesis for tests",
    )


def _placed(snap, dec) -> PlacedTrial:
    feet = {"left": _foot_truth(snap, dec, 0), "right": _foot_truth(snap, dec, 1)}
    n = TIMES_S.size
    loads = {k: np.full(n, v[0]) for k, v in feet.items()}
    cops = {k: np.tile(v[1], (n, 1)) for k, v in feet.items()}
    cops["net"] = loaded_net_cop(loads, cops)
    surface = snap.foot_contacts[0].position[2]
    trial = VerticalPlateTrial(
        times_s=TIMES_S,
        load_n=loads,
        cop_m=cops,
        body_weight_n=float(sum(v[0] for v in feet.values())),
        handedness="RH",
        events_s={},
        cop_min_load_n=1e-6,
        capacity_n=1e9,
        sensing_depth_m=DEPTH_M,
        provenance=PlateProvenance(
            device="fixture",
            source_format="synthetic",
            source_revision="none",
            file_sha256={},
        ),
        uncertainty=_uncertainty(min(v[0] for v in feet.values())),
    )
    placement = PlatePlacement(rotation=np.eye(3), origin_m=np.array([0, 0, surface]))
    return PlacedTrial(trial=trial, placement=placement)


def _alignment(**changes) -> PlateTimeAlignment:
    base = {
        "offset_s": 0.0,
        "sigma_offset_s": 0.0,
        "kind": "manual",
        "note": "fixture clocks agree",
    }
    return PlateTimeAlignment(**{**base, **changes})


def _provider(snap, dec, **changes) -> PlateObservationProvider:
    kwargs = {
        "placed": _placed(snap, dec),
        "alignment": _alignment(),
        "frame_times_s": {i: 0.01 + 0.002 * i for i in range(5)},
        "options": ReadingOptions(coverage=3.0),
        "ground_groups": GROUND_GROUPS,
    }
    return PlateObservationProvider(**{**kwargs, **changes})


def _setup() -> ScreenSetup:
    return ScreenSetup(
        rule=FamilyWiseRule(alpha=0.05),
        hypotheses=hypothesis_set(GRF_ID),
        offsets_s=(0.0,),
    )


# --------------------------------------------------------------- provider --
def test_provider_builds_the_vertical_plate_observation(truth) -> None:
    snap, dec = truth
    obs = _provider(snap, dec)(0, snap)
    assert isinstance(obs, GroundObservation)
    assert obs.labels == (
        "left:normal_load",
        "left:cop_moment_t1",
        "left:cop_moment_t2",
        "right:normal_load",
        "right:cop_moment_t1",
        "right:cop_moment_t2",
    )
    lead = dec.n_actuated + dec.m
    residual = obs.selector @ dec.x_p[lead:] - obs.values
    assert np.max(np.abs(residual / obs.sigma)) < 1e-6


def test_provider_returns_none_without_data(truth) -> None:
    snap, dec = truth
    provider = _provider(snap, dec, frame_times_s={0: 0.01, 1: 10.0})
    assert provider(7, snap) is None  # no time for the frame
    assert provider(1, snap) is None  # outside the trial


def test_with_shift_moves_the_trial_time(truth) -> None:
    snap, dec = truth
    provider = _provider(snap, dec, frame_times_s={0: 0.049})
    assert provider(0, snap) is not None
    assert provider.with_shift(-0.002, 3.0)(0, snap) is None  # 0.051 s is past the end


def test_with_shift_reads_at_the_requested_coverage(truth) -> None:
    snap, dec = truth
    obs = _provider(snap, dec).with_shift(0.0, 4.5)(0, snap)
    assert obs.coverage == 4.5


def test_profiling_half_step_widens_the_skew(truth) -> None:
    snap, dec = truth
    provider = _provider(snap, dec)
    widened = provider.with_shift(0.0, 3.0, half_step_s=0.0025)
    assert widened.options.skew_s == pytest.approx(provider.options.skew_s + 0.0025)


def test_frame_keys_must_be_integers(truth) -> None:
    snap, dec = truth
    with pytest.raises(TypeError):
        _provider(snap, dec, frame_times_s={1.7: 0.01})


def test_ground_groups_must_name_both_feet(truth) -> None:
    snap, dec = truth
    with pytest.raises(ValueError, match="ground_groups"):
        _provider(snap, dec, ground_groups={"left": "foot_0"})


def test_frame_times_must_be_finite(truth) -> None:
    snap, dec = truth
    with pytest.raises(ValueError, match="frame_times_s"):
        _provider(snap, dec, frame_times_s={0: float("nan")})


def test_profile_offsets_span_the_declared_offset_sigma(truth) -> None:
    snap, dec = truth
    provider = _provider(snap, dec, alignment=_alignment(sigma_offset_s=0.004))
    shifts = provider.profile_offsets(reach_sigmas=2.0)
    assert shifts[0] == pytest.approx(-0.008) and shifts[-1] == pytest.approx(0.008)
    assert np.all(np.diff(shifts) <= 0.005 + 1e-12)


# ---------------------------------------------------------------- screen --
def _frames(snap, dec, n=5):
    return tuple(ScreenFrame(i, snap, dec) for i in range(n))


def test_true_candidate_survives_the_full_plate_path(truth) -> None:
    snap, dec = truth
    report = screen_with_plates(
        _frames(snap, dec), _provider(snap, dec), _setup(), model_identity=MODEL
    )
    assert report.tested and not report.eliminated


def test_light_model_is_eliminated_through_the_full_plate_path(truth) -> None:
    snap, dec = truth
    light, light_dec = scaled_mass_model(snap, 0.95)
    report = screen_with_plates(
        _frames(light, light_dec), _provider(snap, dec), _setup(), model_identity=MODEL
    )
    assert report.eliminated


def test_alignment_derived_from_the_model_under_test_is_refused(truth) -> None:
    snap, dec = truth
    derived = _alignment(kind="cross_correlation", derived_from_model=MODEL)
    with pytest.raises(ValueError, match="derived from"):
        screen_with_plates(
            _frames(snap, dec),
            _provider(snap, dec, alignment=derived),
            _setup(),
            model_identity=MODEL,
        )
    other = dataclasses.replace(derived, derived_from_model="another-model")
    report = screen_with_plates(
        _frames(snap, dec),
        _provider(snap, dec, alignment=other),
        _setup(),
        model_identity=MODEL,
    )
    assert not report.eliminated


# ------------------------------------------------------------ facade seam --
def _wide_capacity(n: int):
    from src.shared.python.loop_nullspace.capacity import CapacityVector

    names = tuple(f"j{i}" for i in range(n))
    return CapacityVector(
        coordinate_order=names,
        actuated_indices=tuple(range(n)),
        actuated_names=names,
        tau_max_negative_nm=np.full(n, 1e6),
        tau_max_positive_nm=np.full(n, 1e6),
        assumed_slots=(),
        verified_slots=names,
        warnings=(),
        force_velocity={},
    )


def test_provider_is_the_facade_observation_feed(truth) -> None:
    from src.shared.python.loop_nullspace.contracts import ModelProvenance
    from src.shared.python.loop_nullspace.facade import EngineContext

    snap, dec = truth
    provider = _provider(snap, dec)
    engine = EngineContext(
        name="spatial-fixture",
        model_provenance=ModelProvenance("spatial_two_arm_club", "1", "abc"),
        observation_for=provider,
        ground_groups=GROUND_GROUPS,
    )
    assert isinstance(engine.observation_for(0, snap), GroundObservation)


def test_shipped_grf_preset_with_cones_and_capacity_keeps_the_truth(truth) -> None:
    from src.shared.python.loop_nullspace import registry

    snap, dec = truth
    setup = ScreenSetup(
        rule=FamilyWiseRule(alpha=0.05),
        hypotheses=registry.load_registry().preset("envelope_grf_observed"),
        offsets_s=(0.0,),
        capacity=_wide_capacity(dec.n_actuated),
    )
    report = screen_with_plates(
        _frames(snap, dec), _provider(snap, dec), setup, model_identity=MODEL
    )
    assert report.tested and not report.eliminated
    light, light_dec = scaled_mass_model(snap, 0.95)
    light_report = screen_with_plates(
        _frames(light, light_dec), _provider(snap, dec), setup, model_identity=MODEL
    )
    assert light_report.eliminated
