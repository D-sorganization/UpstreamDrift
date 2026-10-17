"""OF-11 (#13067): per-foot vertical-cell plates as ground-observation rows.

A vertical-cell plate under each foot measures that foot's normal load and the
moment of the foot's loads about the plate's force-sensing plane, a depth ``h``
below the contacts.  The rows are checked against the exact NSX-00 golden
``lower_body_observation`` (selectors ``vertical_per_foot`` with ``h = 0`` and
``vertical_per_foot_sensing_depth`` with ``h = 1/50``; sympy, no repository
code).  The apparent CoP of the true loads is computed here by hand from the
plate's moment balance.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.loop_nullspace.fixtures import (
    load_golden,
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.ground_contact import (
    FootLoadReading,
    GroundObservation,
    ObservationInconsistentError,
    observation_accounting,
    restrict_to_observation,
)
from tests.unit.loop_nullspace._ground_fixtures import VARIANTS, make_consistent

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

DEPTHS = {"vertical_per_foot": 0.0, "vertical_per_foot_sensing_depth": 0.02}
SIGMA_N = 5.0
SIGMA_COP = 0.005


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def plate_case(request):
    snap, _ = spatial_two_arm_club_snapshot(
        floating_root=True, feet=True, zero_root_pose=True, **VARIANTS[request.param]
    )
    snap = make_consistent(snap)
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    golden = load_golden("lower_body_observation", request.param).payload
    per_foot = VARIANTS[request.param]["foot_points"]
    names = [c.name for c in snap.foot_contacts]
    groups = {
        "left": names[:per_foot],
        "right": names[per_foot:],
    }
    return snap, dec, golden, groups


def _exact(values) -> np.ndarray:
    return np.array([float(Fraction(v)) for v in values], dtype=float)


def _golden_readings(golden, name, groups, loads=(400.0, 400.0)):
    entry = golden["observations"][name]
    return {
        label: FootLoadReading(
            normal_load_n=load,
            cop_m=_exact(cop),
            sigma_force_n=SIGMA_N,
            sigma_cop_m=SIGMA_COP,
            sigma_sensing_m=0.0,
        )
        for label, cop, load in zip(groups, entry["cop_per_foot"], loads, strict=True)
    }


def _true_loads(snap, dec) -> np.ndarray:
    lead = dec.n_actuated + dec.m
    return np.asarray(dec.x_p[lead:]).reshape(-1, 3)


def _apparent_cop(snap, dec, members, sensing_z):
    """Hand oracle: CoP on the sensing plane from the plate moment balance.

    With ``h_k = p_k.z - sensing_z`` the cells balance
    ``sum_k [(p_k - c)_x f_z - h_k f_x] = 0`` and the y analogue.
    """
    names = [c.name for c in snap.foot_contacts]
    f = _true_loads(snap, dec)
    load = sum(f[names.index(n), 2] for n in members)
    out = np.zeros(3)
    for axis in (0, 1):
        moment = 0.0
        for n in members:
            k = names.index(n)
            p = snap.foot_contacts[k].position
            moment += p[axis] * f[k, 2] - (p[2] - sensing_z) * f[k, axis]
        out[axis] = moment / load
    out[2] = sensing_z
    return float(load), out


def _true_readings(snap, dec, groups, depth, *, shift=(0.0, 0.0)):
    readings = {}
    for label, members in groups.items():
        sensing_z = snap.foot_contacts[0].position[2] - depth
        load, cop = _apparent_cop(snap, dec, members, sensing_z)
        cop[:2] += shift
        readings[label] = FootLoadReading(
            normal_load_n=load,
            cop_m=cop,
            sigma_force_n=SIGMA_N,
            sigma_cop_m=SIGMA_COP,
            sigma_sensing_m=0.0,
        )
    return readings


# ------------------------------------------------------------ golden rows --
@pytest.mark.parametrize("name", sorted(DEPTHS))
def test_selector_matches_the_exact_golden(plate_case, name) -> None:
    snap, _, golden, groups = plate_case
    obs = GroundObservation.from_vertical_plates(
        snap, _golden_readings(golden, name, groups), groups
    )
    expected = np.array(
        [_exact(row) for row in golden["observations"][name]["selector"]]
    )
    np.testing.assert_allclose(obs.selector, expected, atol=1e-12)
    assert obs.labels[:3] == (
        "left:normal_load",
        "left:cop_moment_t1",
        "left:cop_moment_t2",
    )


@pytest.mark.parametrize("name", sorted(DEPTHS))
def test_reduction_equals_the_golden_rank(plate_case, name) -> None:
    snap, dec, golden, groups = plate_case
    entry = golden["observations"][name]
    readings = _true_readings(snap, dec, groups, DEPTHS[name])
    acc = observation_accounting(
        dec, GroundObservation.from_vertical_plates(snap, readings, groups)
    )
    assert acc.rank_visible == entry["reduction"]
    assert acc.rank_selector == entry["rank_selector"]


@pytest.mark.parametrize("name", sorted(DEPTHS))
def test_true_loads_are_consistent_and_restrict(plate_case, name) -> None:
    snap, dec, golden, groups = plate_case
    entry = golden["observations"][name]
    obs = GroundObservation.from_vertical_plates(
        snap, _true_readings(snap, dec, groups, DEPTHS[name]), groups
    )
    acc = observation_accounting(dec, obs)
    assert acc.max_normalised_residual < 1e-9
    restricted = restrict_to_observation(dec, obs)
    assert restricted.nullity == entry["nullity_observed"]


@pytest.mark.parametrize("name", sorted(DEPTHS))
def test_shear_rank_after_observation_matches_the_golden(plate_case, name) -> None:
    snap, dec, golden, groups = plate_case
    entry = golden["observations"][name]
    obs = GroundObservation.from_vertical_plates(
        snap, _true_readings(snap, dec, groups, DEPTHS[name]), groups
    )
    restricted = restrict_to_observation(dec, obs)
    lead = dec.n_actuated + dec.m
    n_pts = len(snap.foot_contacts)
    shear = [lead + 3 * k + c for k in range(n_pts) for c in (0, 1)]
    rank = np.linalg.matrix_rank(restricted.N[shear], tol=1e-9)
    assert rank == entry["shear_rank_observed"]
    assert np.linalg.matrix_rank(dec.N[shear], tol=1e-9) == entry["shear_rank_free"]


# ----------------------------------------------------------- falsification --
def test_a_cop_outside_the_band_is_refused(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0, shift=(0.05, 0.0))
    obs = GroundObservation.from_vertical_plates(snap, readings, groups)
    with pytest.raises(ObservationInconsistentError):
        restrict_to_observation(dec, obs)


def test_load_on_a_foot_the_model_leaves_unloaded_is_refused(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    unloaded = {"left": groups["left"], "right": []}
    obs = GroundObservation.from_vertical_plates(snap, readings, unloaded)
    assert obs.n_rows == 4  # the empty foot keeps only its load row
    assert not obs.selector[3].any()
    with pytest.raises(ObservationInconsistentError):
        restrict_to_observation(dec, obs)


def test_foot_below_threshold_contributes_only_its_load_row(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    readings["right"] = dataclasses.replace(readings["right"], cop_m=None)
    obs = GroundObservation.from_vertical_plates(snap, readings, groups)
    assert obs.n_rows == 4
    assert obs.labels[3] == "right:normal_load"


def test_sensing_depth_uncertainty_widens_the_cop_rows(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.02)
    exact = GroundObservation.from_vertical_plates(snap, readings, groups)
    widened = {
        k: dataclasses.replace(r, sigma_sensing_m=0.004) for k, r in readings.items()
    }
    obs = GroundObservation.from_vertical_plates(snap, widened, groups)
    mu = snap.foot_contacts[0].mu
    load = readings["left"].normal_load_n
    assert obs.sigma[0] == exact.sigma[0] == SIGMA_N
    assert obs.sigma[1] == pytest.approx(load * np.hypot(SIGMA_COP, mu * 0.004))


# ----------------------------------------------------------- preconditions --
def test_readings_and_groups_must_name_the_same_feet(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    with pytest.raises(ValueError, match="groups"):
        GroundObservation.from_vertical_plates(snap, readings, {"left": groups["left"]})


def test_groups_must_name_known_contacts_once(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    with pytest.raises(ValueError, match="unknown"):
        GroundObservation.from_vertical_plates(
            snap, readings, {"left": ["nope"], "right": groups["right"]}
        )
    with pytest.raises(ValueError, match="more than one"):
        GroundObservation.from_vertical_plates(
            snap, readings, {"left": groups["left"], "right": groups["left"]}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"normal_load_n": float("inf")},
        {"normal_load_n": float("nan")},
        {"sigma_force_n": 0.0},
        {"sigma_cop_m": 0.0},
        {"sigma_sensing_m": -0.1},
        {"cop_m": np.zeros(2)},
    ],
)
def test_foot_load_reading_validates_its_fields(changes) -> None:
    base = {
        "normal_load_n": 400.0,
        "cop_m": np.zeros(3),
        "sigma_force_n": 5.0,
        "sigma_cop_m": 0.005,
        "sigma_sensing_m": 0.0,
    }
    with pytest.raises(ValueError):
        FootLoadReading(**{**base, **changes})


def test_a_slightly_negative_measured_load_is_kept_as_measured() -> None:
    # An unloaded plate reads small negative noise; the observation keeps it.
    reading = FootLoadReading(
        normal_load_n=-3.0,
        cop_m=None,
        sigma_force_n=2.0,
        sigma_cop_m=0.005,
        sigma_sensing_m=0.0,
    )
    assert reading.normal_load_n == -3.0


def test_a_cop_needs_a_load_above_zero() -> None:
    with pytest.raises(ValueError, match="cop_m"):
        FootLoadReading(
            normal_load_n=-3.0,
            cop_m=np.zeros(3),
            sigma_force_n=2.0,
            sigma_cop_m=0.005,
            sigma_sensing_m=0.0,
        )


def test_a_group_may_not_repeat_a_contact(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    left = list(groups["left"])
    with pytest.raises(ValueError, match="repeats"):
        GroundObservation.from_vertical_plates(
            snap, readings, {"left": left + left, "right": groups["right"]}
        )


def test_moment_sigma_does_not_collapse_for_a_load_near_its_noise(plate_case) -> None:
    snap, dec, _, groups = plate_case
    readings = _true_readings(snap, dec, groups, 0.0)
    tiny = dataclasses.replace(readings["left"], normal_load_n=1.0, sigma_force_n=5.0)
    obs = GroundObservation.from_vertical_plates(
        snap, {**readings, "left": tiny}, groups
    )
    # The multiplier is max(N, sigma_N) = 5 N, not the 1 N reading.
    assert obs.sigma[1] == pytest.approx(5.0 * SIGMA_COP)


def test_foot_load_reading_arrays_are_read_only() -> None:
    reading = FootLoadReading(
        normal_load_n=400.0,
        cop_m=np.zeros(3),
        sigma_force_n=5.0,
        sigma_cop_m=0.005,
        sigma_sensing_m=0.0,
    )
    with pytest.raises(ValueError):
        reading.cop_m[0] = 1.0
