"""OF-12 (#13070): family-wise screen that eliminates inconsistent candidates.

The spatial two-point-feet fixture is the candidate; its own contact loads,
seen through a vertical-cell plate (OF-11) with a 2 cm sensing depth, are the
"measurement".  Statistical claims are checked by seeded Monte Carlo against
the declared Gaussian noise; the elimination frames of a wrong model are
computed by hand from the loads, not read back from the screen.
"""

from __future__ import annotations

import dataclasses
from statistics import NormalDist

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import Decomposition, decompose
from src.shared.python.loop_nullspace.fixtures import spatial_two_arm_club_snapshot
from src.shared.python.loop_nullspace.ground_contact import (
    GRF_ID,
    FootLoadReading,
    GroundObservation,
)
from src.shared.python.loop_nullspace.plate_screen import (
    FamilyWiseRule,
    PlateConsistencyReport,
    ScreenFrame,
    ScreenSetup,
    offset_grid,
    plate_consistency,
)
from src.shared.python.loop_nullspace.registry import (
    Hypothesis,
    HypothesisSet,
    Source,
)
from tests.unit.loop_nullspace.test_ground_contact import _consistent

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

DEPTH_M = 0.02
N_FRAMES = 12
ALPHA = 0.05


def _hypotheses(*ids: str) -> HypothesisSet:
    return HypothesisSet(
        "t",
        "Plate screen",
        tuple(Hypothesis(i, "t", Source(None, "t"), "NOT_APPLICABLE") for i in ids),
        (),
    )


@pytest.fixture(scope="module")
def truth():
    pytest.importorskip("highspy")
    snap, _ = spatial_two_arm_club_snapshot(
        floating_root=True, feet=True, zero_root_pose=True, foot_points=1
    )
    snap = _consistent(snap)
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    return snap, dec


def _groups(snap: ClosedLoopSnapshot) -> dict[str, list[str]]:
    names = [c.name for c in snap.foot_contacts]
    return {"left": [names[0]], "right": [names[1]]}


def _loads(snap: ClosedLoopSnapshot, dec: Decomposition) -> np.ndarray:
    lead = dec.n_actuated + dec.m
    return np.asarray(dec.x_p[lead:]).reshape(-1, 3)


def _plate_reading(snap, dec, member: int, sigmas) -> tuple[float, np.ndarray]:
    """Hand oracle: per-foot load and CoP on the sensing plane (point foot)."""
    f = _loads(snap, dec)[member]
    p = snap.foot_contacts[member].position
    sensing_z = p[2] - DEPTH_M
    cop = np.array(
        [
            p[0] - (p[2] - sensing_z) * f[0] / f[2],
            p[1] - (p[2] - sensing_z) * f[1] / f[2],
            sensing_z,
        ]
    )
    return float(f[2]), cop


@dataclasses.dataclass(frozen=True)
class Sigmas:
    force_n: float
    cop_m: float


def _readings(snap, dec, sigmas: Sigmas, rng=None) -> dict[str, FootLoadReading]:
    out = {}
    for label, member in (("left", 0), ("right", 1)):
        load, cop = _plate_reading(snap, dec, member, sigmas)
        if rng is not None:
            load += rng.normal(0.0, sigmas.force_n)
            cop[:2] += rng.normal(0.0, sigmas.cop_m, size=2)
        out[label] = FootLoadReading(
            normal_load_n=load,
            cop_m=cop,
            sigma_force_n=sigmas.force_n,
            sigma_cop_m=sigmas.cop_m,
            sigma_sensing_m=0.0,
        )
    return out


def _sigmas(truth) -> Sigmas:
    snap, dec = truth
    loads = _loads(snap, dec)[:, 2]
    return Sigmas(force_n=0.005 * float(loads.min()), cop_m=0.002)


def _static_provider(snap, readings_by_frame):
    """Provider factory: the same data for every offset shift."""

    def factory(_shift_s: float):
        def provider(index: int, snapshot: ClosedLoopSnapshot):
            readings = readings_by_frame.get(index)
            if readings is None:
                return None
            return GroundObservation.from_vertical_plates(
                snapshot, readings, _groups(snapshot)
            )

        return provider

    return factory


def _frames(snap, dec, n=N_FRAMES) -> tuple[ScreenFrame, ...]:
    return tuple(ScreenFrame(i, snap, dec) for i in range(n))


def _setup(offsets=(0.0,)) -> ScreenSetup:
    return ScreenSetup(
        rule=FamilyWiseRule(alpha=ALPHA),
        hypotheses=_hypotheses(GRF_ID),
        offsets_s=offsets,
    )


# ------------------------------------------------------------------- rule --
def test_family_wise_rule_is_the_bonferroni_normal_quantile() -> None:
    rule = FamilyWiseRule(alpha=0.05)
    assert rule.coverage(1) == pytest.approx(NormalDist().inv_cdf(1 - 0.025))
    assert rule.coverage(100) == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 200))
    assert rule.coverage(100) > rule.coverage(10) > rule.coverage(1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
def test_family_wise_rule_rejects_a_bad_level(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        FamilyWiseRule(alpha=alpha)


def test_family_wise_rule_needs_a_positive_row_count() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        FamilyWiseRule(alpha=0.05).coverage(0)


# ------------------------------------------------------------- statistics --
def test_true_candidate_survives_noise_at_the_declared_level(truth) -> None:
    snap, dec = truth
    sigmas = _sigmas(truth)
    rng = np.random.default_rng(13070)
    runs, eliminated = 200, 0
    for _ in range(runs):
        data = {i: _readings(snap, dec, sigmas, rng) for i in range(N_FRAMES)}
        report = plate_consistency(
            _frames(snap, dec), _static_provider(snap, data), _setup()
        )
        eliminated += report.eliminated
    # Bonferroni bounds the family-wise rate by alpha; allow a 3-sigma binomial margin.
    margin = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / runs)
    assert eliminated / runs <= ALPHA + margin


def test_noise_free_truth_is_consistent_in_both_verdicts(truth) -> None:
    snap, dec = truth
    data = {i: _readings(snap, dec, _sigmas(truth)) for i in range(N_FRAMES)}
    report = plate_consistency(
        _frames(snap, dec), _static_provider(snap, data), _setup()
    )
    assert isinstance(report, PlateConsistencyReport)
    assert not report.eliminated
    (only,) = report.offsets
    assert only.n_rows == 6 * N_FRAMES
    assert only.coverage == pytest.approx(FamilyWiseRule(ALPHA).coverage(6 * N_FRAMES))
    assert all(f.observation_consistent and f.joint_feasible for f in only.frames)
    assert report.surviving_shifts_s == (0.0,)


# ---------------------------------------------------------- falsification --
def _light_model(snap: ClosedLoopSnapshot, scale: float) -> tuple:
    light = _consistent(dataclasses.replace(snap, M=scale * snap.M, h=scale * snap.h))
    return light, decompose(light, weights=np.eye(light.topology.n_actuated))


def test_a_five_percent_light_model_is_eliminated_where_the_loads_disagree(
    truth,
) -> None:
    snap, dec = truth
    sigmas = _sigmas(truth)
    data = {i: _readings(snap, dec, sigmas) for i in range(N_FRAMES)}
    light, light_dec = _light_model(snap, 0.95)
    k = FamilyWiseRule(ALPHA).coverage(6 * N_FRAMES)
    # Hand oracle: a point foot's vertical load is fixed by the balance, so the
    # light model's load row misses the data by |N_meas - N_model|.
    delta = np.abs(_loads(snap, dec)[:, 2] - _loads(light, light_dec)[:, 2])
    expect_eliminated = bool(np.any(delta > k * sigmas.force_n))
    assert expect_eliminated, "fixture must make the 5 % error visible"
    report = plate_consistency(
        _frames(light, light_dec), _static_provider(light, data), _setup()
    )
    assert report.eliminated
    (only,) = report.offsets
    assert only.first_eliminating_index == 0
    assert not any(f.joint_feasible for f in only.frames)
    assert GRF_ID in only.frames[0].conflict


# ------------------------------------------------------- offset profiling --
def _offset_provider(snap, dec, sigmas, true_shift: float, wrong):
    """Data are right only when the profiled shift matches ``true_shift``."""
    right = _readings(snap, dec, sigmas)

    def factory(shift_s: float):
        readings = right if abs(shift_s - true_shift) < 1e-12 else wrong

        def provider(index: int, snapshot: ClosedLoopSnapshot):
            return GroundObservation.from_vertical_plates(
                snapshot, readings, _groups(snapshot)
            )

        return provider

    return factory


def test_offset_profiling_survives_iff_the_grid_reaches_the_true_offset(truth) -> None:
    snap, dec = truth
    sigmas = _sigmas(truth)
    light, light_dec = _light_model(snap, 0.95)
    wrong = _readings(light, light_dec, sigmas)
    factory = _offset_provider(snap, dec, sigmas, 0.004, wrong)
    reached = plate_consistency(
        _frames(snap, dec), factory, _setup(offsets=(-0.004, 0.0, 0.004))
    )
    assert not reached.eliminated
    assert reached.surviving_shifts_s == (0.004,)
    missed = plate_consistency(_frames(snap, dec), factory, _setup(offsets=(0.0,)))
    assert missed.eliminated


def test_offset_grid_is_symmetric_includes_zero_and_respects_the_step() -> None:
    grid = offset_grid(sigma_offset_s=0.010, reach_sigmas=2.0, max_step_s=0.005)
    assert grid[0] == pytest.approx(-0.020) and grid[-1] == pytest.approx(0.020)
    assert 0.0 in grid
    assert np.all(np.diff(grid) <= 0.005 + 1e-12)
    assert offset_grid(sigma_offset_s=0.0, reach_sigmas=3.0, max_step_s=0.005) == (0.0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_offset_s": -1.0, "reach_sigmas": 2.0, "max_step_s": 0.005},
        {"sigma_offset_s": 0.01, "reach_sigmas": 0.0, "max_step_s": 0.005},
        {"sigma_offset_s": 0.01, "reach_sigmas": 2.0, "max_step_s": 0.0},
    ],
)
def test_offset_grid_validates_its_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        offset_grid(**kwargs)


# ---------------------------------------------------------- missing data --
def test_frames_without_data_are_untested_never_fabricated(truth) -> None:
    snap, dec = truth
    data = {0: _readings(snap, dec, _sigmas(truth))}
    report = plate_consistency(
        _frames(snap, dec), _static_provider(snap, data), _setup()
    )
    (only,) = report.offsets
    assert only.n_rows == 6
    tested = [f for f in only.frames if f.tested]
    assert [f.index for f in tested] == [0]
    assert not report.eliminated


def test_a_trajectory_without_any_data_is_not_a_verdict(truth) -> None:
    snap, dec = truth
    report = plate_consistency(_frames(snap, dec), _static_provider(snap, {}), _setup())
    assert not report.eliminated
    assert report.offsets[0].n_rows == 0
    assert not report.tested


# ---------------------------------------------------------- preconditions --
def test_setup_requires_the_grf_hypothesis() -> None:
    with pytest.raises(ValueError, match="H-GRF-OBSERVATION"):
        ScreenSetup(
            rule=FamilyWiseRule(ALPHA),
            hypotheses=_hypotheses("H-FRICTION-CONE"),
            offsets_s=(0.0,),
        )


@pytest.mark.parametrize("offsets", [(), (np.nan,), (0.0, 0.0)])
def test_setup_requires_distinct_finite_offsets(offsets) -> None:
    with pytest.raises(ValueError, match="offsets_s"):
        _setup(offsets=offsets)


def test_frames_must_be_unique_and_typed(truth) -> None:
    snap, dec = truth
    with pytest.raises(ValueError, match="unique"):
        plate_consistency(
            (ScreenFrame(0, snap, dec), ScreenFrame(0, snap, dec)),
            _static_provider(snap, {}),
            _setup(),
        )
    with pytest.raises(TypeError, match="decomposition"):
        ScreenFrame(0, snap, "dec")  # type: ignore[arg-type]


def test_report_serialises_to_plain_json(truth) -> None:
    import json

    snap, dec = truth
    data = {i: _readings(snap, dec, _sigmas(truth)) for i in range(2)}
    report = plate_consistency(
        _frames(snap, dec, 2), _static_provider(snap, data), _setup()
    )
    payload = report.to_dict()
    json.dumps(payload)
    assert payload["deciding_verdict"] == "joint"
    assert payload["alpha"] == ALPHA
