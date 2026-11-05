"""``ShotOutcomeModel`` protocol, backend registry and baseline backend (#14565)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.transition import sample_transitions
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.outcome import (
    LandingSamples,
    ShotOutcomeModel,
)
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
)
from src.shared.python.sg_optimizer.shot_model.registry import (
    BASELINE_BACKEND,
    available_backends,
    get_shot_model,
    quick_sg,
    register_shot_model,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
PGA_TOUR_YAML = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"
_FAIRWAY = LIE_CODES["fairway"]
_ROUGH = LIE_CODES["rough"]
_SAND = LIE_CODES["sand"]


def _player() -> PlayerProfile:
    return PlayerProfile(
        name="p",
        baseline=str(PGA_TOUR_YAML),
        clubs={"7_iron": ClubSkill(skill_mult_lat=1.4, distance_offset=-5.0)},
    )


def test_landing_samples_validates_shapes() -> None:
    LandingSamples(along=np.zeros(3), lateral=np.zeros(3))
    with pytest.raises(ContractViolationError):
        LandingSamples(along=np.zeros(3), lateral=np.zeros(2))
    with pytest.raises(ContractViolationError):
        LandingSamples(along=np.array([np.nan]), lateral=np.zeros(1))


def test_baseline_backend_is_registered_and_satisfies_protocol() -> None:
    assert BASELINE_BACKEND in available_backends()
    model = get_shot_model(BASELINE_BACKEND)
    assert isinstance(model, ShotOutcomeModel)


def test_unknown_backend_fails_closed() -> None:
    with pytest.raises(ContractViolationError, match="unknown shot-model backend"):
        get_shot_model("no_such_backend")


def test_duplicate_registration_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        register_shot_model(
            BASELINE_BACKEND, lambda baseline: get_shot_model("baseline")
        )


def test_quick_sg_always_returns_baseline_backend() -> None:
    model = quick_sg()
    bag = load_baseline(PGA_TOUR_YAML)
    out = model.sample(
        _player(),
        ShotAction(club="7_iron", aim_angle_rad=0.0),
        _FAIRWAY,
        CourseConditions.benign(),
        np.random.default_rng(1),
        500,
    )
    assert out.along.shape == (500,)
    expected = bag.get("7_iron").total_mean - 5.0
    assert abs(out.along.mean() - expected) < 3.0


def test_baseline_samples_match_player_effective_distribution() -> None:
    bag = load_baseline(PGA_TOUR_YAML)
    player = _player()
    model = get_shot_model(BASELINE_BACKEND, baseline=bag)
    shot = ShotAction(club="7_iron", aim_angle_rad=0.0)
    out = model.sample(
        player,
        shot,
        _FAIRWAY,
        CourseConditions.benign(),
        np.random.default_rng(7),
        4000,
    )
    dist = player.effective_distribution("7_iron", bag)
    assert out.lateral.std() == pytest.approx(dist.sigma_lat, rel=0.08)
    assert out.along.std() == pytest.approx(dist.sigma_long, rel=0.08)


class _ConstModel:
    """Test double: every shot lands at a fixed (along, lateral)."""

    def __init__(self, along: float, lateral: float) -> None:
        self.xy = (along, lateral)

    def sample(self, player, shot, lie, conditions, rng, n):  # noqa: ANN001
        return LandingSamples(np.full(n, self.xy[0]), np.full(n, self.xy[1]))


def _raster(size: int, origin: float) -> LieRaster:
    return LieRaster(
        codes=np.full((size, size), _FAIRWAY, dtype=np.int8),
        origin=(origin, origin),
        resolution_yd=5.0,
        pin=(0.0, 0.0),
        tee=(0.0, 0.0),
    )


def test_transition_uses_injected_backend() -> None:
    raster = _raster(80, -100.0)
    bag = load_baseline(PGA_TOUR_YAML)
    outcomes = sample_transitions(
        State(x=0.0, y=0.0, lie=_FAIRWAY),
        ShotAction("7_iron", 0.0),
        _player(),
        bag,
        CourseConditions.benign(),
        raster,
        4,
        np.random.default_rng(0),
        model=_ConstModel(100.0, 5.0),
    )
    assert [(o.next_state.x, o.next_state.y) for o in outcomes] == [(100.0, 5.0)] * 4


def test_transition_default_matches_legacy_formula() -> None:
    """Behaviour-preserving: same seed gives the pre-refactor landing points."""
    bag = load_baseline(PGA_TOUR_YAML)
    player, cond = _player(), CourseConditions.tournament()
    raster = _raster(80, -200.0)
    ang = 0.3
    got = sample_transitions(
        State(x=0.0, y=0.0, lie=_ROUGH),
        ShotAction("7_iron", ang),
        player,
        bag,
        cond,
        raster,
        20,
        np.random.default_rng(11),
    )
    dist_mult = cond.rough.distance_multiplier()
    disp = cond.rough.dispersion_multiplier()
    distr = player.effective_distribution("7_iron", bag).scaled(disp, disp)
    off = distr.sample(20, np.random.default_rng(11))
    along = player.effective_distance("7_iron", bag) * dist_mult + off[:, 0]
    lat = off[:, 1]
    ex = along * np.cos(ang) - lat * np.sin(ang)
    ey = along * np.sin(ang) + lat * np.cos(ang)
    np.testing.assert_allclose([o.next_state.x for o in got], ex)
    np.testing.assert_allclose([o.next_state.y for o in got], ey)


@pytest.fixture
def restore_registry():
    """Snapshot/restore the backend registry so tests can register freely."""
    from src.shared.python.sg_optimizer.shot_model import registry

    saved = dict(registry._BACKENDS)
    yield
    registry._BACKENDS.clear()
    registry._BACKENDS.update(saved)


def test_factory_returning_non_model_fails_postcondition(restore_registry) -> None:
    register_shot_model("broken", lambda baseline: object())
    with pytest.raises(ContractViolationError, match="not a model"):
        get_shot_model("broken")


def test_hole_mdp_honours_injected_backend(restore_registry) -> None:
    """A registered backend must change the solve, not be silently ignored."""
    from src.shared.python.sg_optimizer.mdp.action import ActionSet
    from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP

    grid = np.full((12, 12), _FAIRWAY, dtype=np.int8)
    grid[6, 6] = LIE_CODES["holed"]
    raster = LieRaster(
        codes=grid, origin=(-30.0, -30.0), resolution_yd=5.0, pin=(0, 0), tee=(0, 0)
    )
    args = (
        raster,
        _player(),
        load_baseline(PGA_TOUR_YAML),
        CourseConditions.benign(),
        ActionSet(clubs=("7_iron",), aim_grid_deg=np.array([0.0])),
    )
    base = HoleMDP(*args, n_samples=8, seed=1).bellman_backup(np.zeros((12, 12)))[0]
    fixed = HoleMDP(
        *args, n_samples=8, seed=1, model=_ConstModel(1.0, 0.0)
    ).bellman_backup(np.zeros((12, 12)))[0]
    assert not np.allclose(base, fixed)
