"""Name -> backend registry for ``ShotOutcomeModel`` (config as data, fail closed).

``quick_sg`` is the quick fallback: it never consults the registry, so it works
whatever other backends are (mis)registered.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.shot_model.baseline import (
    BaselineBag,
    load_baseline,
)
from src.shared.python.sg_optimizer.shot_model.baseline_model import BaselineShotModel
from src.shared.python.sg_optimizer.shot_model.outcome import ShotOutcomeModel

BASELINE_BACKEND = "baseline"
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "sg_optimizer"
    / "baselines"
    / "pga_tour.yaml"
)

# A factory takes the tour ``BaselineBag`` (the fallback reference every backend
# may lean on) and returns a ready ``ShotOutcomeModel``.
BackendFactory = Callable[[BaselineBag], ShotOutcomeModel]
_BACKENDS: dict[str, BackendFactory] = {BASELINE_BACKEND: BaselineShotModel}


@lru_cache(maxsize=1)
def _tour_baseline() -> BaselineBag:
    return load_baseline(DEFAULT_BASELINE_PATH)


def register_shot_model(name: str, factory: BackendFactory) -> None:
    """Register ``factory(baseline) -> ShotOutcomeModel`` under a new ``name``."""
    require(bool(name), "backend name must be non-empty")
    require(name not in _BACKENDS, f"shot-model backend {name!r} already registered")
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_shot_model(
    name: str = BASELINE_BACKEND, baseline: BaselineBag | None = None
) -> ShotOutcomeModel:
    """Build the backend ``name``; unknown names fail closed (no silent fallback)."""
    require(
        name in _BACKENDS,
        f"unknown shot-model backend {name!r}; have {available_backends()}",
    )
    model = _BACKENDS[name](baseline if baseline is not None else _tour_baseline())
    ensure(isinstance(model, ShotOutcomeModel), f"backend {name!r} is not a model")
    return model


def quick_sg(baseline: BaselineBag | None = None) -> ShotOutcomeModel:
    """The tour-baseline backend: the always-available quick strokes-gained model."""
    return BaselineShotModel(baseline if baseline is not None else _tour_baseline())
