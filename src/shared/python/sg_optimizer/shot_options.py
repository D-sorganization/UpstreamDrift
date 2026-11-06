"""Per-scenario shot options with dispersion ellipses (SHOW-B8, #14572)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.mdp.action import ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline_model import BaselineShotModel
from src.shared.python.sg_optimizer.shot_model.outcome import (
    LandingSamples,
    ShotOutcomeModel,
)


@dataclass(frozen=True)
class DispersionEllipse:
    """Landing ellipse in the aim frame (yards); ``angle_deg`` is the major axis
    measured from the along-shot axis."""

    centre_along: float
    centre_lateral: float
    semi_major: float
    semi_minor: float
    angle_deg: float


@dataclass(frozen=True)
class ShotOption:
    """The best aim for one club, its expected strokes and where it lands."""

    club: str
    aim_deg: float
    expected_strokes: float
    ellipse: DispersionEllipse


def _fold_axis_deg(angle: float) -> float:
    """An axis has no direction: fold the angle into (-90, 90]."""
    folded = (angle + 90.0) % 180.0 - 90.0
    return 90.0 if folded == -90.0 else folded


def dispersion_ellipse(
    samples: LandingSamples, coverage: float = 0.95
) -> DispersionEllipse:
    """Ellipse holding ``coverage`` of a Gaussian fitted to the landings."""
    require(0.0 < coverage < 1.0, "coverage must be in (0, 1)", coverage)
    require(len(samples) >= 2, "need at least two samples", len(samples))
    pts = np.stack([samples.along, samples.lateral])
    centre = pts.mean(axis=1)
    eigval, eigvec = np.linalg.eigh(np.cov(pts))
    k = math.sqrt(-2.0 * math.log(1.0 - coverage))
    major = eigvec[:, 1]
    return DispersionEllipse(
        centre_along=float(centre[0]),
        centre_lateral=float(centre[1]),
        semi_major=k * math.sqrt(max(float(eigval[1]), 0.0)),
        semi_minor=k * math.sqrt(max(float(eigval[0]), 0.0)),
        angle_deg=_fold_axis_deg(math.degrees(math.atan2(major[1], major[0]))),
    )


def shot_options(
    mdp: HoleMDP,
    value: NDArray[np.float64],
    state: State,
    *,
    model: ShotOutcomeModel | None = None,
    top: int = 5,
    n: int = 400,
) -> tuple[ShotOption, ...]:
    """Best aim per club at ``state``, best first, with its dispersion ellipse."""
    require(top > 0, "top must be > 0", top)
    best: dict[str, tuple[ShotAction, float]] = {}
    for action, q in mdp.action_values(state, value):
        if action.club not in best or q < best[action.club][1]:
            best[action.club] = (action, q)
    sampler = model if model is not None else BaselineShotModel(mdp.baseline)
    rng = np.random.default_rng(0)
    options = [
        ShotOption(
            club,
            math.degrees(action.aim_angle_rad),
            q,
            dispersion_ellipse(
                sampler.sample(mdp.profile, action, state.lie, mdp.conditions, rng, n)
            ),
        )
        for club, (action, q) in best.items()
    ]
    options.sort(key=lambda o: o.expected_strokes)
    return tuple(options[:top])
