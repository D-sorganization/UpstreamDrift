"""Player-relative shot dispersion + putting model.

Curated public API. Internals (``_*``) are private.
"""

from __future__ import annotations

from src.shared.python.sg_optimizer.shot_model.baseline import (
    BaselineBag,
    ClubBaseline,
    load_baseline,
)
from src.shared.python.sg_optimizer.shot_model.distributions import (
    TiltedBivariateGaussian,
)
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
    PuttingSkill,
)
from src.shared.python.sg_optimizer.shot_model.putting import (
    leave_distance_distribution,
    make_probability,
)
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    CurvePuttingProvider,
    LogisticPuttingProvider,
    PuttingProvider,
    SimulatedPuttingCurve,
    putting_provider,
)

__all__ = [
    "BaselineBag",
    "ClubBaseline",
    "ClubSkill",
    "CurvePuttingProvider",
    "LogisticPuttingProvider",
    "PlayerProfile",
    "PuttingProvider",
    "PuttingSkill",
    "SimulatedPuttingCurve",
    "TiltedBivariateGaussian",
    "leave_distance_distribution",
    "load_baseline",
    "make_probability",
    "putting_provider",
]
