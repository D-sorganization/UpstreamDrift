# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Imitation learning algorithms."""

from __future__ import annotations

from src.learning.imitation._base import ImitationLearner, TrainingConfig
from src.learning.imitation._bc import BehaviorCloning
from src.learning.imitation._dagger import DAgger
from src.learning.imitation._gail import GAIL

__all__ = [
    "BehaviorCloning",
    "DAgger",
    "GAIL",
    "ImitationLearner",
    "TrainingConfig",
]
