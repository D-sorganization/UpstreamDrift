"""Imitation Learning framework for robotics.

This module provides algorithms for learning from demonstrations:
- Behavior Cloning (supervised learning)
- DAgger (Dataset Aggregation with expert queries)
- GAIL (Generative Adversarial Imitation Learning)
"""

from __future__ import annotations

from src.learning.imitation.dataset import Demonstration, DemonstrationDataset
from src.learning.imitation.learners import (
    ImitationLearner,
    BehaviorCloning,
    DAgger,
    GAIL,
)

__all__ = [
    "Demonstration",
    "DemonstrationDataset",
    "ImitationLearner",
    "BehaviorCloning",
    "DAgger",
    "GAIL",
]
