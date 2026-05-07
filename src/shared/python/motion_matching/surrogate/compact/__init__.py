"""Compact-schema Option-2 NN swing surrogate (issue #4075).

This package implements the ``SwingSurrogate`` forward model that maps a
189-dim polynomial-coefficient vector to the resulting hand-path
trajectory described by ``COMPACT_DATASET_SCHEMA.md``. Once trained, it
replaces the slow Simscape forward simulation with a millisecond-scale
neural network during motion-matching fits.

Public API:
    SwingSurrogate    -- pure-PyTorch ``nn.Module`` (4-layer MLP w/ residual blocks).
    SurrogateConfig   -- frozen dataclass of architectural hyperparameters.
    CoeffNormalizer   -- normalises the 189-dim coefficient vector to ``[-1, 1]``.
    train_surrogate   -- training entry-point returning a :class:`TrainingResult`.
    TrainingResult    -- frozen dataclass returned by training.
    predict_trajectory -- vectorised inference helper returning physical units.

Sibling implementation:
    The earlier FiLM-MLP variant (PR #4025) lives in
    ``src/shared/python/motion_matching/surrogate/model.py`` and is preserved
    untouched for backward compatibility. The compact-schema variant in this
    subpackage follows the contract documented in #4075 — single 12-channel
    output trajectory of length 31 plus a normalised 189-dim input.
"""

from __future__ import annotations

from .model import CoeffNormalizer, SurrogateConfig, SwingSurrogate
from .predict import predict_trajectory
from .training import TrainingResult, train_surrogate

__all__ = [
    "CoeffNormalizer",
    "SurrogateConfig",
    "SwingSurrogate",
    "TrainingResult",
    "predict_trajectory",
    "train_surrogate",
]
