"""SwingSurrogate FiLM-MLP forward surrogate for Option 2.

This package implements the differentiable forward surrogate
``f_theta : coefficients -> club kinematic trajectory`` per the design
in ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/
motion_matching/option2_nn_surrogate/APPROACH.md``.

Public API:
    ClubTrajectory   -- batch-first surrogate output (butt, clubhead, q, joints).
    SurrogateConfig  -- frozen architectural configuration.
    SwingSurrogate   -- the ``nn.Module`` itself.
    NormalizationStats -- per-feature z-score statistics fitted on train split.
    TrainConfig      -- training hyperparameters and loss weights.
    TrainedSurrogate -- bundle of (model, stats, curves) returned by training.
    train_surrogate  -- end-to-end training entry-point.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._normalize import NormalizationStats
    from .invert import FitResult, InvertOptions
    from .invert import fit_swing_via_surrogate
    from .model import ClubTrajectory, SurrogateConfig, SwingSurrogate
    from .train import TrainConfig, TrainedSurrogate
    from .train import train_surrogate
    from .validate import ValidationReport
    from .validate import validate_against_simscape

__all__ = [
    "ClubTrajectory",
    "FitResult",
    "InvertOptions",
    "NormalizationStats",
    "SurrogateConfig",
    "SwingSurrogate",
    "TrainConfig",
    "TrainedSurrogate",
    "ValidationReport",
    "fit_swing_via_surrogate",
    "train_surrogate",
    "validate_against_simscape",
]

_LAZY_EXPORTS = {
    "ClubTrajectory": ".model",
    "FitResult": ".invert",
    "InvertOptions": ".invert",
    "NormalizationStats": "._normalize",
    "SurrogateConfig": ".model",
    "SwingSurrogate": ".model",
    "TrainConfig": ".train",
    "TrainedSurrogate": ".train",
    "ValidationReport": ".validate",
    "fit_swing_via_surrogate": ".invert",
    "train_surrogate": ".train",
    "validate_against_simscape": ".validate",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
