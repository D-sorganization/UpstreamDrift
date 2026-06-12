"""Per-step dynamics surrogate (Option 2 — `(q, q_dot, tau) -> (q, q_dot, q_ddot)`).

This subpackage hosts PR #3966's per-step approach. It complements the
trajectory-level surrogate exposed by the parent
``src.shared.python.motion_matching.surrogate`` package. See
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
option2_nn_surrogate/APPROACH.md`` for the per-step vs trajectory tradeoff.

Public API:
    DynamicsMLP, TrainConfig, train_dynamics_surrogate -- per-step trainer.
    optimize_torque_sequence                           -- Adam-on-grid inversion.
    extract_dataset                                    -- parquet slimmer used
                                                          to build the per-step
                                                          training dataset.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .extract_dataset import main as extract_dataset
    from .optimize import main as optimize_torque_sequence
    from .train import DynamicsMLP, TrainConfig
    from .train import main as train_dynamics_surrogate

__all__ = [
    "DynamicsMLP",
    "TrainConfig",
    "extract_dataset",
    "optimize_torque_sequence",
    "train_dynamics_surrogate",
]

_LAZY_EXPORTS = {
    "extract_dataset": (".extract_dataset", "main"),
    "optimize_torque_sequence": (".optimize", "main"),
    "DynamicsMLP": (".train", "DynamicsMLP"),
    "TrainConfig": (".train", "TrainConfig"),
    "train_dynamics_surrogate": (".train", "main"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name, __package__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
