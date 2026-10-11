"""Same-input cross-engine dynamics parity (epic #11605).

The public facade keeps the engine-agnostic API available while loading each
provider only when one of its symbols is requested. In particular, importing
the closure projection does not initialize reference generation or optional
engine and visualization dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ALL_ENGINES": (".plant", "ALL_ENGINES"),
    "OPTIONAL_ENGINES": (".plant", "OPTIONAL_ENGINES"),
    "PARITY_ENGINES": (".plant", "PARITY_ENGINES"),
    "ROOT_COORDINATES": (".integrator", "ROOT_COORDINATES"),
    "SCHEMA": (".bundle", "SCHEMA"),
    "ClosureProjection": (".closure", "ClosureProjection"),
    "InputBundle": (".bundle", "InputBundle"),
    "ReplayScore": (".scoring", "ReplayScore"),
    "Rollout": (".integrator", "Rollout"),
    "StepPolicy": (".integrator", "StepPolicy"),
    "VectorPlant": (".plant", "VectorPlant"),
    "closed_loop": (".reference", "closed_loop"),
    "generate_reference_bundle": (".reference", "generate_reference_bundle"),
    "growth_rate": (".scoring", "growth_rate"),
    "integrate": (".integrator", "integrate"),
    "open_loop": (".integrator", "open_loop"),
    "project_to_closure": (".closure", "project_to_closure"),
    "score_replay": (".scoring", "score_replay"),
    "segmented_replay": (".scoring", "segmented_replay"),
    "zoh_rk4_step": (".integrator", "zoh_rk4_step"),
}

__all__ = [
    "ALL_ENGINES",
    "OPTIONAL_ENGINES",
    "PARITY_ENGINES",
    "ROOT_COORDINATES",
    "SCHEMA",
    "ClosureProjection",
    "InputBundle",
    "ReplayScore",
    "Rollout",
    "StepPolicy",
    "VectorPlant",
    "closed_loop",
    "generate_reference_bundle",
    "growth_rate",
    "integrate",
    "open_loop",
    "project_to_closure",
    "score_replay",
    "segmented_replay",
    "zoh_rk4_step",
]

if TYPE_CHECKING:
    from .bundle import SCHEMA, InputBundle
    from .closure import ClosureProjection, project_to_closure
    from .integrator import (
        ROOT_COORDINATES,
        Rollout,
        StepPolicy,
        integrate,
        open_loop,
        zoh_rk4_step,
    )
    from .plant import ALL_ENGINES, OPTIONAL_ENGINES, PARITY_ENGINES, VectorPlant
    from .reference import closed_loop, generate_reference_bundle
    from .scoring import ReplayScore, growth_rate, score_replay, segmented_replay


def __getattr__(name: str) -> Any:
    """Load a public symbol from its provider only when requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public symbols in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
