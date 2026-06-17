# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Per-exercise runtime state for the Movement Optimizer GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import BodyModel
from ..trajectory import OptimizationResult


@dataclass
class ExerciseRuntimeState:
    """Mutable runtime state owned as a single unit per exercise tab."""

    result: OptimizationResult | None = None
    anim_frame: int = 0
    body: BodyModel | None = None
    dynamics: Any = None
