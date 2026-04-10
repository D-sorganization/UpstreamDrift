"""Coordination and sequencing plotting renderer."""

from __future__ import annotations

from src.shared.python.plotting.renderers._coordination_alignment import (
    CoordinationAlignmentMixin,
)
from src.shared.python.plotting.renderers._coordination_phase import (
    CoordinationPhaseMixin,
)
from src.shared.python.plotting.renderers._coordination_sequence import (
    CoordinationSequenceMixin,
)
from src.shared.python.plotting.renderers._coordination_synergy import (
    CoordinationSynergyMixin,
)
from src.shared.python.plotting.renderers.base import BaseRenderer

__all__ = ["CoordinationRenderer"]


class CoordinationRenderer(
    CoordinationPhaseMixin,
    CoordinationAlignmentMixin,
    CoordinationSequenceMixin,
    CoordinationSynergyMixin,
    BaseRenderer,
):
    """Renderer for coordination, sequencing, and variability plots."""
