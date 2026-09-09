"""Axial color source adapter for analytical double/triple pendulum reactions."""

from __future__ import annotations

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)

from .simulation import SimulationResult
from .simulation_triple import TripleSimulationResult


def pendulum_axial_loads(result: object, frame_idx: int) -> AxialLoadFrame | None:
    """Project existing transmitted joint reactions at each proximal section.

    Unsupported result types return None. In particular the golfer point-mass
    net-force output is not a transmitted section reaction and is not reused.
    The existing result validates frame indices and computes dynamics unchanged.
    """
    if isinstance(result, TripleSimulationResult):
        links = (
            ("arm", "shoulder", "wrist1", "shoulder"),
            ("forearm", "wrist1", "wrist2", "wrist1"),
            ("club", "wrist2", "tip", "wrist2"),
        )
    elif isinstance(result, SimulationResult):
        links = (("arm", "hub", "wrist", "shoulder"), ("club", "wrist", "tip", "wrist"))
    else:
        return None
    positions = result.positions_at(frame_idx)
    reactions = result.joint_forces_at(frame_idx)
    values = {
        segment: axial_force_from_proximal_reaction(
            reactions[reaction], positions[proximal], positions[distal]
        )
        for segment, proximal, distal, reaction in links
    }
    return AxialLoadFrame(
        float(result.t[frame_idx]),
        values,
        "Analytical pendulum parent-on-segment joint reaction at proximal section; world XY",
    )
