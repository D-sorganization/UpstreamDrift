"""Planar-fixture trajectory and engine for the NSX-40 report tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.contracts import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.facade import EngineContext, TrajectoryInput
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "loop-nullspace-report-v1.json"
)
N_FRAMES = 4


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def planar_capacity(limit_nm: float = 500.0, n: int = 4) -> CapacityVector:
    names = tuple(f"j{i}" for i in range(n))
    return CapacityVector(
        coordinate_order=names,
        actuated_indices=tuple(range(n)),
        actuated_names=names,
        tau_max_negative_nm=np.full(n, limit_nm),
        tau_max_positive_nm=np.full(n, limit_nm),
        assumed_slots=(names[0],),
        verified_slots=names[1:],
        warnings=(),
        force_velocity={},
    )


def planar_trajectory(indices: tuple[int, ...] = (0, 3, 6, 9)) -> TrajectoryInput:
    snap = planar_golfer_snapshot(
        generic_planar_config(), actuation="club_unactuated"
    ).snapshot
    return TrajectoryInput(
        snapshots=(snap,) * len(indices),
        times_s=tuple(0.01 * i for i in range(len(indices))),
        frame_indices=indices,
        provenance=InputProvenance("synthetic", "planar-fixture", "repeated frame"),
    )


def planar_engine(limit_nm: float = 500.0) -> EngineContext:
    return EngineContext(
        name="planar-numpy",
        model_provenance=ModelProvenance("planar_two_arm_golfer_numpy", "1", "abc"),
        capacity=planar_capacity(limit_nm),
    )


class PlanarSource:
    """A server-side trajectory source over the planar fixture (unit tests)."""

    def __init__(self, limit_nm: float = 500.0) -> None:
        self.limit_nm = limit_nm

    def describe(self, request):
        from src.api.services.loop_nullspace_service import SourceDescription

        return SourceDescription(
            {"kind": "synthetic", "source": "planar-fixture"},
            {
                "name": "planar_two_arm_golfer_numpy",
                "version": "1",
                "limit": self.limit_nm,
            },
        )

    def load(self, request):
        return planar_trajectory(tuple(request.frame_indices)), planar_engine(
            self.limit_nm
        )
