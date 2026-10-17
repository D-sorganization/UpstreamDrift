"""Server-side trajectory source over the Pinocchio anthro ground-support fixture.

The API never receives arrays.  A request names ``anthro_ground_support`` and
frame indices of ``ground_support/dynamics_record.npz``; this source builds the
snapshots with the NSX-60 test support (``AnthroBench``), so the report is the
same computation as the 200-frame benchmark.  A frame whose balance cannot be
closed (the airborne frame 510) is handed to the facade as *refused*.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache, cache
from pathlib import Path

from src.api.services.loop_nullspace_service import (
    LoopNullspaceRequest,
    SourceDescription,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance, ModelProvenance
from src.shared.python.loop_nullspace.decomposition import BalanceInconsistentError
from src.shared.python.loop_nullspace.facade import (
    EngineContext,
    RefusedFrame,
    TrajectoryInput,
)
from tests.integration.loop_nullspace_lower_body_support import (
    RECORD,
    SIDES,
    SPEC,
    AnthroBench,
)

SOURCE_ID = "anthro_ground_support"


@cache
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _bench() -> AnthroBench:
    return AnthroBench()


class _AdapterOrderPlant:
    """The plant's closure residual for a ``q`` in the adapter's coordinate order.

    Snapshots carry ``q`` in ``adapter.coordinate_order`` while ``VectorPlant``
    keeps its own order; the benchmark converts through coordinate names, and so
    does this wrapper.  Without it the position residual is meaningless.
    """

    def __init__(self, bench: AnthroBench) -> None:
        self._plant = bench.plant
        self._order = list(bench.order)

    def _plant_q(self, q):
        return self._plant._ordered(dict(zip(self._order, map(float, q), strict=True)))

    def closure_pose_residual(self, q):
        return self._plant.closure_pose_residual(self._plant_q(q))

    def closure_rate_matrix(self, q):
        return self._plant.closure_rate_matrix(self._plant_q(q))


class AnthroGroundSupportSource:
    """``TrajectorySource`` over the committed anthro ground-support evidence."""

    def describe(self, request: LoopNullspaceRequest) -> SourceDescription:
        if request.engine != "pinocchio":
            raise ValueError(
                f"engine {request.engine!r} is not supported; use pinocchio"
            )
        n = len(_bench().record["time_s"])
        if max(request.frame_indices) >= n:
            raise ValueError(f"frame index beyond the {n}-frame record")
        return SourceDescription(
            {
                "kind": "dynamics_record",
                "source": "ground_support/dynamics_record.npz",
                "sha256": _sha256(RECORD),
            },
            {
                "name": "anthro_full_body_pinocchio",
                "version": "full_body_spec_hipcal_scaled",
                "content_hash": _sha256(SPEC),
            },
        )

    def load(
        self, request: LoopNullspaceRequest
    ) -> tuple[TrajectoryInput, EngineContext]:
        bench = _bench()
        frames, refused = {}, []
        for i in request.frame_indices:
            try:
                frames[i] = bench.frame(i)
            except BalanceInconsistentError as err:
                refused.append(
                    RefusedFrame(i, "decomposition", str(err), err.residual_scaled)
                )
        kept = sorted(frames)
        description = self.describe(request)
        engine = EngineContext(
            name="pinocchio",
            model_provenance=ModelProvenance(
                "anthro_full_body_pinocchio",
                "full_body_spec_hipcal_scaled",
                description.model_provenance["content_hash"],
            ),
            capacity=bench.capacity,
            plant=_AdapterOrderPlant(bench),
            club_newton_euler=lambda snap: bench.adapter.club_newton_euler(
                snap.q, snap.v, snap.a
            ),
            observation_for=lambda index, _snap: bench.record_observation(
                frames[index]
            ),
            ground_groups=SIDES,
        )
        trajectory = TrajectoryInput(
            snapshots=tuple(frames[i].snapshot for i in kept),
            times_s=tuple(float(bench.record["time_s"][i]) for i in kept),
            frame_indices=tuple(kept),
            provenance=InputProvenance(
                "dynamics_record",
                "ground_support/dynamics_record.npz",
                f"sha256={description.input_provenance['sha256'][:16]}",
            ),
            refused=tuple(refused),
        )
        return trajectory, engine
