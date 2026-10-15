"""NSX-23 (#12566) acceptance: anthro-fixture trajectory allocation.

"Anthro fixture allocation converges with recorded solve time and KKT
residuals" is BLOCKED on the foot-contact Jacobian ``J_g``: the Pinocchio
anthro model is ground-supported, ``snapshot_from_adapter`` leaves ``J_g``
unset, and ``decompose`` refuses every frame (see
``test_loop_nullspace_anthro_envelope_blocker.py`` and the blocker section of
``docs/research/loop_nullspace/contracts.md``).  Parallel child NSX-60 adds the
rows.  This test skips with that reason until ``J_g`` exists, then runs the
allocation over evenly strided frames and records solve time and KKT residuals.
It never substitutes a fake ``J_g`` or treats the root as actuated.

The post-unblock branch (everything after the skip) has not been executed: the
skip is the only path that can run today.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation import (  # noqa: E402
    ObjectiveSpec,
    allocate_trajectory,
    torque_scale_from_capacity,
)
from src.shared.python.loop_nullspace.capacity import (  # noqa: E402
    CapacitySubject,
    capacity_for,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import decompose  # noqa: E402
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.registry import preset  # noqa: E402
from src.shared.python.loop_nullspace.snapshot import (  # noqa: E402
    snapshot_from_adapter,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (  # noqa: E402
    EngineType,
    create_engine_force_adapter,
)
from src.shared.python.motion_matching.same_input import (  # noqa: E402
    VectorPlant,
    project_to_closure,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC = EVIDENCE / "full_body_spec_hipcal_scaled.json"
ROOT_PREFIXES = ("Translation", "HipInput")
FRAMES = tuple(range(0, 600, 3))
BLOCKER = (
    "blocked on NSX-60: the anthro adapter supplies no foot-contact Jacobian "
    "J_g, so the ground-supported root rows cannot be balanced "
    "(contracts.md, NSX-22 blocker; test_loop_nullspace_anthro_envelope_blocker)"
)

logger = logging.getLogger(__name__)


def _snapshot(adapter, plant, rec, frame):
    q0, v0, tau = rec["q"][frame], rec["v"][frame], rec["tau"][frame]
    proj = project_to_closure(plant, q0, v0)
    order = list(adapter.coordinate_order)
    named = [
        plant._named(x)
        for x in (proj.q, proj.v, plant.acceleration(proj.q, proj.v, tau))
    ]
    q, v, a = (np.array([d[n] for n in order]) for d in named)
    return snapshot_from_adapter(
        adapter, q, v, a, input_provenance=InputProvenance("test", f"frame {frame}")
    )


def test_anthro_trajectory_allocation_converges_with_recorded_kkt(caplog) -> None:
    adapter = create_engine_force_adapter(
        EngineType.PINOCCHIO, SPEC, allow_synthetic=False
    )
    plant = VectorPlant("pinocchio", SPEC.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        first = _snapshot(adapter, plant, rec, FRAMES[0])
        if first.J_g is None:
            pytest.skip(BLOCKER)
        snaps = [first] + [_snapshot(adapter, plant, rec, f) for f in FRAMES[1:]]
        times = np.asarray(rec["t"])[list(FRAMES)] if "t" in rec else None
    order = list(adapter.coordinate_order)
    actuated = [n for n in order if not n.startswith(ROOT_PREFIXES)]
    cap = capacity_for(order, CapacitySubject(80.0, 1.8), actuated=actuated)
    hyp = preset("smooth_capacity")
    frames = []
    for snap in snaps:
        dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
        frames.append(FeasibleSet.from_hypotheses(dec, hyp, capacity=cap))
    if times is None:
        times = 0.01 * np.arange(len(frames))
    spec = ObjectiveSpec(
        times_s=times,
        torque_scale=torque_scale_from_capacity(cap),
        effort_weight=1.0,
        rate_weight=0.1,
    )
    t0 = time.perf_counter()
    result = allocate_trajectory(frames, spec)
    wall = time.perf_counter() - t0
    with caplog.at_level(logging.INFO):
        logger.info(
            "NSX-23 anthro allocation: %d frames, %.3f s wall, %.3f s solve, KKT %s",
            len(frames),
            wall,
            result.solve_time_s,
            result.kkt_residuals,
        )
    assert result.status.name == "OPTIMAL"
    assert result.kkt_residuals is not None
    assert result.kkt_residuals.max() < 1e-6
    assert result.record is not None
