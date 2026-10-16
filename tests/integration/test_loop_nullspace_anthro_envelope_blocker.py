"""NSX-22 (#12565): why the 200-frame anthro envelope benchmark cannot run yet.

The Pinocchio anthro model (41 coordinates, 35 actuated, root = 6 unactuated)
is ground-supported: the six root rows of ``M a + h = S^T tau + J_c^T lambda``
are balanced by foot-ground loads.  ``snapshot_from_adapter`` leaves ``J_g``
unset ("foot contacts are not part of this seam"), so ``decompose`` correctly
refuses every frame with ``BalanceInconsistentError`` (the unactuated rows
cannot be balanced by the closure alone).  Substituting a fake ``J_g`` or
treating the root as actuated would change the problem, so this test pins the
blocker instead.  It fails (and must be replaced by the real benchmark) once
the adapter supplies ``J_g`` and the contact cones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.capacity import (  # noqa: E402
    CapacitySubject,
    capacity_for,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    BalanceInconsistentError,
    decompose,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter  # noqa: E402
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


@pytest.fixture(scope="module")
def adapter():
    return create_engine_force_adapter(
        EngineType.PINOCCHIO, SPEC, allow_synthetic=False
    )


@pytest.fixture(scope="module")
def plant():
    return VectorPlant("pinocchio", SPEC.read_bytes())


def test_anthro_capacity_rows_exist_for_every_actuated_coordinate(adapter) -> None:
    order = list(adapter.coordinate_order)
    actuated = [n for n in order if not n.startswith(ROOT_PREFIXES)]
    cap = capacity_for(order, CapacitySubject(80.0, 1.8), actuated=actuated)
    assert len(cap.actuated_names) == 35


@pytest.mark.parametrize("frame", [0, 300])
def test_ground_supported_frames_are_refused_without_foot_contact(
    adapter, plant, frame
) -> None:
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0, tau = rec["q"][frame], rec["v"][frame], rec["tau"][frame]
    proj = project_to_closure(plant, q0, v0)
    order = list(adapter.coordinate_order)
    named = [
        plant._named(x)
        for x in (proj.q, proj.v, plant.acceleration(proj.q, proj.v, tau))
    ]
    q, v, a = (np.array([d[n] for n in order]) for d in named)
    snap = snapshot_from_adapter(
        adapter, q, v, a, input_provenance=InputProvenance("test", f"frame {frame}")
    )
    assert snap.J_g is None
    assert [order[i] for i in snap.topology.unactuated_indices][:3] == [
        "TranslationInputX",
        "TranslationInputY",
        "TranslationInputZ",
    ]
    with pytest.raises(BalanceInconsistentError) as err:
        decompose(snap, weights=np.eye(snap.topology.n_actuated))
    assert err.value.residual_scaled > 1.0
