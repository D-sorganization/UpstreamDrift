"""NSX-61 (#12586): kinematic ensemble on the anthro fixture.

Sampling and admission are kinematic, so they run on the ground-supported
anthro model without a foot Jacobian ``J_g``: the 41-coordinate Pinocchio
model, three consecutive frames of ``ground_support/dynamics_record.npz``.
The *envelope* part of the ensemble report needs ``decompose``, which still
refuses every anthro frame until the lower-body loop (NSX-60, #12585) adds
foot-contact rows; that test skips with the blocker named, and fails once the
blocker lifts so that it must be replaced by the real ensemble envelope.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from src.shared.python.loop_nullspace.admission import (  # noqa: E402
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    BalanceInconsistentError,
    decompose,
)
from src.shared.python.loop_nullspace.ensemble import (  # noqa: E402
    ENSEMBLE_HYPOTHESIS_ID,
    EnsembleSource,
    SamplingHypothesis,
    kinematic_ensemble,
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
FRAMES = (0, 1, 2)
N_SAMPLES = 8
SEED = 12586
SIGMA_Q = 0.02
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def adapter():
    return create_engine_force_adapter(
        EngineType.PINOCCHIO, SPEC, allow_synthetic=False
    )


class _AdapterOrderPlant:
    """``VectorPlant`` (spec order) seen in the adapter's coordinate order."""

    def __init__(self, plant: VectorPlant, order: tuple[str, ...]) -> None:
        self.plant = plant
        self.order = order
        self.perm = [list(plant.coordinate_order).index(n) for n in order]

    def _to_plant(self, x):
        return self.plant._ordered(dict(zip(self.order, x, strict=True)))

    def closure_pose_residual(self, q):
        return self.plant.closure_pose_residual(self._to_plant(q))

    def closure_rate_matrix(self, q):
        return self.plant.closure_rate_matrix(self._to_plant(q))[:, self.perm]


@pytest.fixture(scope="module")
def vector_plant():
    return VectorPlant("pinocchio", SPEC.read_bytes())


@pytest.fixture(scope="module")
def plant(adapter, vector_plant):
    return _AdapterOrderPlant(vector_plant, tuple(adapter.coordinate_order))


def _rebuild(adapter):
    def rebuild(q, v, a):
        return snapshot_from_adapter(
            adapter, q, v, a, input_provenance=InputProvenance("test", "anthro")
        )

    return rebuild


@pytest.fixture(scope="module")
def nominal(adapter, plant, vector_plant):
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        times = rec["time_s"][list(FRAMES)]
        states = [(rec["q"][k], rec["v"][k], rec["tau"][k]) for k in FRAMES]
    rebuild = _rebuild(adapter)
    snaps = []
    for q0, v0, tau in states:
        proj = project_to_closure(vector_plant, q0, v0)
        order = list(adapter.coordinate_order)
        q, v, a = (
            np.array([d[n] for n in order])
            for d in (
                vector_plant._named(proj.q),
                vector_plant._named(proj.v),
                vector_plant._named(vector_plant.acceleration(proj.q, proj.v, tau)),
            )
        )
        snaps.append(rebuild(q, v, a))
    report = admit(
        snaps,
        gates=AdmissionGates(),
        times=times,
        plant=plant,
        repair="project_to_closure",
        rebuild=rebuild,
    )
    return AdmittedTrajectory.from_report(report)


@pytest.fixture(scope="module")
def result(adapter, plant, nominal):
    source = EnsembleSource(
        nominal=nominal,
        plant=plant,
        rebuild=_rebuild(adapter),
        sampling=SamplingHypothesis(sigma_q=SIGMA_Q),
    )
    return kinematic_ensemble(source, N_SAMPLES, SEED)


def test_anthro_ensemble_admitted_and_rejected_counts(result) -> None:
    logger.info(
        "anthro ensemble n=%d seed=%d sigma_q=%g: accepted=%d (repaired=%d) "
        "rejected=%d by %s",
        result.n_requested,
        result.seed,
        result.sigma_q,
        result.n_accepted,
        result.n_repaired,
        result.n_rejected,
        result.rejected_by_reason,
    )
    assert result.n_accepted + result.n_rejected == N_SAMPLES
    assert result.hypothesis_id == ENSEMBLE_HYPOTHESIS_ID
    assert result.n_accepted >= 1


def test_anthro_samples_differ_from_nominal_and_are_deterministic(
    result, nominal, adapter, plant
) -> None:
    q0 = nominal.snapshots[0].q
    q1 = result.accepted[0].snapshots[0].q
    assert 0.0 < np.max(np.abs(q1 - q0)) < 1.0
    assert np.max(np.abs(plant.closure_pose_residual(q1))) < 1e-6


def test_anthro_ensemble_envelope_is_blocked_on_foot_contact(result, adapter) -> None:
    snap = result.accepted[0].snapshots[0]
    try:
        decompose(snap, weights=np.eye(snap.topology.n_actuated))
    except BalanceInconsistentError:
        pytest.skip(
            "ensemble envelope BLOCKED: anthro decompose refuses ground-supported "
            "frames until NSX-60 (#12585) adds foot-contact rows J_g; see "
            "docs/research/loop_nullspace/contracts.md"
        )
    pytest.fail("blocker lifted: replace this test with the anthro ensemble envelope")
