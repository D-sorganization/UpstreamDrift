"""NSX-62 (#12587): inertial parameter ensemble on the anthro fixture.

The 41-coordinate Pinocchio anthro model carries a nominal inertia per body.
Over a small window of record frames the unactuated root rows of its torque
regressor, with the ground reaction and the (known) club wrench taken as
observed, identify only a few combinations of those parameters; the rest are
sampled inside the pseudo-inertia LMI (``H-INERTIAL-PARAMETER-ENSEMBLE-PRIOR``)
and each sample's snapshots, feasible sets and envelopes are rebuilt with the
``foot_contact_tol_m`` rows of NSX-60, the unchanged ``H-TORQUE-CAPACITY`` prior
and Coulomb cones.  A sample whose root rows cannot be balanced is counted as
refused, never repaired; a frame that is infeasible under the capacity prior is
reported, never repaired.  The report is modest (three windows of three
consecutive record frames) because the host is shared.  Nothing here is a
subject-specific inertia claim.

Set ``LOOP_NULLSPACE_PARAMETER_ENSEMBLE_JSON`` to a file path to write the
measured record.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import decompose  # noqa: E402
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    BoundStatus,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.ground_envelopes import (  # noqa: E402
    closure_wrench_quantities,
)
from src.shared.python.loop_nullspace.parameter_ensemble import (  # noqa: E402
    PARAMETER_ENSEMBLE_HYPOTHESIS_ID,
    ParameterSamplingHypothesis,
    evaluate_parameter_ensemble,
    identifiability_basis,
    sample_parameters,
    stacked_unactuated_regressor,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter  # noqa: E402

from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    GAP_TOL_M,
    AnthroBench,
    Frame,
    host_record,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]

WINDOW_STARTS = (0, 150, 300)
WINDOW_LENGTH = 3
N_SAMPLES = 10
SEED = 12587
#: 5 percent entries of Delta in J' = J^(1/2) (I + Delta) J^(1/2).
SIGMA_REL = 0.05
#: A wide draw (no envelopes) that shows the LMI rejection mechanism at work.
STRESS_SIGMA_REL = 0.5
STRESS_SAMPLES = 100
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def bench() -> AnthroBench:
    return AnthroBench()


@pytest.fixture(autouse=True)
def _restore_inertia(bench):
    nominal = bench.adapter.inertial_parameters()
    yield
    bench.adapter.set_inertial_parameters(nominal)


def test_adapter_inertia_round_trip_and_regressor_consistency(bench) -> None:
    adapter = bench.adapter
    pi0 = adapter.inertial_parameters()
    assert pi0.shape == (adapter.n_inertial_parameters,) == (410,)
    frame = bench.frame(0)
    snap = frame.snapshot
    y = adapter.torque_regressor(snap.q, snap.v, snap.a)
    np.testing.assert_allclose(y @ pi0, snap.M @ snap.a + snap.h, atol=1e-8)
    changed = pi0.copy().reshape(-1, 10)
    changed[5, 0] *= 1.0  # no-op write must leave the model untouched
    adapter.set_inertial_parameters(changed.ravel())
    np.testing.assert_array_equal(adapter.inertial_parameters(), pi0)
    with pytest.raises(ValueError, match="finite values"):
        adapter.set_inertial_parameters(pi0[:-1])
    assert 0 <= adapter.club_body_index < pi0.size // 10


def _rebuild_frames(bench, frames):
    """Frames at the same ``(q, v, a)`` under the model's current inertia."""
    rebuilt = []
    for frame in frames:
        old = frame.snapshot
        snap = snapshot_from_adapter(
            bench.adapter,
            old.q,
            old.v,
            old.a,
            input_provenance=InputProvenance("test", "anthro parameter ensemble"),
            foot_contact_tol_m=GAP_TOL_M,
        )
        dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
        names = tuple(c.name for c in snap.foot_contacts)
        rebuilt.append(Frame(frame.index, snap, dec, {}, {}, names))
    return rebuilt


def _series(bench, frames, factory):
    sets = [bench.feasible_set(f, None) for f in _rebuild_frames(bench, frames)]
    return static_envelopes(sets, factory)


def _quantities():
    torques, wrench = joint_torque_quantities(None), closure_wrench_quantities()
    return lambda fs: [*torques(fs), *wrench(fs)]


def _window(bench, start):
    indices = tuple(range(start, start + WINDOW_LENGTH))
    frames = [bench.frame(i) for i in indices]
    adapter = bench.adapter
    pi0 = adapter.inertial_parameters()
    regressor = stacked_unactuated_regressor(adapter, [f.snapshot for f in frames])
    basis = identifiability_basis(
        regressor, pi0, fixed_bodies=(adapter.club_body_index,)
    )
    samples = sample_parameters(
        basis, ParameterSamplingHypothesis(sigma_rel=SIGMA_REL), N_SAMPLES, SEED
    )
    stress = sample_parameters(
        basis, ParameterSamplingHypothesis(STRESS_SIGMA_REL), STRESS_SAMPLES, SEED
    )
    factory = _quantities()
    nominal = _series(bench, frames, factory)
    root_change, actuated_change = [], []
    nominal_rows = [(f.snapshot.M @ f.snapshot.a + f.snapshot.h) for f in frames]
    unact = list(frames[0].snapshot.topology.unactuated_indices)
    act = list(frames[0].snapshot.topology.actuated_indices)

    def series_of():
        rebuilt = _rebuild_frames(bench, frames)
        for r, ref in zip(rebuilt, nominal_rows, strict=True):
            diff = r.snapshot.M @ r.snapshot.a + r.snapshot.h - ref
            root_change.append(float(np.max(np.abs(diff[unact]))))
            actuated_change.append(float(np.max(np.abs(diff[act]))))
        return static_envelopes([bench.feasible_set(f, None) for f in rebuilt], factory)

    report = evaluate_parameter_ensemble(samples, adapter, series_of, nominal)
    names = [adapter.coordinate_order[b] for b in basis.sampled_bodies]
    summary = report.to_dict()
    nominal_infeasible = sum(
        st is BoundStatus.INFEASIBLE for st in nominal.chebyshev_status
    )
    medians = [
        q.union_width_over_nominal_median
        for q in report.quantities
        if q.union_width_over_nominal_median is not None
    ]
    return (
        {
            "record_frames": list(indices),
            "sampled_bodies": names,
            "n_parameters_sampled": basis.n_free,
            "regressor_rows": int(regressor.shape[0]),
            "identified_rank_svd_qr": [basis.rank, basis.qr_rank],
            "null_dimension": basis.null_dimension,
            "nominal_infeasible_frames": int(nominal_infeasible),
            "requested": summary["n_requested"],
            "lmi_rejected": summary["n_lmi_rejected"],
            "violating_bodies": {
                adapter.coordinate_order[b]: n
                for b, n in samples.violating_bodies.items()
            },
            "lmi_stress": {
                "sigma_rel": STRESS_SIGMA_REL,
                "requested": STRESS_SAMPLES,
                "accepted": stress.n_accepted,
                "rejected_by_reason": stress.rejected_by_reason,
                "max_identified_drift_scaled": stress.max_identified_drift,
            },
            "refused_unbalanced": summary["n_refused_unbalanced"],
            "enveloped": summary["n_enveloped"],
            "member_frames_infeasible": summary["member_frames_infeasible"],
            "max_identified_drift_scaled": summary["max_identified_drift"],
            "max_root_row_change_nm": max(root_change, default=0.0),
            "max_actuated_row_change_nm": max(actuated_change, default=0.0),
            "union_width_over_nominal_median_of_medians": (
                float(np.median(medians)) if medians else None
            ),
            "quantities": summary["quantities"],
        },
        report,
        samples,
        basis,
        regressor,
    )


@pytest.fixture(scope="module")
def windows(bench):
    return [_window(bench, start) for start in WINDOW_STARTS]


def test_samples_satisfy_lmi_and_identified_combinations_stay_fixed(windows) -> None:
    for record, _report, samples, basis, regressor in windows:
        assert samples.n_accepted >= 1
        for pi in samples.accepted:
            eig = _min_pseudo_eigenvalues(pi)
            assert np.all(eig >= -1e-12)
            scale = np.max(np.abs(regressor @ basis.pi0))
            assert np.max(np.abs(regressor @ (pi - basis.pi0))) < 1e-9 * scale
        assert record["max_identified_drift_scaled"] < 1e-9
        stress = record["lmi_stress"]
        assert stress["accepted"] + sum(stress["rejected_by_reason"].values()) == (
            STRESS_SAMPLES
        )
        assert stress["max_identified_drift_scaled"] < 1e-9
        # The snapshot's own root rows (M a + h) are unchanged by the inertia swap.
        assert record["max_root_row_change_nm"] < 1e-6


def _min_pseudo_eigenvalues(pi):
    from src.shared.python.estimation.mosaic.inertial import pseudo_inertia
    from src.shared.python.loop_nullspace.parameter_ensemble import (
        swap_inertia_layout,
    )

    bodies = swap_inertia_layout(pi.reshape(-1, 10))
    return np.linalg.eigvalsh(pseudo_inertia(bodies)).min(axis=1)


@pytest.mark.slow
def test_anthro_parameter_sensitivity_report(windows, bench) -> None:
    pytest.importorskip("highspy")
    for record, report, samples, _basis, _y in windows:
        assert record["requested"] == N_SAMPLES
        assert samples.n_accepted + samples.n_rejected == N_SAMPLES
        assert record["enveloped"] + record["refused_unbalanced"] == samples.n_accepted
        assert record["null_dimension"] > 0
        assert report.hypothesis_ids == (PARAMETER_ENSEMBLE_HYPOTHESIS_ID,)
        for q in report.quantities:
            med = q.union_width_over_nominal_median
            assert med is None or med >= 1.0 - 1e-6  # union contains the nominal
    assert sum(w[0]["enveloped"] for w in windows) >= len(WINDOW_STARTS)
    # The unidentified inertia moves the actuated rows, hence some torque bound.
    assert any(w[0]["max_actuated_row_change_nm"] > 0.0 for w in windows)
    assert any(q.max_bound_shift > 0.0 for w in windows for q in w[1].quantities)
    records = [w[0] for w in windows]
    logger.info("anthro parameter ensemble report: %s", records)
    target = os.environ.get("LOOP_NULLSPACE_PARAMETER_ENSEMBLE_JSON")
    if target:
        Path(target).write_text(
            json.dumps(
                {
                    "host": host_record(),
                    "seed": SEED,
                    "sigma_rel": SIGMA_REL,
                    "hypothesis": PARAMETER_ENSEMBLE_HYPOTHESIS_ID,
                    "windows": records,
                },
                indent=2,
                default=str,
            )
        )
