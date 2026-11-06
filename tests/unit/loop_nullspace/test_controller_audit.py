"""Controller synergy audit on Option A-2 traces, slice 1 (#13884).

The audit reuses the NSX-97 UCM/TNC modules and the NSX-93 gain norm. Its
analytic golden: for a linear closed loop ``x+ = (A - B K) x + G w`` the Monte
Carlo ``V_UCM``/``V_ORT`` match the Lyapunov covariance projected on ``N(C)``
and ``R(C^T)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.loop_nullspace.controller_audit import (
    AuditInputs,
    ControllerAuditSpec,
    audit_controller,
    audit_inputs_from_traces,
    predicted_ucm_variances,
)
from src.shared.python.loop_nullspace.minimum_intervention import (
    minimum_intervention_design,
)
from src.shared.python.loop_nullspace.variance_propagation import (
    propagate_covariance,
)

pytestmark = pytest.mark.unit

_C = np.array([[1.0, 1.0, 0.0, 0.0]])
_SCALE = np.array([1.0, 1.0, 2.0, 2.0])
_COORDS = ("q0", "q1", "v0", "v1")
_SEED = 13884


def _spec(**overrides) -> ControllerAuditSpec:
    fields = {
        "task_jacobian": _C,
        "state_scale": _SCALE,
        "coordinates": _COORDS,
        "seed": _SEED,
    }
    fields.update(overrides)
    return ControllerAuditSpec(**fields)


def _plant(steps: int = 12) -> LinearizedDynamics:
    rng = np.random.default_rng(7)
    dt = 0.05
    a = np.array(
        [
            np.eye(4) + dt * np.block([[np.zeros((2, 2)), np.eye(2)], [m, -np.eye(2)]])
            for m in rng.normal(scale=0.5, size=(steps, 2, 2))
        ]
    )
    b = np.repeat(dt * np.vstack([np.zeros((2, 2)), np.eye(2)])[None], steps, 0)
    return LinearizedDynamics(a, b)


def _closed_loop(plant: LinearizedDynamics, gains: np.ndarray) -> np.ndarray:
    return np.asarray(plant.a - plant.b @ gains)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"task_jacobian": np.ones((4, 4))}, "dim"),
        ({"task_jacobian": np.array([[1.0, np.nan, 0.0, 0.0]])}, "finite"),
        ({"state_scale": None}, "state_scale"),
        ({"state_scale": np.array([1.0, 0.0, 1.0, 1.0])}, "state_scale"),
        ({"coordinates": ("q0", "q1")}, "coordinates"),
        ({"min_tnc_trials": 1}, "min_tnc_trials"),
    ],
)
def test_spec_rejects_invalid_task_maps(overrides, fragment) -> None:
    with pytest.raises(ValueError, match=fragment):
        _spec(**overrides)


def test_analytic_golden_matches_projected_lyapunov_covariance() -> None:
    plant = _plant()
    gains = tvlqr_gains(plant, np.eye(4), 0.1 * np.eye(2), np.eye(4))
    # Noise enters through G = B, so the closed loop reuses the plant's B.
    closed = LinearizedDynamics(_closed_loop(plant, gains), plant.b)
    sigma = np.repeat((0.3 * np.eye(2))[None], plant.n_steps, 0)
    p0 = np.diag([0.02, 0.03, 0.05, 0.04])
    p_terminal = propagate_covariance(closed, p0, sigma)[-1]

    rng = np.random.default_rng(_SEED)
    x = rng.multivariate_normal(np.zeros(4), p0, size=6000)
    for t in range(plant.n_steps):
        w = rng.multivariate_normal(np.zeros(2), sigma[t], size=x.shape[0])
        x = x @ closed.a[t].T + w @ plant.b[t].T

    spec = _spec()
    report = audit_controller(spec, AuditInputs(x, ("recovered",) * x.shape[0]))
    v_ucm, v_ort = predicted_ucm_variances(spec, p_terminal)
    assert report.ucm.v_ucm == pytest.approx(v_ucm, rel=0.06)
    assert report.ucm.v_ort == pytest.approx(v_ort, rel=0.06)


def test_tnc_fails_closed_below_minimum_trials_and_records_seed() -> None:
    rng = np.random.default_rng(1)
    few = audit_controller(_spec(), AuditInputs(rng.normal(size=(29, 4)), ("r",) * 29))
    assert few.tnc is None
    assert "29 < 30" in few.tnc_withheld
    enough = audit_controller(
        _spec(), AuditInputs(rng.normal(size=(30, 4)), ("r",) * 30)
    )
    assert enough.tnc_withheld is None
    assert enough.tnc.seed == _SEED
    assert enough.tnc.n_trials == 30
    assert enough.tnc.identity_residual < 1e-9


def test_report_records_task_map_coordinates_seed_and_strata() -> None:
    rng = np.random.default_rng(2)
    labels = ("recovered",) * 5 + ("unrecoverable",) * 3
    report = audit_controller(_spec(), AuditInputs(rng.normal(size=(8, 4)), labels))
    record = report.as_record()
    assert record["task_jacobian"] == _C.tolist()
    assert record["coordinates"] == list(_COORDS)
    assert record["seed"] == _SEED
    assert record["strata"] == {"recovered": 5, "unrecoverable": 3}
    assert record["null_space_gain_norm"] is None


def test_strata_labels_never_change_the_task_variable() -> None:
    states = np.random.default_rng(3).normal(size=(8, 4))
    first = audit_controller(_spec(), AuditInputs(states, ("recovered",) * 8))
    second = audit_controller(_spec(), AuditInputs(states, ("unrecoverable",) * 8))
    assert first.ucm == second.ucm


def test_gain_norm_separates_minimum_intervention_from_full_state() -> None:
    plant = _plant()
    states = np.random.default_rng(4).normal(size=(8, 4))
    inputs = AuditInputs(states, ("recovered",) * 8)
    mip = minimum_intervention_design(plant, _C, np.eye(1), 0.1 * np.eye(2))
    full = tvlqr_gains(plant, np.eye(4), 0.1 * np.eye(2), _C.T @ _C)
    lean = audit_controller(_spec(), inputs, gains=mip.gains, plant=plant)
    busy = audit_controller(_spec(), inputs, gains=full, plant=plant)
    assert lean.null_space_gain_norm <= 1e-10
    assert busy.null_space_gain_norm > 1e-3
    with pytest.raises(ValueError, match="together"):
        audit_controller(_spec(), inputs, gains=full)


def test_adapter_takes_one_terminal_state_per_trace_trial() -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.humanoid_robotics.option_a.fixture import (
        run_swing_recovery_fixture,
    )

    bundle = run_swing_recovery_fixture(13837)
    inputs = audit_inputs_from_traces(bundle)
    trials = np.asarray(bundle["trial_index"])
    assert inputs.states.shape == (8, 4)
    last = np.flatnonzero(trials == 3)[-1]
    np.testing.assert_array_equal(inputs.states[3], bundle["x_k"][last])
    assert inputs.strata == tuple(str(s) for s in bundle["outcome_label"])
    with pytest.raises(ValueError, match="unsupported"):
        audit_inputs_from_traces({**bundle, "schema_version": "v0"})
