"""NSX-62: inertial parameter ensemble over the unidentified directions (#12587).

Synthetic oracles: a random stacked regressor of known rank and physically
consistent bodies.  The two contract tests come first: sampled bodies satisfy the
pseudo-inertia LMI, and the identified combinations (the regressor's row space)
do not move.  Nothing here is a subject-specific inertia claim.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.inertial import (
    from_log_cholesky,
    pseudo_inertia,
)
from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.decomposition import BalanceInconsistentError
from src.shared.python.loop_nullspace.envelopes import BoundStatus, SignClass
from src.shared.python.loop_nullspace.parameter_ensemble import (
    PARAMETER_ENSEMBLE_HYPOTHESIS_ID,
    ParameterSamplingHypothesis,
    evaluate_parameter_ensemble,
    identifiability_basis,
    sample_parameters,
    stacked_unactuated_regressor,
    swap_inertia_layout,
)
from tests.unit.loop_nullspace.test_ensemble import _series

pytestmark = pytest.mark.unit

N_BODIES = 5
MASSLESS = 3  # structurally absent body
FIXED = 4  # declared known (the club carrier in the anthro model)


def _consistent_bodies(rng: np.random.Generator) -> np.ndarray:
    """Strictly consistent bodies in the Pinocchio layout, one massless."""
    theta = rng.normal(scale=0.4, size=(N_BODIES, 10))
    theta[:, 0] += 0.5  # log-diagonal: keeps masses of order one
    theta[:, 1:4] -= 0.3
    bodies = swap_inertia_layout(from_log_cholesky(theta))
    bodies[MASSLESS] = 0.0
    return bodies


@pytest.fixture(scope="module")
def pi0() -> np.ndarray:
    return _consistent_bodies(np.random.default_rng(3)).ravel()


@pytest.fixture(scope="module")
def regressor() -> np.ndarray:
    """18 rows (three frames of six root rows) of rank 12 on 50 columns."""
    rng = np.random.default_rng(4)
    left = rng.normal(size=(18, 12))
    return left @ rng.normal(size=(12, 10 * N_BODIES))


@pytest.fixture(scope="module")
def basis(pi0, regressor):
    return identifiability_basis(regressor, pi0, fixed_bodies=(FIXED,))


@pytest.fixture(scope="module")
def samples(basis):
    return sample_parameters(basis, ParameterSamplingHypothesis(sigma_rel=0.05), 40, 7)


def _lmi_min_eigenvalues(pi: np.ndarray) -> np.ndarray:
    bodies = swap_inertia_layout(pi.reshape(-1, 10))
    return np.linalg.eigvalsh(pseudo_inertia(bodies)).min(axis=1)


# ------------------------------------------------------ contract test 1: LMI --
def test_nominal_and_every_accepted_sample_satisfy_the_pseudo_inertia_lmi(
    pi0, samples
) -> None:
    assert samples.n_accepted >= 10
    assert np.all(_lmi_min_eigenvalues(pi0) >= -1e-12)
    for pi in samples.accepted:
        assert np.all(_lmi_min_eigenvalues(pi) >= -1e-12)


def test_rejections_are_counted_never_projected(basis) -> None:
    wide = sample_parameters(basis, ParameterSamplingHypothesis(sigma_rel=2.0), 30, 1)
    assert wide.n_accepted + wide.n_rejected == 30 == wide.n_requested
    assert wide.n_rejected > 0
    assert set(wide.rejected_by_reason) == {"pseudo_inertia_not_psd"}
    assert len(wide.sample_indices) == wide.n_accepted
    # Accepted draws are the raw null-space draws (no projection onto the cone).
    again = sample_parameters(basis, ParameterSamplingHypothesis(sigma_rel=2.0), 30, 1)
    for k, pi in zip(wide.sample_indices, wide.accepted, strict=True):
        np.testing.assert_array_equal(pi, again.accepted[again.sample_indices.index(k)])
        assert np.all(_lmi_min_eigenvalues(pi) >= -1e-12)
    assert wide.violating_bodies  # per-body diagnostics of what failed


# ------------------------------------- contract test 2: identified combinations --
def test_identified_combinations_stay_fixed_across_samples(
    basis, regressor, samples
) -> None:
    assert basis.rank == 12
    for pi in samples.accepted:
        assert np.max(np.abs(basis.identified_drift(pi))) < 1e-10
        scale = np.max(np.abs(regressor @ basis.pi0))
        assert np.max(np.abs(regressor @ (pi - basis.pi0))) < 1e-9 * scale
    assert samples.max_identified_drift < 1e-10


def test_samples_move_unidentified_directions_only_on_sampled_bodies(
    basis, samples
) -> None:
    stack = np.stack(samples.accepted).reshape(-1, N_BODIES, 10)
    nominal = basis.pi0.reshape(N_BODIES, 10)
    for body in (MASSLESS, FIXED):  # structurally absent / declared known
        assert np.all(stack[:, body] == nominal[body])
    assert np.all(np.std(stack[:, [0, 1, 2]], axis=0) > 0.0)
    assert np.array_equal(samples.accepted[0].reshape(-1, 10)[MASSLESS], np.zeros(10))
    assert samples.sampled_bodies == (0, 1, 2)


# --------------------------------------------------------------- the basis --
def test_basis_reports_rank_qr_and_null_dimension(basis) -> None:
    assert basis.rank == 12 and basis.qr_rank == 12
    assert basis.n_free == 30  # three sampled bodies of ten parameters
    assert basis.null_dimension == 30 - basis.rank
    assert len(basis.base_columns) == basis.rank
    assert len(set(basis.base_columns)) == basis.rank
    assert basis.identified_basis.shape == (30, 12)
    assert basis.null_basis.shape == (30, 18)
    np.testing.assert_allclose(
        basis.null_basis.T @ basis.identified_basis, 0.0, atol=1e-12
    )


def test_full_rank_regressor_leaves_nothing_to_sample(pi0) -> None:
    rng = np.random.default_rng(5)
    full = rng.normal(size=(60, 10 * N_BODIES))
    basis = identifiability_basis(full, pi0, fixed_bodies=(FIXED,))
    assert basis.null_dimension == 0
    with pytest.raises(ValueError, match="no unidentified direction"):
        sample_parameters(basis, ParameterSamplingHypothesis(sigma_rel=0.05), 3, 1)


def test_determinism_and_seed_sensitivity(basis) -> None:
    hyp = ParameterSamplingHypothesis(sigma_rel=0.05)
    a, b, c = (sample_parameters(basis, hyp, 6, s) for s in (7, 7, 8))
    assert a.sample_indices == b.sample_indices
    assert all(
        np.array_equal(x, y) for x, y in zip(a.accepted, b.accepted, strict=True)
    )
    assert not np.allclose(a.accepted[0], c.accepted[0])
    assert a.seed == 7 and a.hypothesis_id == PARAMETER_ENSEMBLE_HYPOTHESIS_ID


# ------------------------------------------------------------ preconditions --
def test_hypothesis_is_declared_in_the_registry() -> None:
    assert (
        registry.hypothesis(PARAMETER_ENSEMBLE_HYPOTHESIS_ID).id
        == PARAMETER_ENSEMBLE_HYPOTHESIS_ID
    )


def test_input_validation(pi0, regressor, basis) -> None:
    with pytest.raises(ValueError, match="sigma_rel"):
        ParameterSamplingHypothesis(sigma_rel=0.0)
    hyp = ParameterSamplingHypothesis(sigma_rel=0.05)
    with pytest.raises(ValueError, match="n must be"):
        sample_parameters(basis, hyp, 0, 1)
    with pytest.raises(TypeError, match="seed"):
        sample_parameters(basis, hyp, 3, 1.5)
    with pytest.raises(ValueError, match="columns"):
        identifiability_basis(regressor[:, :-1], pi0)
    with pytest.raises(ValueError, match="finite"):
        identifiability_basis(np.full_like(regressor, np.nan), pi0)
    with pytest.raises(ValueError, match="fixed body"):
        identifiability_basis(regressor, pi0, fixed_bodies=(N_BODIES,))
    bad = pi0.copy().reshape(-1, 10)
    bad[0, 4:] *= -1.0  # negative inertia: outside the LMI
    with pytest.raises(ValueError, match="not physically consistent"):
        identifiability_basis(regressor, bad.ravel())
    with pytest.raises(ValueError, match="no sampled body"):
        identifiability_basis(regressor, pi0, fixed_bodies=(0, 1, 2, 4))


@pytest.mark.requires_pinocchio
def test_layout_swap_is_an_involution_that_matches_pinocchio_order() -> None:
    pin = pytest.importorskip("pinocchio")
    inertia = pin.Inertia(
        2.0,
        np.array([0.1, 0.2, 0.3]),
        np.array([[2.0, 0.1, 0.2], [0.1, 2.5, 0.3], [0.2, 0.3, 3.0]]),
    )
    native = inertia.toDynamicParameters()[None, :]
    np.testing.assert_array_equal(
        swap_inertia_layout(swap_inertia_layout(native)), native
    )
    # In the LMI layout this real body is consistent; read in the wrong layout the
    # off-diagonal terms would be mis-assigned and the pseudo-inertia would differ.
    assert np.all(np.linalg.eigvalsh(pseudo_inertia(swap_inertia_layout(native))) > 0)
    wrong = np.linalg.eigvalsh(pseudo_inertia(native))
    right = np.linalg.eigvalsh(pseudo_inertia(swap_inertia_layout(native)))
    assert not np.allclose(wrong, right)


# --------------------------------------------- evaluation over a model (fake) --
class _FakeModel:
    """Records the parameters a series factory observed and restores them."""

    def __init__(self, pi0: np.ndarray) -> None:
        self.current = pi0.copy()
        self.history: list[np.ndarray] = []

    def inertial_parameters(self) -> np.ndarray:
        return self.current.copy()

    def set_inertial_parameters(self, pi: np.ndarray) -> None:
        self.current = np.asarray(pi, dtype=float).copy()


def _factory(model: _FakeModel, refuse_every: int = 0):
    calls = {"n": 0}

    def series_of():
        calls["n"] += 1
        model.history.append(model.current.copy())
        if refuse_every and calls["n"] % refuse_every == 0:
            raise BalanceInconsistentError("unbalanced", 1.0)
        shift = float(np.sum(model.current) - np.sum(model.history[0])) * 1e-3
        return _series([[0.0 + shift, -1.0]], [[2.0 + shift, 1.0]])

    return series_of


def test_evaluation_counts_refusals_restores_the_model_and_reports_sensitivity(
    basis, pi0
) -> None:
    samples = sample_parameters(basis, ParameterSamplingHypothesis(0.05), 8, 2)
    model = _FakeModel(pi0)
    nominal = _series([[0.0, -1.0]], [[2.0, 1.0]])
    report = evaluate_parameter_ensemble(
        samples, model, _factory(model, refuse_every=3), nominal
    )
    assert np.array_equal(model.current, pi0)  # restored
    assert report.n_requested == 8
    assert report.n_lmi_rejected == samples.n_rejected
    assert report.n_enveloped + report.n_refused_unbalanced == samples.n_accepted
    assert report.n_refused_unbalanced == samples.n_accepted // 3
    assert report.hypothesis_ids[0] == PARAMETER_ENSEMBLE_HYPOTHESIS_ID
    assert [q.name for q in report.quantities] == ["x0", "x1"]
    assert report.quantities[1].max_bound_shift == pytest.approx(0.0)
    assert report.quantities[0].union_width_over_nominal_max >= 1.0
    assert report.envelope.member_labels[0] == "nominal"
    assert report.non_claims  # no subject-specific inertia claim


def test_evaluation_restores_the_model_when_the_factory_raises(basis, pi0) -> None:
    samples = sample_parameters(basis, ParameterSamplingHypothesis(0.05), 3, 2)
    model = _FakeModel(pi0)

    def boom():
        raise RuntimeError("solver failure")

    with pytest.raises(RuntimeError, match="solver failure"):
        evaluate_parameter_ensemble(
            samples, model, boom, _series([[0.0, -1.0]], [[2.0, 1.0]])
        )
    assert np.array_equal(model.current, pi0)


def _with_class(series, cls):
    return dataclasses.replace(series, sign_class=((cls, cls),))


def test_sensitivity_flags_sign_class_disagreement_and_infeasible_members(
    basis, pi0
) -> None:
    samples = sample_parameters(basis, ParameterSamplingHypothesis(0.05), 40, 2)
    samples = dataclasses.replace(
        samples,
        accepted=samples.accepted[:4],
        sample_indices=samples.sample_indices[:4],
    )
    model = _FakeModel(pi0)
    pos, neg, ind = SignClass.POSITIVE, SignClass.NEGATIVE, SignClass.INDETERMINATE
    infeasible = dataclasses.replace(
        _series([[np.nan, np.nan]], [[np.nan, np.nan]]),
        chebyshev_status=(BoundStatus.INFEASIBLE,),
    )
    queue = iter(
        [
            _with_class(_series([[1.0, 5.0]], [[3.0, 6.0]]), pos),
            _with_class(_series([[-1.0, 5.0]], [[3.0, 6.0]]), ind),
            infeasible,
            _with_class(_series([[1.0, 5.0]], [[3.0, 6.0]]), neg),
        ]
    )
    nominal = _with_class(_series([[1.0, 5.0]], [[3.0, 6.0]]), pos)
    report = evaluate_parameter_ensemble(samples, model, lambda: next(queue), nominal)
    assert report.n_enveloped == 4
    assert report.member_frames_infeasible == 1
    q0, q1 = report.quantities
    assert q0.max_bound_shift == pytest.approx(2.0)
    assert q0.union_width_over_nominal_max == pytest.approx(2.0)
    assert q1.max_bound_shift == pytest.approx(0.0)
    # Members disagree (positive, indeterminate, negative); the union interval
    # [-1, 3] of quantity 0 straddles zero, so its decided nominal sign is lost.
    assert q0.frames_with_sign_class_disagreement == 1
    assert q0.frames_losing_decided_sign == 1
    assert q1.frames_losing_decided_sign == 0
    assert report.to_dict()["quantities"]["x0"]["max_bound_shift"] == pytest.approx(2.0)


# --------------------------------------------- stacked regressor (fake model) --
class _LinearModel(_FakeModel):
    """``tau = Y pi`` with a fixed regressor, for the consistency check."""

    def __init__(self, pi0: np.ndarray, y: np.ndarray) -> None:
        super().__init__(pi0)
        self.y = y

    def torque_regressor(self, q, v, a) -> np.ndarray:
        return self.y


def _snapshot(y: np.ndarray, pi0: np.ndarray, unactuated: tuple[int, ...]):
    tau = y @ pi0
    return SimpleNamespace(
        q=np.zeros(4),
        v=np.zeros(4),
        a=np.ones(4),
        M=np.eye(4),
        h=tau - np.ones(4),
        topology=SimpleNamespace(unactuated_indices=unactuated),
    )


def test_stacked_regressor_selects_unactuated_rows_and_checks_the_dynamics(pi0) -> None:
    y = np.random.default_rng(9).normal(size=(4, pi0.size))
    model = _LinearModel(pi0, y)
    snaps = [_snapshot(y, pi0, (0, 1)), _snapshot(y, pi0, (0, 1))]
    stacked = stacked_unactuated_regressor(model, snaps)
    np.testing.assert_array_equal(stacked, np.vstack([y[:2], y[:2]]))
    wrong = _snapshot(y, pi0 * 1.1, (0, 1))
    with pytest.raises(ValueError, match="does not reproduce"):
        stacked_unactuated_regressor(model, [wrong])
    with pytest.raises(ValueError, match="at least one"):
        stacked_unactuated_regressor(model, [])
