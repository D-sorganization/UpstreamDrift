"""UCM, TNC and session-split falsification of impact variability (NSX-97).

Synthetic fixtures only.  No capture data is read or implied.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.swing_registration import (
    MultiSwingTrial,
    SwingEvent,
    TrialProvenance,
)
from src.shared.python.loop_nullspace.variability_directions import (
    compare_outcome_directions,
    predicted_outcome_covariance,
    principal_angles_deg,
)
from src.shared.python.loop_nullspace.variability_registration import (
    registration_residual_report,
)
from src.shared.python.loop_nullspace.variability_report import (
    NON_CLAIMS,
    FalsificationOptions,
    falsify_variability,
)
from src.shared.python.loop_nullspace.variability_split import (
    LeakageError,
    SessionSplit,
    SwingSample,
    assert_fit_not_leaked,
    fit_noise_amplitudes,
)
from src.shared.python.loop_nullspace.variability_tnc import (
    DEFAULT_SEED,
    linear_outcome_fn,
    permute_across_trials,
    tnc_decompose,
)
from src.shared.python.loop_nullspace.variability_ucm import ucm_decompose

pytestmark = pytest.mark.unit

N_STATE = 6
N_OUT = 2


def _jacobian() -> np.ndarray:
    return np.random.default_rng(7).normal(size=(N_OUT, N_STATE))


def _kernel(jac: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(jac)
    return vt[N_OUT:].T  # (n, n - rank)


def _planted(n: int, ucm_sd: float, ort_sd: float, seed: int = 1) -> np.ndarray:
    jac = _jacobian()
    ker = _kernel(jac)
    _, _, vt = np.linalg.svd(jac)
    row = vt[:N_OUT].T
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, ker.shape[1])) * ucm_sd @ ker.T
        + rng.normal(size=(n, N_OUT)) * ort_sd @ row.T
    )


# ---------------------------------------------------------------- UCM


def test_variance_planted_in_kernel_gives_zero_ort() -> None:
    res = ucm_decompose(_planted(400, 1.0, 0.0), _jacobian())
    assert res.v_ort < 1e-20 * res.v_ucm
    assert res.v_ort_per_dim < 1e-20
    assert res.delta_v == pytest.approx(N_STATE / (N_STATE - N_OUT), rel=1e-9)
    assert res.delta_v_z > 5.0


def test_isotropic_variance_gives_no_synergy() -> None:
    res = ucm_decompose(_planted(20000, 1.0, 1.0, seed=3), _jacobian())
    assert abs(res.delta_v) < 0.08
    # dVz is 0 at dV = 0 only for d_ucm = d_ort; in general it is
    # 0.5 ln(d_ucm / d_ort) (the standard transform's asymmetric bounds).
    assert res.delta_v_z == pytest.approx(0.5 * np.log(res.n_ucm / res.n_ort), abs=0.05)


def test_variance_only_in_row_space_gives_negative_synergy() -> None:
    res = ucm_decompose(_planted(400, 0.0, 1.0), _jacobian())
    assert res.v_ucm < 1e-20 * res.v_ort
    assert res.delta_v == pytest.approx(-N_STATE / N_OUT, rel=1e-9)


def test_ucm_per_dimension_normalisation_and_total() -> None:
    states = _planted(300, 1.0, 0.5, seed=5)
    res = ucm_decompose(states, _jacobian())
    assert (res.n_ucm, res.n_ort) == (N_STATE - N_OUT, N_OUT)
    assert res.v_ucm_per_dim == pytest.approx(res.v_ucm / res.n_ucm)
    assert res.v_ort_per_dim == pytest.approx(res.v_ort / res.n_ort)
    total = float(np.trace(np.cov(states, rowvar=False)))
    assert res.v_ucm + res.v_ort == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize(
    "states, jac",
    [
        (np.zeros((1, N_STATE)), _jacobian()),
        (np.zeros((10, N_STATE + 1)), _jacobian()),
        (np.zeros((10, N_STATE)), np.eye(N_STATE)),
        (np.full((10, N_STATE), np.nan), _jacobian()),
    ],
)
def test_ucm_rejects_bad_input(states: np.ndarray, jac: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ucm_decompose(states, jac)


# ---------------------------------------------------------------- TNC


def _compensating(n: int, seed: int = 2) -> np.ndarray:
    return _planted(n, 1.0, 0.05, seed=seed)


def test_permuting_across_trials_removes_covariation() -> None:
    states = _compensating(500)
    jac = _jacobian()
    perm = permute_across_trials(states, np.random.default_rng(0))
    assert np.allclose(np.sort(perm, axis=0), np.sort(states, axis=0))
    corr = np.corrcoef(perm, rowvar=False)
    assert np.max(np.abs(corr - np.eye(N_STATE))) < 0.2
    original = ucm_decompose(states, jac).delta_v
    assert original > 1.4  # upper bound is n / d_ucm = 1.5
    assert ucm_decompose(perm, jac).delta_v < original - 0.8


def test_tnc_covariation_is_beneficial_for_compensating_data() -> None:
    jac = _jacobian()
    res = tnc_decompose(
        _compensating(200),
        linear_outcome_fn(jac),
        np.zeros(N_OUT),
        n_permutations=100,
        n_rotations=100,
    )
    assert res.cost_measured < res.cost_permuted
    assert res.covariation_cost > 0.5 * res.cost_permuted
    assert res.p_value_covariation < 0.05
    assert res.covariation_cost_rotation > 0.0


def test_tnc_independent_noise_has_small_covariation() -> None:
    states = np.random.default_rng(4).normal(size=(400, N_STATE))
    res = tnc_decompose(
        states,
        linear_outcome_fn(_jacobian()),
        np.zeros(N_OUT),
        n_permutations=100,
        n_rotations=100,
    )
    assert abs(res.covariation_cost) < 0.1 * res.cost_permuted
    assert res.p_value_covariation > 0.05


def test_tnc_accounting_identity_and_tolerance_cost() -> None:
    jac = _jacobian()
    states = _compensating(200) + 0.7  # offset: distribution is off-target
    res = tnc_decompose(
        states,
        linear_outcome_fn(jac),
        np.zeros(N_OUT),
        n_permutations=50,
        n_rotations=20,
    )
    assert res.tolerance_cost > 0.0
    assert res.noise_cost >= -1e-9
    total = res.tolerance_cost + res.noise_cost + res.cost_floor
    assert res.cost_measured == pytest.approx(total - res.covariation_cost, rel=1e-9)
    assert res.identity_residual < 1e-9


def test_tnc_seed_is_declared_and_reproducible() -> None:
    assert DEFAULT_SEED == 20261016
    args = (_compensating(80), linear_outcome_fn(_jacobian()), np.zeros(N_OUT))
    a = tnc_decompose(*args, n_permutations=30, n_rotations=30)
    b = tnc_decompose(*args, n_permutations=30, n_rotations=30)
    c = tnc_decompose(*args, n_permutations=30, n_rotations=30, seed=1)
    assert a.seed == DEFAULT_SEED and a == b
    assert a.cost_permuted != c.cost_permuted


def test_tnc_rejects_bad_input() -> None:
    fn = linear_outcome_fn(_jacobian())
    with pytest.raises(ValueError):
        tnc_decompose(np.zeros((1, N_STATE)), fn, np.zeros(N_OUT))
    with pytest.raises(ValueError):
        tnc_decompose(np.zeros((5, N_STATE)), fn, np.zeros(N_OUT), n_permutations=0)


# ---------------------------------------------------------------- directions


def test_principal_angles_identical_and_orthogonal() -> None:
    e = np.eye(4)
    assert principal_angles_deg(e[:, :2], e[:, :2]) == pytest.approx([0.0, 0.0])
    assert principal_angles_deg(e[:, :2], e[:, 2:]) == pytest.approx([90.0, 90.0])


def test_direction_comparison_detects_alignment_with_permutation_null() -> None:
    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    sd = np.array([5.0, 1.0, 0.3, 0.3, 0.3])
    outcomes = (rng.normal(size=(300, 5)) * sd) @ basis.T
    pred = (basis * sd**2) @ basis.T
    good = compare_outcome_directions(pred, outcomes, k=1, n_permutations=200)
    assert good.mean_angle_deg < 5.0
    assert good.p_value < 0.05
    wrong = (basis[:, ::-1] * sd**2) @ basis[:, ::-1].T
    bad = compare_outcome_directions(wrong, outcomes, k=1, n_permutations=200)
    assert bad.mean_angle_deg > 45.0
    assert bad.p_value > 0.05
    assert good.seed == DEFAULT_SEED


def test_predicted_outcome_covariance_is_jacobian_congruence() -> None:
    jac = _jacobian()
    cov = np.diag(np.arange(1.0, N_STATE + 1))
    assert np.allclose(predicted_outcome_covariance(jac, cov), jac @ cov @ jac.T)
    with pytest.raises(ValueError):
        predicted_outcome_covariance(jac, np.eye(N_STATE + 1))


# ---------------------------------------------------------------- split


def _samples(sessions: dict[str, int], seed: int = 0) -> list[SwingSample]:
    rng = np.random.default_rng(seed)
    out: list[SwingSample] = []
    for sid, count in sessions.items():
        offset = rng.normal(size=N_STATE) * 10.0  # session mean must not count
        for k in range(count):
            out.append(
                SwingSample(f"{sid}-{k}", sid, offset + rng.normal(size=N_STATE) * 0.5)
            )
    return out


def test_split_is_session_level_disjoint_and_seeded() -> None:
    sessions = [f"s{i}" for i in range(8)]
    a = SessionSplit.from_sessions(sessions, test_fraction=0.25, seed=3)
    b = SessionSplit.from_sessions(sessions, test_fraction=0.25, seed=3)
    assert a == b
    assert not (a.train_sessions & a.test_sessions)
    assert a.train_sessions | a.test_sessions == frozenset(sessions)
    assert len(a.test_sessions) == 2
    with pytest.raises(ValueError):
        SessionSplit.from_sessions(["only"], test_fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        SessionSplit(frozenset({"a"}), frozenset({"a"}), seed=0)


def test_noise_fit_uses_only_train_and_removes_session_means() -> None:
    swings = _samples({"a": 30, "b": 30, "c": 30})
    split = SessionSplit(frozenset({"a", "b"}), frozenset({"c"}), seed=0)
    train = [s for s in swings if s.session_id in split.train_sessions]
    fit = fit_noise_amplitudes(train, split)
    assert set(fit.swing_ids) == {s.swing_id for s in train}
    assert fit.train_sessions == split.train_sessions
    assert np.all(np.abs(fit.amplitudes - 0.5) < 0.2)  # not ~10 session offsets


def test_leakage_of_test_swings_into_the_fit_is_rejected() -> None:
    swings = _samples({"a": 20, "b": 20, "c": 20})
    split = SessionSplit(frozenset({"a", "b"}), frozenset({"c"}), seed=0)
    with pytest.raises(LeakageError, match="test"):
        fit_noise_amplitudes(swings, split)  # includes session c
    train = [s for s in swings if s.session_id != "c"]
    fit = fit_noise_amplitudes(train, split)
    tampered = type(fit)(
        amplitudes=fit.amplitudes,
        covariance=fit.covariance,
        train_sessions=fit.train_sessions,
        swing_ids=fit.swing_ids + ("c-0",),
    )
    with pytest.raises(LeakageError):
        assert_fit_not_leaked(tampered, split, ["c-0", "c-1"])
    assert_fit_not_leaked(fit, split, ["c-0", "c-1"])
    stray = SwingSample("z-0", "zz", np.zeros(N_STATE))
    with pytest.raises(LeakageError):
        fit_noise_amplitudes([*train, stray], split)


# ---------------------------------------------------------------- registration


def _trial(idx: int, scale: float) -> MultiSwingTrial:
    times = np.linspace(0.0, 1.5 * scale, 301)
    ev = {
        SwingEvent.ADDRESS: 0.0,
        SwingEvent.TAKEAWAY: 0.2 * scale,
        SwingEvent.TOP: 0.8 * scale,
        SwingEvent.IMPACT: 1.1 * scale,
        SwingEvent.FINISH: 1.5 * scale,
    }
    phase = np.interp(
        times,
        [ev[e] for e in ev],
        [0.0, 0.2, 0.6, 0.8, 1.0],
    )
    return MultiSwingTrial(
        trial_id=f"t{idx}",
        times=times,
        trajectories={"inputs": np.sin(2 * np.pi * phase)[:, None]},
        events=ev,
        provenance=TrialProvenance(subject_id="anon", session_id="s"),
    )


def test_registration_report_keeps_phase_and_physical_residuals() -> None:
    trials = [_trial(i, s) for i, s in enumerate([0.9, 1.0, 1.1, 1.05, 0.95])]
    rep = registration_residual_report(trials, "inputs", n_bins=40)
    assert rep.phase_residuals.shape[:2] == (5, 40)
    assert rep.physical_residuals.shape[:2] == (5, 40)
    assert rep.phase_rms < 0.05
    assert rep.physical_rms > 5 * rep.phase_rms
    assert rep.event_durations.shape == (5, 4)
    assert rep.event_duration_sd.shape == (4,)
    assert rep.variance_raw >= 0.0 and rep.variance_registered >= 0.0


# ---------------------------------------------------------------- report


def _scenario() -> tuple[list[SwingSample], SessionSplit, np.ndarray]:
    jac = _jacobian()
    swings: list[SwingSample] = []
    for sid in ["a", "b", "c", "d"]:
        for k, x in enumerate(_planted(40, 1.0, 0.05, seed=ord(sid))):
            swings.append(SwingSample(f"{sid}-{k}", sid, x))
    split = SessionSplit(frozenset({"a", "b", "c"}), frozenset({"d"}), seed=0)
    return swings, split, jac


def test_falsify_variability_end_to_end_reports_everything() -> None:
    swings, split, jac = _scenario()
    res = falsify_variability(
        swings,
        split,
        jac,
        np.zeros(N_OUT),
        options=FalsificationOptions(n_permutations=60, n_rotations=30),
    )
    assert res.seed == DEFAULT_SEED
    assert res.ucm.delta_v > 1.4
    assert res.tnc.covariation_cost > 0.0
    assert res.directions.k >= 1
    assert res.n_test_swings == 40
    assert res.noise_fit.train_sessions == split.train_sessions
    assert res.outcome_source == "linearised_from_state"
    assert res.records["delta_v"].units == "1"
    assert res.records["delta_v"].validation_status.name == "UNVALIDATED"
    assert any("neural intent" in c for c in res.non_claims)
    assert any("co-contraction" in c for c in res.non_claims)
    assert res.non_claims == NON_CLAIMS


def test_falsify_variability_needs_test_swings() -> None:
    swings, split, jac = _scenario()
    only_train = [s for s in swings if s.session_id in split.train_sessions]
    with pytest.raises(ValueError, match="test"):
        falsify_variability(only_train, split, jac, np.zeros(N_OUT))


# ---------------------------------------------------------------- receipt / CLI


def test_receipt_records_hashes_without_paths(tmp_path) -> None:
    from src.shared.python.loop_nullspace.variability_cli import hash_inputs

    f1 = tmp_path / "private_subject_42.npz"
    f1.write_bytes(b"abc")
    receipt = hash_inputs([f1])
    assert receipt.files[0].sha256 == hashlib.sha256(b"abc").hexdigest()
    assert receipt.files[0].size_bytes == 3
    blob = json.dumps(receipt.to_dict())
    assert "private_subject_42" not in blob and str(tmp_path) not in blob
    with pytest.raises(FileNotFoundError):
        hash_inputs([tmp_path / "missing.npz"])


def test_cli_produces_report_and_receipt_outside_repo(tmp_path) -> None:
    from src.shared.python.loop_nullspace.variability_cli import main

    swings, _, jac = _scenario()
    npz = tmp_path / "cap.npz"
    np.savez(
        npz,
        states=np.stack([s.state for s in swings]),
        swing_ids=np.array([s.swing_id for s in swings]),
        session_ids=np.array([s.session_id for s in swings]),
        outcome_jacobian=jac,
        outcome_target=np.zeros(N_OUT),
    )
    out = tmp_path / "report.json"
    rc = main(
        [
            "--input", str(npz), "--output", str(out), "--receipt",
            str(tmp_path / "receipt.json"), "--test-fraction", "0.25",
            "--permutations", "30", "--rotations", "10",
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads(out.read_text())
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert report["seed"] == DEFAULT_SEED
    assert report["non_claims"]
    assert receipt["files"][0]["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(report)
