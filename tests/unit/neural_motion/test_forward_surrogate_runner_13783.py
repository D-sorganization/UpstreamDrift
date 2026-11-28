"""ML-WS-6 (#13783): Forward residual surrogate training, held-out eval, and receipt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.neural_motion.surrogates import PhysicsStructuredSurrogate
from src.shared.python.neural_motion.surrogates.forward_surrogate_runner import (
    FORWARD_SURROGATE_RECEIPT_SCHEMA,
    ForwardSurrogateRunResult,
    SurrogateTrainingConfig,
    SurrogateTrainingReceipt,
    compare_against_mean_baseline,
    load_surrogate_receipt,
    serialize_receipt_json,
    train_and_evaluate_forward_surrogate,
)

pytestmark = pytest.mark.unit


def _make_episodes(n_eps: int, prefix: str, seed: int = 13783) -> tuple:
    """Generate deterministic synthetic 2-DOF native-corpus episode trajectories."""
    rng, t = np.random.default_rng(seed), np.linspace(0.0, 1.0, 20)
    xs, ys, ids = [], [], []
    for idx in range(n_eps):
        a, f, p = rng.uniform(0.6, 1.4), rng.uniform(1.5, 3.0), rng.uniform(-0.8, 0.8)
        q0, q1 = a * np.sin(f * t + p), 0.5 * a * np.cos(f * t - p)
        v0, v1 = a * f * np.cos(f * t + p), -0.5 * a * f * np.sin(f * t - p)
        r0 = -1.8 * np.sin(q0) - 0.35 * v0 + 0.25 * q0 * q1 + 0.15
        r1 = -0.9 * np.sin(q1) - 0.25 * v1 + 0.20 * q0 * v1 - 0.10
        xs.append(np.column_stack([q0, q1, v0, v1]))
        ys.append(np.column_stack([r0, r1]) + rng.normal(0.0, 0.01, (20, 2)))
        ids.extend([f"{prefix}_ep_{idx:03d}"] * 20)
    return np.vstack(xs), np.vstack(ys), tuple(ids)


def _run(
    tr: tuple, va: tuple, *, ep: int = 40, seed: int = 42, out: Path | None = None
) -> ForwardSurrogateRunResult:
    """Run train_and_evaluate_forward_surrogate on train and val episode tuples."""
    cfg = SurrogateTrainingConfig(epochs=ep, seed=seed)
    return train_and_evaluate_forward_surrogate(
        tr[0],
        tr[1],
        episode_ids=tr[2],
        val_features=va[0],
        val_targets=va[1],
        val_episode_ids=va[2],
        config=cfg,
        receipt_path=out,
    )


def _receipt(
    *, proposal_only: bool = True, claims_match: bool = False
) -> SurrogateTrainingReceipt:
    """Construct a minimal receipt with explicit honesty-flag overrides."""
    return SurrogateTrainingReceipt(
        model_id="driven_double_pendulum",
        seed=0,
        epochs=2,
        train_sample_count=10,
        val_sample_count=4,
        train_episode_ids=("ep_0",),
        val_episode_ids=("ep_1",),
        train_loss_curve=(1.0, 0.5),
        val_loss_curve=(1.1, 0.6),
        held_out_rmse=0.6,
        mean_baseline_rmse=1.2,
        beats_mean_baseline=True,
        checkpoint_sha256="a" * 64,
        proposal_only=proposal_only,
        claims_match=claims_match,
    )


def test_training_and_held_out_eval_beats_mean_baseline(tmp_path: Path) -> None:
    """Training tracks monotonic curves, beats mean baseline, and produces a valid fit."""
    tr, va = _make_episodes(6, "train", 13783), _make_episodes(3, "val", 13784)
    out = tmp_path / "surrogate_receipt.json"
    res = _run(tr, va, ep=60, seed=42, out=out)
    r = res.receipt
    tr_c, va_c = r.train_loss_curve, r.val_loss_curve
    assert r.schema == FORWARD_SURROGATE_RECEIPT_SCHEMA and len(tr_c) == 60
    assert tr_c[-1] < tr_c[0] and va_c[-1] < va_c[0]
    assert all(b <= a + 1e-12 for a, b in zip(tr_c, tr_c[1:], strict=False))
    assert r.beats_mean_baseline is True and r.held_out_rmse < r.mean_baseline_rmse
    assert r.held_out_rmse == pytest.approx(va_c[-1])
    assert len(r.checkpoint_sha256) == 64
    assert r.proposal_only is True and r.claims_match is False and out.is_file()
    np.testing.assert_allclose(res.fit.predict(va[0]), res.val_predictions)
    held, base, beats = compare_against_mean_baseline(tr[1], va[1], res.val_predictions)
    assert beats is True and held == pytest.approx(r.held_out_rmse)
    assert base == pytest.approx(r.mean_baseline_rmse)
    fit_2d = _run((tr[0][:, :2], tr[1], tr[2]), (va[0][:, :2], va[1], va[2]), ep=25).fit
    sur = PhysicsStructuredSurrogate(n_dof=2, n_coeffs=7, residual=fit_2d)
    traj = sur.forward_trajectory(np.zeros(14), np.linspace(0.0, 0.3, 12))
    assert traj.shape == (12, 2)


def test_episode_id_auto_split_keeps_episodes_disjoint() -> None:
    """When val_features is omitted, splitting by episode_ids prevents leakage."""
    tr, va = _make_episodes(6, "train", 13783), _make_episodes(3, "val", 13784)
    cfg = SurrogateTrainingConfig(epochs=40, seed=11)
    res = train_and_evaluate_forward_surrogate(
        np.vstack([tr[0], va[0]]),
        np.vstack([tr[1], va[1]]),
        episode_ids=tr[2] + va[2],
        config=cfg,
    )
    assert set(res.receipt.train_episode_ids).isdisjoint(
        set(res.receipt.val_episode_ids)
    )
    assert res.receipt.beats_mean_baseline is True


def test_proposal_only_honesty_invariants_reject_overclaims() -> None:
    """Receipts refuse proposal_only=False or claims_match=True."""
    valid = _receipt()
    assert valid.proposal_only is True and valid.claims_match is False
    with pytest.raises(ValueError, match="proposal_only"):
        _receipt(proposal_only=False)
    with pytest.raises(ValueError, match="claims_match"):
        _receipt(claims_match=True)


def test_fail_closed_guards_for_bad_inputs_and_overlap(tmp_path: Path) -> None:
    """Empty datasets, NaN/Inf inputs, overlapping episode IDs, and tampered JSON fail closed."""
    tr, va = _make_episodes(3, "train", 1), _make_episodes(2, "val", 2)
    fn = train_and_evaluate_forward_surrogate
    with pytest.raises(ValueError, match="empty"):
        fn(np.empty((0, 4)), np.empty((0, 2)), val_features=va[0], val_targets=va[1])
    bad_x, bad_y = tr[0].copy(), va[1].copy()
    bad_x[0, 0], bad_y[0, 1] = np.nan, np.inf
    with pytest.raises(ValueError, match="finite"):
        fn(bad_x, tr[1], val_features=va[0], val_targets=va[1])
    with pytest.raises(ValueError, match="finite"):
        fn(tr[0], tr[1], val_features=va[0], val_targets=bad_y)
    with pytest.raises(ValueError, match="overlap"):
        _run(tr, (va[0], va[1], (tr[2][0],) + va[2][1:]), ep=5)
    with pytest.raises(ValueError):
        SurrogateTrainingConfig(model_id="")
    with pytest.raises(ValueError):
        SurrogateTrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        SurrogateTrainingConfig(learning_rate=0.0)
    tamper = tmp_path / "tampered.json"
    _run(tr, va, ep=10, seed=3, out=tamper)
    payload = json.loads(tamper.read_text(encoding="utf-8"))
    payload["claims_match"] = True
    tamper.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_surrogate_receipt(tamper)


def test_deterministic_serialization_and_checkpoint_sha256_stability(
    tmp_path: Path,
) -> None:
    """Identical seeds produce byte-identical JSON receipts and identical checkpoint_sha256."""
    tr, va = _make_episodes(6, "train", 13783), _make_episodes(3, "val", 13784)
    a, b, other = (
        _run(tr, va, ep=35, seed=99),
        _run(tr, va, ep=35, seed=99),
        _run(tr, va, ep=35, seed=100),
    )
    assert a.receipt.checkpoint_sha256 == b.receipt.checkpoint_sha256
    assert serialize_receipt_json(a.receipt) == serialize_receipt_json(b.receipt)
    assert a.receipt.receipt_sha256 != other.receipt.receipt_sha256
    out = tmp_path / "receipt.json"
    a.receipt.write_json(out)
    assert load_surrogate_receipt(out) == a.receipt
