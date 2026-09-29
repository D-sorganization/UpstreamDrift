"""Unit tests for ``validate_against_simscape`` (#3999 / #030).

Covers the contract laid out in
``docs/issues/backlog/030_round_trip_validation_against_simscape.md``:
sim_fn injection, the extrapolation flag, threshold configurability,
the no-MATLAB error path, and the public report contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.surrogate import (
    FitResult,
    NormalizationStats,
    SurrogateConfig,
    SwingSurrogate,
    TrainConfig,
    TrainedSurrogate,
    ValidationReport,
    validate_against_simscape,
)
from src.shared.python.motion_matching.surrogate.train import TrainingCurves

from ._fixtures import make_target

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSimOutput:
    """Minimal stand-in for ``SimscapeOutput`` (only ``r_clubhead`` is read)."""

    r_clubhead: np.ndarray


def _tiny_surrogate(n_joints: int = 2, seq_len: int = 16) -> SwingSurrogate:
    cfg = SurrogateConfig(
        n_joints=n_joints,
        coeffs_per_joint=7,
        seq_len=seq_len,
        hidden_dim=8,
        n_layers=2,
        time_embed_dim=8,
        encoder_layers=2,
    )
    return SwingSurrogate(cfg)


def _make_trained_bundle(model: SwingSurrogate) -> TrainedSurrogate:
    coeff_dim = model.cfg.coeff_dim
    stats = NormalizationStats(
        coeffs_mean=np.zeros(coeff_dim, dtype=np.float32),
        coeffs_std=np.ones(coeff_dim, dtype=np.float32),
        butt_mean=np.zeros(3, dtype=np.float32),
        butt_std=np.ones(3, dtype=np.float32),
        clubhead_mean=np.zeros(3, dtype=np.float32),
        clubhead_std=np.ones(3, dtype=np.float32),
    )
    return TrainedSurrogate(
        model=model,
        config=model.cfg,
        train_config=TrainConfig(n_epochs=1),
        norm_stats=stats,
        curves=TrainingCurves(),
        joint_names=[f"j{i}" for i in range(model.cfg.n_joints)],
        seq_len=model.cfg.seq_len,
        final_val_loss=float("nan"),
    )


def _fit_result_from(
    surrogate: SwingSurrogate,
    coeffs_np: np.ndarray,
    *,
    fake_pred_to_target: object | None = None,
) -> FitResult:
    """Build a ``FitResult``; if ``fake_pred_to_target`` is a ``ClubTarget``,
    we synthesize a :class:`ClubTrajectory` that matches that target so the
    surrogate-vs-target RMSE is essentially zero (lets us exercise the
    extrapolation flag without needing a fully trained model).
    """
    from src.shared.python.motion_matching.surrogate.model import ClubTrajectory

    surrogate.eval()
    if fake_pred_to_target is not None:
        tgt = fake_pred_to_target
        # Add a small (1 mm) offset so surrogate RMSE is finite-positive
        # rather than exactly zero (avoids inf ratios).
        small = np.full_like(np.asarray(tgt.clubhead, dtype=np.float32), 0.001)
        pred = ClubTrajectory(
            butt=torch.from_numpy(np.asarray(tgt.butt, dtype=np.float32))
            .unsqueeze(0)
            .clone(),
            clubhead=torch.from_numpy(
                np.asarray(tgt.clubhead, dtype=np.float32) + small
            )
            .unsqueeze(0)
            .clone(),
            club_quat=torch.from_numpy(np.asarray(tgt.club_quat, dtype=np.float32))
            .unsqueeze(0)
            .clone(),
            joint_q=torch.zeros(1, surrogate.cfg.seq_len, surrogate.cfg.n_joints),
        )
    else:
        with torch.no_grad():
            pred = surrogate(torch.from_numpy(coeffs_np).float().unsqueeze(0))
    return FitResult(
        coefficients=coeffs_np.astype(np.float32),
        final_loss=0.0,
        history={"loss": np.zeros((1, 1), dtype=np.float32)},
        all_starts=[coeffs_np.copy()],
        surrogate_pred=pred,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validation_uses_provided_sim_fn() -> None:
    """The injected ``sim_fn`` is called exactly once with the fit coefficients."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs)

    calls: list[np.ndarray] = []

    def fake_sim(c: np.ndarray) -> _FakeSimOutput:
        calls.append(c.copy())
        return _FakeSimOutput(r_clubhead=np.asarray(target.clubhead))

    report = validate_against_simscape(result, target, bundle, sim_fn=fake_sim)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], coeffs.astype(np.float64))
    assert isinstance(report, ValidationReport)


@pytest.mark.unit
def test_extrapolation_flag_when_simscape_far_from_surrogate() -> None:
    """If Simscape diverges from the target, the extrapolation flag fires."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs, fake_pred_to_target=target)

    # Force a large simscape error: shift the clubhead by 1 m on every axis.
    far_head = np.asarray(target.clubhead) + 1.0

    def fake_sim(_c: np.ndarray) -> _FakeSimOutput:
        return _FakeSimOutput(r_clubhead=far_head)

    report = validate_against_simscape(
        result, target, bundle, sim_fn=fake_sim, threshold=2.0
    )
    assert report.simscape_rmse_m > report.surrogate_rmse_m
    assert report.is_extrapolation is True
    assert report.extrapolation_factor > 2.0


@pytest.mark.unit
def test_no_extrapolation_when_close() -> None:
    """If Simscape matches the surrogate's prediction, no extrap flag."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs)

    # Use the surrogate's own prediction as the simscape output -> ratio == 1.
    surrogate_head = result.surrogate_pred.clubhead.detach().cpu().numpy()[0]

    def fake_sim(_c: np.ndarray) -> _FakeSimOutput:
        return _FakeSimOutput(r_clubhead=surrogate_head.astype(np.float64))

    report = validate_against_simscape(
        result, target, bundle, sim_fn=fake_sim, threshold=2.0
    )
    assert report.is_extrapolation is False
    assert report.extrapolation_factor == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_threshold_configurable() -> None:
    """The threshold parameter changes which factors trigger the flag."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs, fake_pred_to_target=target)

    # Pick a moderate offset that yields factor ~1.5 in clubhead RMSE.
    small_offset = np.asarray(target.clubhead) + 0.05

    def fake_sim(_c: np.ndarray) -> _FakeSimOutput:
        return _FakeSimOutput(r_clubhead=small_offset)

    strict = validate_against_simscape(
        result, target, bundle, sim_fn=fake_sim, threshold=1.0
    )
    lax = validate_against_simscape(
        result, target, bundle, sim_fn=fake_sim, threshold=100.0
    )
    # Same factor, different flags.
    assert strict.extrapolation_factor == pytest.approx(
        lax.extrapolation_factor, rel=1e-9
    )
    assert strict.is_extrapolation is True
    assert lax.is_extrapolation is False


@pytest.mark.unit
def test_raises_when_no_sim_fn_and_no_matlab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``sim_fn`` and without MATLAB we get a helpful error."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs)

    monkeypatch.setenv("UD_SIMSCAPE_FORCE_NO_MATLAB", "1")
    # Also force the engine_pool's predicate to report unavailable.
    from src.engines.simscape import _engine_pool

    monkeypatch.setattr(_engine_pool, "is_matlab_available", lambda: False)

    from src.engines.simscape._errors import SimscapeNotInstalledError

    with pytest.raises(SimscapeNotInstalledError, match="MATLAB"):
        validate_against_simscape(result, target, bundle)


@pytest.mark.unit
def test_validation_report_contract() -> None:
    """Report fields are present, finite, and frozen."""
    surrogate = _tiny_surrogate()
    bundle = _make_trained_bundle(surrogate)
    target = make_target(n=surrogate.cfg.seq_len)
    coeffs = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    result = _fit_result_from(surrogate, coeffs)

    sim_out = _FakeSimOutput(r_clubhead=np.asarray(target.clubhead) + 0.1)

    report = validate_against_simscape(
        result, target, bundle, sim_fn=lambda _c: sim_out
    )

    assert isinstance(report, ValidationReport)
    assert np.isfinite(report.surrogate_rmse_m)
    assert np.isfinite(report.simscape_rmse_m)
    assert report.surrogate_rmse_m >= 0.0
    assert report.simscape_rmse_m >= 0.0
    assert np.isfinite(report.extrapolation_factor)
    assert report.simscape_out is sim_out
    # Frozen dataclass: assignment must raise FrozenInstanceError.
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        report.surrogate_rmse_m = 0.0  # type: ignore[misc]
