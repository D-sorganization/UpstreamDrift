"""Unit tests for ``fit_swing_hybrid`` (#4000 / #031).

The surrogate inversion is exercised against a tiny ``SwingSurrogate``;
the polish stage is dependency-injected via the ``polish_fn`` parameter
so these tests do not require MATLAB.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.hybrid import (
    HybridFitResult,
    HybridOptions,
    fit_swing_hybrid,
)
from src.shared.python.motion_matching.surrogate import (
    InvertOptions,
    SurrogateConfig,
    SwingSurrogate,
)

from ._fixtures import make_provenance


def _tiny_surrogate(n_joints: int = 2, seq_len: int = 16) -> SwingSurrogate:
    cfg = SurrogateConfig(
        n_joints=n_joints,
        coeffs_per_joint=7,
        seq_len=seq_len,
        hidden_dim=16,
        n_layers=2,
        time_embed_dim=8,
        encoder_layers=2,
    )
    return SwingSurrogate(cfg)


def _target_for_surrogate(surrogate: SwingSurrogate) -> ClubTarget:
    """Forward the surrogate at zero coefficients to get a feasible target."""
    coeff_dim = surrogate.cfg.coeff_dim
    seq_len = surrogate.cfg.seq_len
    surrogate.eval()
    with torch.no_grad():
        coeffs = torch.zeros(1, coeff_dim)
        pred = surrogate(coeffs)
    butt = np.clip(pred.butt[0].cpu().numpy().astype(np.float64), -1.5, 1.5)
    head = np.clip(pred.clubhead[0].cpu().numpy().astype(np.float64), -1.5, 1.5)
    quat = pred.club_quat[0].cpu().numpy().astype(np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return ClubTarget(
        time=np.linspace(0.0, 0.3, seq_len),
        butt=butt,
        clubhead=head,
        club_quat=quat,
        impact_idx=seq_len // 2,
        source=make_provenance(),
    )


def _fast_invert_opts() -> InvertOptions:
    return InvertOptions(n_starts=2, n_iters_per_start=3, lr=1e-2, seed=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hybrid_options_validation() -> None:
    """``HybridOptions`` rejects bad hyperparameters."""
    with pytest.raises(ValueError, match="polish_solver"):
        HybridOptions(polish_solver="weird")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="invert"):
        HybridOptions(invert={"not": "an_options_obj"})  # type: ignore[arg-type]
    # Defaults construct fine.
    opts = HybridOptions()
    assert opts.polish_solver == "fmincon"
    assert opts.skip_polish_tol == float("-inf")


@pytest.mark.unit
def test_hybrid_requires_polish_fn_when_polishing() -> None:
    """``fit_swing_hybrid`` raises if polish_fn is missing for fmincon polish."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)
    opts = HybridOptions(invert=_fast_invert_opts(), polish_solver="fmincon")
    with pytest.raises(ValueError, match="polish_fn is required"):
        fit_swing_hybrid(target, surrogate, options=opts, polish_fn=None)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hybrid_returns_combined_result() -> None:
    """The result records both phases and the combined solver label."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)

    received: dict[str, np.ndarray] = {}

    def polish_fn(_t: ClubTarget, theta_warm: np.ndarray) -> dict[str, object]:
        received["theta_warm"] = theta_warm.copy()
        # Pretend the polish made it 10x better in RMSE.
        return {
            "coefficients": theta_warm * 0.5,
            "final_rmse_m": 1.0e-3,
            "solver": "fmincon",
        }

    opts = HybridOptions(invert=_fast_invert_opts(), polish_solver="fmincon")
    result = fit_swing_hybrid(target, surrogate, options=opts, polish_fn=polish_fn)

    assert isinstance(result, HybridFitResult)
    assert result.method == "surrogate+fmincon"
    assert result.surrogate_phase is not None
    assert result.polish_phase is not None
    assert result.polish_phase["solver"] == "fmincon"
    assert result.final_loss == pytest.approx(1.0e-3)
    assert result.theta_optimal.shape == (surrogate.cfg.coeff_dim,)


@pytest.mark.unit
def test_hybrid_calls_polish_after_surrogate_inversion() -> None:
    """The polish callable is invoked with the surrogate's warm-start theta."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)

    captured: dict[str, np.ndarray] = {}

    def polish_fn(_t: ClubTarget, theta_warm: np.ndarray) -> dict[str, object]:
        captured["theta_warm"] = theta_warm.copy()
        return {"coefficients": theta_warm, "final_rmse_m": 0.0}

    opts = HybridOptions(invert=_fast_invert_opts(), polish_solver="fmincon")
    result = fit_swing_hybrid(target, surrogate, options=opts, polish_fn=polish_fn)

    # The captured theta_warm must equal what the surrogate phase reported.
    assert "theta_warm" in captured
    np.testing.assert_allclose(
        captured["theta_warm"],
        np.asarray(result.surrogate_phase.theta_optimal, dtype=np.float64),
        atol=1e-12,
    )


@pytest.mark.unit
def test_hybrid_skips_polish_when_under_tolerance() -> None:
    """When the surrogate's loss is already below skip_polish_tol, no polish."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)
    polish_called = {"n": 0}

    def polish_fn(_t: ClubTarget, theta_warm: np.ndarray) -> dict[str, object]:
        polish_called["n"] += 1
        return {"coefficients": theta_warm, "final_rmse_m": 0.0}

    opts = HybridOptions(
        invert=_fast_invert_opts(),
        polish_solver="fmincon",
        skip_polish_tol=1.0e9,  # ridiculously high -> always skip
    )
    result = fit_swing_hybrid(target, surrogate, options=opts, polish_fn=polish_fn)

    assert polish_called["n"] == 0
    assert result.method == "surrogate"
    assert result.polish_phase is None


@pytest.mark.unit
def test_hybrid_polish_solver_none_returns_warm_start() -> None:
    """``polish_solver='none'`` returns the surrogate warm start unchanged."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)
    opts = HybridOptions(invert=_fast_invert_opts(), polish_solver="none")
    result = fit_swing_hybrid(target, surrogate, options=opts, polish_fn=None)
    assert result.method == "surrogate"
    assert result.polish_phase is None
    np.testing.assert_allclose(
        result.theta_optimal,
        np.asarray(result.surrogate_phase.theta_optimal, dtype=np.float64),
        atol=1e-12,
    )


@pytest.mark.unit
def test_hybrid_polish_fn_bad_return_raises() -> None:
    """Polish callables must return a mapping with 'coefficients'."""
    surrogate = _tiny_surrogate()
    target = _target_for_surrogate(surrogate)

    def bad_polish(_t: ClubTarget, _theta: np.ndarray) -> object:
        return ("not", "a", "dict")

    opts = HybridOptions(invert=_fast_invert_opts(), polish_solver="fmincon")
    with pytest.raises(ValueError, match="polish_fn must return"):
        fit_swing_hybrid(target, surrogate, options=opts, polish_fn=bad_polish)  # type: ignore[arg-type]
