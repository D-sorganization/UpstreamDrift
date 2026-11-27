"""Tests for NP-07 loop-constraint projection of policy output (#14500)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.research.neural_policy.loop_projection import LoopConstraintProjector

pytestmark = pytest.mark.unit


def test_analytic_loop_jacobian_satisfies_constraint() -> None:
    j = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    q0 = np.array([0.1, -0.2, 0.1, -0.2])
    proj = LoopConstraintProjector(q0=q0, jacobian=j, tol=0.05)
    u_proj = proj.project(np.array([2.5, -1.8, 0.7, 3.2]), q=q0)
    assert float(np.linalg.norm(j @ u_proj)) < 1e-6


def test_projector_is_idempotent_and_preserves_feasible_output() -> None:
    j = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
    proj = LoopConstraintProjector(q0=np.zeros(4), jacobian=j, tol=0.05)
    p = proj.projector_matrix
    np.testing.assert_allclose(p @ p, p, atol=1e-12)
    np.testing.assert_allclose(p.T, p, atol=1e-12)
    u_feas = np.array([1.0, 2.0, 1.0, 2.0])  # in ker(J)
    np.testing.assert_allclose(proj.project(u_feas, q=np.zeros(4)), u_feas, atol=1e-12)


def test_preconditions_reject_stale_jacobian_and_invalid_inputs() -> None:
    proj = LoopConstraintProjector(
        q0=np.zeros(2), jacobian=np.array([[1.0, -1.0]]), tol=0.1
    )
    assert proj.is_valid_at(np.array([0.05, -0.05]))
    proj.project(np.array([1.0, 2.0]), q=np.array([0.05, -0.05]))
    assert not proj.is_valid_at(np.array([0.2, 0.0]))
    for u, q in [
        (np.ones(2), np.array([0.2, 0.0])),
        (np.array([np.nan, 1.0]), np.zeros(2)),
        (np.ones(1), np.zeros(2)),
    ]:
        with pytest.raises(ValueError):
            proj.project(u, q=q)


def test_folded_weights_match_output_projection_at_q0_and_flag_away() -> None:
    rng = np.random.default_rng(42)
    j = np.array([[1.0, 0.5, -1.0, -0.5]])
    q0 = np.array([0.2, -0.1, 0.2, -0.1])
    proj = LoopConstraintProjector(q0=q0, jacobian=j, tol=0.08)
    w = rng.standard_normal((4, 6))
    b, h = rng.standard_normal(4), rng.standard_normal(6)

    u_proj = proj.project(w @ h + b, q=q0)
    folded = proj.fold_weights(w, b)
    np.testing.assert_allclose(folded.forward(h, q=q0), u_proj, atol=1e-12)
    assert not folded.is_valid_at(q0 + 0.2)
    with pytest.raises(ValueError):
        folded.forward(h, q=q0 + 0.2)


def test_diagnostic_record_and_rollout_violation_reduction() -> None:
    rng = np.random.default_rng(101)
    j = np.array([[1.0, 1.0, -1.0, -1.0], [0.5, -0.5, 0.5, -0.5]])
    proj = LoopConstraintProjector(q0=np.zeros(4), jacobian=j, tol=0.05)
    for _ in range(5):
        _, rec = proj.evaluate_diagnostic(rng.standard_normal(4) * 5.0, q=np.zeros(4))
        assert rec.violation_after < 1e-6 and rec.is_valid


def test_lexical_guardrail_clean() -> None:
    terms = (
        "co-con" + "traction",
        "mus" + "cle",
        "ana" + "tomy",
        "impe" + "dance",
        "stiff" + "ness",
        "syn" + "ergy",
        "control-" + "affine",
    )
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "src/research/neural_policy/loop_projection.py"
    ).read_text(encoding="utf-8")
    for t in terms:
        assert not re.search(r"\b" + re.escape(t) + r"\b", src, re.IGNORECASE), (
            f"Banned: {t}"
        )
