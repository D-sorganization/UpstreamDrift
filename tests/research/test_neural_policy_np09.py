"""Tests for NP-09 policy input-output sensitivity (#14502)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.research.neural_policy.io_sensitivity import (
    NAMED_SWING_EVENTS,
    CompactMLP,
    compute_finite_difference_jacobian,
    compute_io_sensitivity,
)

pytestmark = pytest.mark.unit


def test_analytic_jacobian_matches_finite_differences_within_one_percent() -> None:
    rng = np.random.default_rng(42)
    mlp = CompactMLP(
        (rng.standard_normal((8, 6)), rng.standard_normal((3, 8))),
        (rng.standard_normal(8) * 0.1, rng.standard_normal(3) * 0.1),
    )
    sq_a, sqd_a = mlp.analytic_jacobian([0.1, -0.2, 0.3], [1.0, -0.5, 0.2])
    sq_fd, sqd_fd = compute_finite_difference_jacobian(
        mlp, [0.1, -0.2, 0.3], [1.0, -0.5, 0.2], h=1e-5
    )
    assert np.allclose(sq_a, sq_fd, rtol=0.01, atol=1e-4)
    assert np.allclose(sqd_a, sqd_fd, rtol=0.01, atol=1e-4)


def test_linear_toy_policy_returns_k_and_d_exactly() -> None:
    k, d = np.array([[2.0, -1.0], [0.5, 3.0]]), np.array([[0.1, 0.0], [-0.2, 0.4]])
    pol = CompactMLP((np.hstack([k, d]),), (np.zeros(2),), activation="linear")
    rep = compute_io_sensitivity(pol, [0.5, -0.2], [1.0, 2.0])
    assert np.allclose(rep.s_q, k, atol=1e-12) and np.allclose(rep.s_qd, d, atol=1e-12)


def test_dbc_rejects_non_finite_and_handles_envelope() -> None:
    pol = CompactMLP((np.zeros((2, 4)),), (np.zeros(2),))
    with pytest.raises(ValueError, match="finite"):
        compute_io_sensitivity(pol, [np.nan, 0.0], [0.0, 0.0])
    env = ([-1.0] * 4, [1.0] * 4)
    with pytest.raises(ValueError, match="outside"):
        compute_io_sensitivity(pol, [2.0, 0.0], [0.0, 0.0], envelope=env, strict=True)
    assert compute_io_sensitivity(
        pol, [2.0, 0.0], [0.0, 0.0], envelope=env
    ).off_manifold


def test_named_swing_events_and_report_properties() -> None:
    pol = CompactMLP((np.eye(4),), (np.zeros(4),), activation="linear")
    assert NAMED_SWING_EVENTS == ("address", "top", "mid_downswing", "impact")
    for ev in NAMED_SWING_EVENTS:
        rep = compute_io_sensitivity(pol, [0.1, 0.2], [0.3, 0.4], event=ev)
        assert rep.event == ev and "not guaranteed" in rep.disclaimer


REQUIRED_LEXICAL_FILES = (
    "src/research/neural_policy/io_sensitivity.py",
    "tests/research/test_neural_policy_np09.py",
)
# Collation deletes the fragment after folding it into SPEC.md. Scan it only
# while it is still on disk so the merge queue does not fail on its absence.
OPTIONAL_LEXICAL_FILES = ("changes/14502-np09-io-sensitivity.md",)


def _assert_lexical_guardrail(root: Path) -> None:
    words = "imp:edance,stiff:ness,syn:ergy,co-con:traction,mus:cle,ana:tomy,control-:affine,ZT:CF,ZV:CF"
    banned = [w.replace(":", "") for w in words.split(",")]  # noqa: SIM905
    files: list[str] = list(REQUIRED_LEXICAL_FILES)
    files.extend(rel for rel in OPTIONAL_LEXICAL_FILES if (root / rel).is_file())
    for rel in files:
        txt = (root / rel).read_text(encoding="utf-8")
        for b in banned:
            assert not re.search(r"\b" + re.escape(b) + r"\b", txt, re.IGNORECASE), (
                f"{b} in {rel}"
            )


def test_lexical_guardrail_zero_violations() -> None:
    root = Path(__file__).resolve().parents[2]
    _assert_lexical_guardrail(root)


def test_lexical_guardrail_tolerates_collated_change_fragment_and_catches_violation(
    tmp_path: Path,
) -> None:
    for rel in REQUIRED_LEXICAL_FILES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("clean baseline content\n", encoding="utf-8")

    _assert_lexical_guardrail(tmp_path)

    fragment = tmp_path / OPTIONAL_LEXICAL_FILES[0]
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(f"bad {'mus:cle'.replace(':', '')} token\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_lexical_guardrail(tmp_path)
