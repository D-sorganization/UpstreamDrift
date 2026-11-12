"""Tests for NP-05 localized edit vs fine-tune benchmark (#14480)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.research.neural_policy.edit_vs_finetune_benchmark import (
    format_markdown_table,
    run_benchmark,
    validate_splits,
)

pytestmark = pytest.mark.unit


def _deterministic_fixtures():
    rng = np.random.default_rng(42)
    in_dim, out_dim = 6, 4
    n_train, n_heldout = 20, 10
    train_x = rng.standard_normal((n_train, in_dim)) + 2.0
    heldout_x = rng.standard_normal((n_heldout, in_dim)) - 2.0
    base_weight = rng.standard_normal((out_dim, in_dim)) * 0.5
    target_weight = base_weight.copy()
    target_weight[0, :] -= 0.15 * base_weight[0, :]
    target_weight[1:, :] += 0.02 * rng.standard_normal((out_dim - 1, in_dim))
    train_y = (target_weight @ train_x.T).T
    return base_weight, train_x, train_y, heldout_x


def test_report_generator_produces_table_deterministically() -> None:
    w, tx, ty, hx = _deterministic_fixtures()
    report = run_benchmark(w, tx, ty, hx, seeds=(42, 43, 44, 45, 46))
    table = format_markdown_table(report)

    assert "`baseline`" in table
    assert "`rank1_edit`" in table
    assert "`lora_finetune`" in table
    assert "0.3047" in table
    assert "0.1141" in table
    assert report.metadata["run_record"] == "#14436/manifest_v1"
    assert report.metadata["experiment_contract"] == "#14476/spec_v1"
    assert "no physiological coordination" in report.disclaimer


def test_heldout_split_validation_rejects_overlap() -> None:
    tx = np.array([[1.0, 2.0], [3.0, 4.0]])
    hx_clean = np.array([[5.0, 6.0], [7.0, 8.0]])
    validate_splits(tx, hx_clean)

    hx_overlap = np.array([[1.0, 2.0], [7.0, 8.0]])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(tx, hx_overlap)


def test_lexical_guardrail_zero_violations() -> None:
    words = "imp:edance,stiff:ness,syn:ergy,co-con:traction,mus:cle,ana:tomy,control-:affine,ZT:CF,ZV:CF"
    banned = [w.replace(":", "") for w in words.split(",")]  # noqa: SIM905
    root = Path(__file__).resolve().parents[2]
    files = (
        "src/research/neural_policy/edit_vs_finetune_benchmark.py",
        "docs/research/np05_edit_vs_finetune_report.md",
        "tests/research/test_neural_policy_np05.py",
        "changes/14480-np05-edit-vs-finetune-benchmark.md",
    )
    for rel in files:
        txt = (root / rel).read_text(encoding="utf-8")
        for b in banned:
            assert not re.search(r"\b" + re.escape(b) + r"\b", txt, re.IGNORECASE), (
                f"{b} in {rel}"
            )
