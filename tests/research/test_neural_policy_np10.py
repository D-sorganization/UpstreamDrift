"""Toy tests for NP-10 weight inspector and lexical guardrail (#14503)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from src.research.neural_policy.weight_inspector import (
    compute_effective_rank,
    compute_input_attribution,
    compute_matrix_conditioning,
    compute_state_dict_sha256,
    count_dead_units,
    export_inspection_json,
    find_lexical_guardrail_violations,
    inspect_layer,
    inspect_policy_weights,
)

pytestmark = pytest.mark.unit


def test_synthetic_matrix_rank_and_conditioning_reported_exactly() -> None:
    # 1) Full-rank diagonal matrix with known singular values
    sv = np.array([10.0, 5.0, 2.0, 1.0], dtype=float)
    cond = compute_matrix_conditioning(sv)
    assert math.isclose(cond, 10.0, rel_tol=1e-12)

    total = np.sum(sv)
    p = sv / total
    expected_entropy = -np.sum(p * np.log(p))
    expected_erank = float(np.exp(expected_entropy))
    actual_erank = compute_effective_rank(sv)
    assert math.isclose(actual_erank, expected_erank, rel_tol=1e-12)

    # 2) Rank-2 matrix with equal non-zero singular values -> erank == 2.0 exactly
    sv_rank2 = np.array([4.0, 4.0, 0.0, 0.0], dtype=float)
    assert math.isclose(compute_effective_rank(sv_rank2), 2.0, rel_tol=1e-12)
    assert math.isinf(compute_matrix_conditioning(sv_rank2))

    # 3) Rank-1 matrix -> erank == 1.0 exactly
    u = np.array([1.0, 2.0, -1.0])
    v = np.array([3.0, 0.0, 1.0])
    w_rank1 = np.outer(u, v)
    layer_rep = inspect_layer(w_rank1)
    assert math.isclose(layer_rep["effective_rank"], 1.0, rel_tol=1e-12)


def test_dead_unit_detection_on_planted_all_zero_row() -> None:
    w: NDArray[np.floating] = np.ones((4, 5), dtype=float)
    w[2, :] = 0.0  # Planted all-zero row
    w[3, :] = 1e-11  # Below 1e-9 threshold

    dead_default = count_dead_units(w, threshold=1e-9)
    assert dead_default == 2

    dead_strict = count_dead_units(w, threshold=0.0)
    assert dead_strict == 1


def test_input_channel_attribution_norms_q_qdot() -> None:
    w: NDArray[np.floating] = np.zeros((2, 4), dtype=float)
    w[0, 0] = 3.0  # q slot (cols 0, 1)
    w[1, 2] = 4.0  # qdot slot (cols 2, 3)

    attr = compute_input_attribution(w)
    assert math.isclose(attr["q"], 3.0, rel_tol=1e-12)
    assert math.isclose(attr["qdot"], 4.0, rel_tol=1e-12)


def test_deterministic_json_output_and_state_dict_sha256() -> None:
    rng = np.random.default_rng(42)
    state_dict = {
        "layer_b": rng.standard_normal((3, 3)),
        "layer_a": rng.standard_normal((2, 4)),
    }
    digest1 = compute_state_dict_sha256(state_dict)
    digest2 = compute_state_dict_sha256(state_dict)
    assert digest1 == digest2
    assert len(digest1) == 64

    report1 = inspect_policy_weights(state_dict)
    report2 = inspect_policy_weights(state_dict)
    json1 = export_inspection_json(report1)
    json2 = export_inspection_json(report2)

    assert json1 == json2
    loaded = json.loads(json1)
    assert loaded["provenance"]["state_dict_sha256"] == digest1
    assert "layer_a" in loaded["layers"]
    assert "layer_b" in loaded["layers"]


def test_lexical_guardrail_fails_on_banned_terms() -> None:
    banned_samples = [
        "imped" + "ance control",
        "stiff" + "ness matrix",
        "syn" + "ergy pattern",
        "co-con" + "traction ratio",
        "mus" + "cle activation",
        "human ana" + "tomy",
        "control-" + "affine model",
        "ZT" + "CF formulation",
        "ZV" + "CF baseline",
    ]
    for sample in banned_samples:
        violations = find_lexical_guardrail_violations(sample)
        assert len(violations) > 0, f"Expected violation for sample: {sample}"


def test_lexical_guardrail_passes_on_clean_and_scoped_and_research_files() -> None:
    clean = "Compact policy evaluation on synthetic state vectors."
    assert len(find_lexical_guardrail_violations(clean)) == 0

    scoped_ztcf = "ZT" + "CF and ZV" + "CF validated under #13624 reference."
    assert len(find_lexical_guardrail_violations(scoped_ztcf)) == 0

    disclaimer = "Serializable rank-1 weight intervention (not a syn" + "ergy claim)."
    assert len(find_lexical_guardrail_violations(disclaimer)) == 0

    # Test that research policy files pass cleanly
    research_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "research"
        / "neural_policy"
    )
    for py_file in research_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        violations = find_lexical_guardrail_violations(text)
        assert violations == [], (
            f"Unexpected violations in {py_file.name}: {violations}"
        )
