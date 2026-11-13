"""Consistency tests for GATES.md, AcceptanceGates, A..G order, and --pelvis-ilc-passes (#15015)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import re

import numpy as np
import pytest

from src.shared.python.motion_matching import constrained_sextic
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
)
from src.shared.python.motion_matching.pipeline import cli, dynamics
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
GATES_DOC = REPO_ROOT / "docs" / "development" / "matched_swing_program" / "GATES.md"
_ROW_RE = re.compile(
    r"^\|\s*[^|]+\|\s*`([a-z0-9_]+)`\s*\|\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)
_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _parse_gates_markdown_table(text: str) -> dict[str, float]:
    """Extract `<field_name>` -> threshold mappings from GATES.md Markdown tables."""
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        match = _ROW_RE.match(line.strip())
        if match:
            parsed[match.group(1)] = float(match.group(2))
    return parsed


def test_gates_md_matches_all_acceptance_gates_fields() -> None:
    """Every numeric field of AcceptanceGates appears in GATES.md with matching value."""
    text = GATES_DOC.read_text(encoding="utf-8")
    parsed = _parse_gates_markdown_table(text)
    gates = AcceptanceGates()

    mismatches: list[str] = []
    for field in dataclasses.fields(AcceptanceGates):
        val = getattr(gates, field.name)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if field.name not in parsed:
            mismatches.append(f"missing {field.name}={val}")
        elif parsed[field.name] != pytest.approx(float(val), rel=1e-6, abs=1e-12):
            mismatches.append(f"{field.name}: doc={parsed[field.name]} != code={val}")

    assert not mismatches, f"GATES.md mismatches AcceptanceGates: {mismatches}"


def test_gates_md_python_snippet_runs_with_acceptance_module() -> None:
    """GATES.md validation example uses acceptance.evaluate and executes cleanly."""
    text = GATES_DOC.read_text(encoding="utf-8")
    assert "acceptance_contract" not in text
    assert "evaluate_receipt_acceptance" not in text

    blocks = _PYTHON_BLOCK_RE.findall(text)
    assert blocks, "GATES.md must include an executable Python example"
    namespace: dict[str, object] = {
        "receipt_dict": {
            "capture": "driver",
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.030,
            "club_marker_rmse_m": 0.050,
            "pelvis_yaw_error_pct": 2.0,
        }
    }
    exec("\n".join(blocks), namespace)
    verdict = namespace.get("verdict")
    assert isinstance(verdict, AcceptanceVerdict)


def test_single_a_to_g_coefficient_order_and_descending_helpers() -> None:
    """Descending polyval helpers are distinct from ascending A..G convention."""
    for obj in (
        constrained_sextic,
        constrained_sextic.simscape_powers_to_normalized,
        constrained_sextic.normalized_to_simscape_powers,
        constrained_sextic.cubic_to_normalized_sextic,
    ):
        doc = obj.__doc__ or ""
        assert "A*t^6" not in doc
        assert "[A, B, C, D, E, F, G]" not in doc

    descending_fn = getattr(
        constrained_sextic, "descending_polyval_to_normalized", None
    )
    ascending_back_fn = getattr(
        constrained_sextic, "normalized_to_descending_polyval", None
    )
    assert callable(descending_fn)
    assert callable(ascending_back_fn)

    rng = np.random.default_rng(7)
    coeffs_asc = rng.standard_normal((2, 7))
    coeffs_desc = coeffs_asc[:, ::-1]
    t_full = 1.814
    t = 0.42

    norm = descending_fn(coeffs_desc, T_full=t_full)
    recovered_desc = ascending_back_fn(norm, T_full=t_full)
    np.testing.assert_allclose(recovered_desc, coeffs_desc, rtol=1e-12, atol=1e-12)

    tau_asc = evaluate_polynomial_torque(coeffs_asc, t)
    tau_norm = constrained_sextic.evaluate_normalized_sextic(
        norm, np.array([t]), T_full=t_full
    )[0]
    np.testing.assert_allclose(tau_norm, tau_asc, rtol=1e-10, atol=1e-10)

    gates_text = GATES_DOC.read_text(encoding="utf-8")
    assert "c_6" in gates_text and "A + B*t" in gates_text


def test_pelvis_ilc_passes_flag_and_shooting_fit_terminology() -> None:
    """CLI accepts --pelvis-ilc-passes and documents iterative-learning semantics."""
    parser = cli.build_parser()
    args_new = parser.parse_args(["--pelvis-ilc-passes", "4"])
    args_legacy = parser.parse_args(["--shooting-fit", "4"])
    assert args_new.shooting_fit == 4
    assert args_legacy.shooting_fit == 4
    assert "iterative-learning" in parser.format_help().lower()
    assert "iterative-learning" in (dynamics.shooting_fit.__doc__ or "").lower()
