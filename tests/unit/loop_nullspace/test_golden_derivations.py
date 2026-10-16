"""NSX-00 golden derivations: regenerate from sympy and compare to checked-in JSON.

The goldens in ``tests/fixtures/loop_nullspace/goldens`` are produced by
``docs/research/loop_nullspace/derivations``.  Those modules import no
repository production code, so a later solver cannot be checked against itself.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("sympy")

ROOT = Path(__file__).resolve().parents[3]
DERIV = ROOT / "docs" / "research" / "loop_nullspace" / "derivations"
GOLDENS = ROOT / "tests" / "fixtures" / "loop_nullspace" / "goldens"
RTOL, ATOL = 1e-12, 1e-13


@pytest.fixture(scope="module")
def generated() -> dict[str, dict[str, Any]]:
    sys.path.insert(0, str(DERIV))
    try:
        spec = importlib.util.spec_from_file_location(
            "loop_nullspace_generate_goldens", DERIV / "generate_goldens.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.generate_all()
    finally:
        sys.path.remove(str(DERIV))
    return json.loads(json.dumps(data))


def _stored(name: str) -> dict[str, Any]:
    return json.loads((GOLDENS / f"{name}.json").read_text(encoding="utf-8"))


def _compare(a: Any, b: Any, path: str = "") -> None:
    """Floats within tolerance; strings, ints, bools and structure exactly."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str):
        assert a == b, path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _compare(x, y, f"{path}[{i}]")
    elif isinstance(a, int) and isinstance(b, int):
        assert a == b, path
    elif a is None:
        assert b is None, path
    else:
        assert a == pytest.approx(b, rel=RTOL, abs=ATOL), path


NAMES = [
    "planar_two_arm_full",
    "planar_two_arm_club_unactuated",
    "planar_straight_arm_rank_deficient",
    "spatial_counts",
    "envelope_analytic_lp",
    "grasp_map_worked_example",
    "c_pm_identity",
    "closure_convention",
    "variability_sensitivity",
    "planar_five_bar_tangent_state_space",
]


@pytest.mark.parametrize("name", NAMES)
def test_checked_in_golden_matches_regeneration(name: str, generated: dict) -> None:
    stored = _stored(name)
    assert stored["schema"] == "loop-nullspace-golden/1"
    _compare(generated[name], stored, name)


def _planar_cases() -> list[tuple[str, dict[str, Any]]]:
    out = [
        ("full", _stored("planar_two_arm_full")),
        ("club_unactuated", _stored("planar_two_arm_club_unactuated")),
    ]
    for key, val in _stored("planar_straight_arm_rank_deficient")["variants"].items():
        out.append((f"straight_{key}", val))
    return out


@pytest.mark.parametrize(("label", "fx"), _planar_cases())
def test_planar_golden_internal_consistency(label: str, fx: dict[str, Any]) -> None:
    """Re-derive the dimension counts from the stored matrices with numpy."""
    mt = {k: np.array(v, dtype=float) for k, v in fx["matrices"].items()}
    m_mat, s_mat, jc = mt["M"], mt["S"], mt["J_c"]
    assert np.allclose(m_mat, m_mat.T, atol=1e-12)
    assert np.linalg.eigvalsh(m_mat).min() > 0.0
    n_a = s_mat.shape[0]
    jcu = jc @ (np.eye(jc.shape[1]) - s_mat.T @ s_mat)
    a_mat = np.hstack([s_mat.T, jc.T])
    rank = np.linalg.matrix_rank
    d = fx["dimensions"]
    assert d["rank_A"] == rank(a_mat, tol=1e-10)
    assert d["rank_J_c"] == rank(jc, tol=1e-10)
    assert d["rank_A"] == n_a + d["rank_J_cu_T"]
    assert d["equality_nullity"] == d["dim_x"] - d["rank_A"]
    assert d["multiplier_gauge"] == d["m"] - d["rank_J_c"]
    assert d["attainable_torque_variation"] == d["rank_J_c"] - d["rank_J_cu_T"]
    assert (
        d["equality_nullity"]
        == d["multiplier_gauge"] + d["attainable_torque_variation"]
    )
    assert d["rank_J_cu_T"] == rank(jcu, tol=1e-10)


def test_zero_side_cases_are_the_declared_three() -> None:
    allowed = {"CONSISTENT_UNIQUE", "CONSISTENT_NONUNIQUE", "INCONSISTENT"}
    for _, fx in _planar_cases():
        for side in fx["zero_side"].values():
            assert side["case"] in allowed


def test_dimension_golden_for_spatial_counts_is_exact() -> None:
    variants = _stored("spatial_counts")["variants"]
    two_feet = variants["floating_root_two_point_feet"]
    assert two_feet["rank_J_g"] == 5
    assert variants["fixed_base_arms_only"]["rank_J_c"] == 6
    for v in variants.values():
        assert v["rank_A"] == v["n_actuated"] + v["rank_stacked_unactuated_columns"]


def test_c_pm_identity_golden_has_unique_uniform_minimiser() -> None:
    d = _stored("c_pm_identity")
    assert d["uniform_minimisers_coincide"] is True
    assert d["heterogeneous_differs_from_uniform"] is True
    assert d["identity_max_error"] < 1e-12
