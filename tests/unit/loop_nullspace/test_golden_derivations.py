"""Unit tests for NSX-00 mathematical derivations, registry seed, and contract appendix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = REPO_ROOT / "docs" / "research" / "loop_nullspace"
DERIVATIONS_SCRIPT = RESEARCH_DIR / "derivations" / "generate_goldens.py"
GOLDEN_JSON_PATH = RESEARCH_DIR / "derivations" / "golden_fixtures.json"
LATEX_SPEC_PATH = RESEARCH_DIR / "loop_nullspace.tex"
CONTRACT_APPENDIX_PATH = RESEARCH_DIR / "contract_appendix.md"
ADR_PATH = REPO_ROOT / "docs" / "adr" / "0054-closed-loop-null-space-exploration.md"
REGISTRY_PATH = REPO_ROOT / "src" / "config" / "loop_nullspace_registry.json"
DV_RECORD_PATH = REPO_ROOT / "docs" / "development" / "planning" / "DV-12554.md"
DV_SOURCE_PATH = REPO_ROOT / "docs" / "development" / "planning" / "source-12554.json"


def _load_generator_module() -> Any:
    assert DERIVATIONS_SCRIPT.is_file(), (
        f"Missing derivation script: {DERIVATIONS_SCRIPT}"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_goldens", DERIVATIONS_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_golden_derivations_regeneration_equality() -> None:
    """Checked-in golden_fixtures.json must match freshly generated SymPy derivations."""
    module = _load_generator_module()
    assert GOLDEN_JSON_PATH.is_file(), f"Missing golden JSON: {GOLDEN_JSON_PATH}"
    checked_in = json.loads(GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    generated = module.build_golden_fixtures()

    assert checked_in["schema_version"] == generated["schema_version"] == "1.0.0"
    assert set(checked_in["fixtures"]) == set(generated["fixtures"])

    # Serialize both with identical formatting and compare structure + floats to 1e-12
    def _assert_close_recursive(a: Any, b: Any, path: str = "") -> None:
        if isinstance(a, dict):
            assert isinstance(b, dict), f"Type mismatch at {path}"
            assert set(a.keys()) == set(b.keys()), f"Key mismatch at {path}"
            for k in a:
                _assert_close_recursive(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), f"List mismatch at {path}"
            for idx, (va, vb) in enumerate(zip(a, b, strict=True)):
                _assert_close_recursive(va, vb, f"{path}[{idx}]")
        elif isinstance(a, float | int) and not isinstance(a, bool):
            assert np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12), (
                f"Value mismatch at {path}: {a} != {b}"
            )
        else:
            assert a == b, f"Mismatch at {path}: {a!r} != {b!r}"

    _assert_close_recursive(checked_in, generated)


def test_dimension_accounting_and_nullspace_identities() -> None:
    """Verify R2 dimension counts across full, underactuated, floating-base, and rank-deficient fixtures."""
    checked_in = json.loads(GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    fixtures = checked_in["fixtures"]

    required_fixtures = {
        "planar_two_arm_fixed_base",
        "planar_two_arm_underactuated",
        "floating_base_two_arm",
        "straight_arm_rank_deficient",
        "grasp_map_6x6",
        "analytic_envelope_case",
        "bare_club_partition",
        "c_plus_minus_identity",
    }
    assert required_fixtures.issubset(set(fixtures.keys()))

    for name in (
        "planar_two_arm_fixed_base",
        "planar_two_arm_underactuated",
        "floating_base_two_arm",
        "straight_arm_rank_deficient",
    ):
        fx = fixtures[name]
        A = np.asarray(fx["A"], dtype=float)
        b = np.asarray(fx["b"], dtype=float)
        x_p = np.asarray(fx["x_particular"], dtype=float)
        N = np.asarray(fx["null_basis"], dtype=float)
        dims = fx["dimensions"]

        assert np.allclose(A @ x_p, b, atol=1e-12), f"{name}: A @ x_p != b"
        assert np.allclose(A @ N, 0.0, atol=1e-12), f"{name}: A @ N != 0"
        assert dims["equality_nullity"] == dims["dim_x"] - dims["rank_A"]
        assert N.shape == (dims["dim_x"], dims["equality_nullity"])

    # Underactuated club reduces attainable torque variation relative to full actuation
    full_dims = fixtures["planar_two_arm_fixed_base"]["dimensions"]
    under_dims = fixtures["planar_two_arm_underactuated"]["dimensions"]
    assert (
        under_dims["attainable_torque_variation"]
        == full_dims["attainable_torque_variation"] - 1
    )
    assert fixtures["planar_two_arm_underactuated"]["lambda_zero_feasible"] is False

    # Straight-arm rank-deficient configuration has positive multiplier gauge
    rank_def_dims = fixtures["straight_arm_rank_deficient"]["dimensions"]
    assert rank_def_dims["multiplier_gauge"] > 0


def test_zero_side_three_case_diagnostics() -> None:
    """Verify R3 three-case zero-side solve and exact minimum-residual projection."""
    checked_in = json.loads(GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    zero_cases = checked_in["fixtures"]["zero_side_cases"]

    assert zero_cases["consistent_unique"]["status"] == "CONSISTENT_UNIQUE"
    assert np.isclose(zero_cases["consistent_unique"]["residual_norm"], 0.0, atol=1e-12)

    assert zero_cases["consistent_nonunique"]["status"] == "CONSISTENT_NONUNIQUE"
    assert zero_cases["consistent_nonunique"]["kernel_dim"] > 0
    assert np.isclose(
        zero_cases["consistent_nonunique"]["residual_norm"], 0.0, atol=1e-12
    )

    inc = zero_cases["inconsistent_overdetermined"]
    assert inc["status"] == "INCONSISTENT"
    J_s_T = np.asarray(inc["J_side_T"], dtype=float)
    tau_s = np.asarray(inc["tau_id_side"], dtype=float)
    proj = np.eye(J_s_T.shape[0]) - J_s_T @ np.linalg.pinv(J_s_T)
    expected_res = float(np.linalg.norm(proj @ tau_s))
    assert np.isclose(inc["residual_norm"], expected_res, rtol=1e-12, atol=1e-12)
    assert expected_res > 0.1


def test_grasp_map_6x6_and_bare_club_partition() -> None:
    """Verify R4 6x6 grasp-map kernel, exact inverse, and bare-club mass partition."""
    checked_in = json.loads(GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    gm = checked_in["fixtures"]["grasp_map_6x6"]

    G = np.asarray(gm["G"], dtype=float)
    K = np.asarray(gm["K"], dtype=float)
    T_map = np.asarray(gm["T_12x12"], dtype=float)
    T_inv = np.asarray(gm["T_12x12_inv"], dtype=float)
    W_net = np.asarray(gm["worked_example"]["net_wrench"], dtype=float)
    s_vec = np.asarray(gm["worked_example"]["split_coordinates"], dtype=float)
    W_L = np.asarray(gm["worked_example"]["wrench_left"], dtype=float)
    W_R = np.asarray(gm["worked_example"]["wrench_right"], dtype=float)

    assert G.shape == (6, 12)
    assert K.shape == (12, 6)
    assert np.allclose(G @ K, 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(K) == 6
    assert np.allclose(T_map @ T_inv, np.eye(12), atol=1e-12)

    stacked = np.concatenate([W_L, W_R])
    assert np.allclose(G @ stacked, W_net, atol=1e-12)
    recovered = T_inv @ stacked
    assert np.allclose(recovered[:6], W_net, atol=1e-12)
    assert np.allclose(recovered[6:], s_vec, atol=1e-12)

    bc = checked_in["fixtures"]["bare_club_partition"]
    assert np.isclose(bc["composite_mass_kg"], 1.4212, atol=1e-9)
    assert np.isclose(bc["welded_hands_mass_kg"], 1.1082, atol=1e-9)
    assert np.isclose(bc["bare_club_mass_kg"], 0.313, atol=1e-9)
    assert np.isclose(
        bc["composite_mass_kg"] - bc["welded_hands_mass_kg"],
        bc["bare_club_mass_kg"],
        atol=1e-9,
    )


def test_c_plus_minus_identity_and_envelope_fixture() -> None:
    """Verify R6 c_+/c_- identity and static envelope Chebyshev center."""
    checked_in = json.loads(GOLDEN_JSON_PATH.read_text(encoding="utf-8"))
    cpm = checked_in["fixtures"]["c_plus_minus_identity"]
    assert np.isclose(
        cpm["lhs_asymmetric_sum"], cpm["rhs_decomposed_sum"], rtol=1e-12, atol=1e-12
    )
    assert np.isclose(
        cpm["net_power_at_z1"], cpm["net_power_at_z2"], rtol=1e-12, atol=1e-12
    )

    env = checked_in["fixtures"]["analytic_envelope_case"]
    assert env["chebyshev_radius"] > 0.0
    for low, center, high in zip(
        env["z_min"], env["chebyshev_center"], env["z_max"], strict=True
    ):
        assert low < center < high


def test_registry_seed_schema_and_citation_gate() -> None:
    """Verify loop_nullspace_registry.json schema v1 and enforce no UNVERIFIED sources in statements."""
    assert REGISTRY_PATH.is_file(), f"Missing registry seed: {REGISTRY_PATH}"
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    assert reg["schema_version"] == "1.0.0"
    assert isinstance(reg["hypotheses"], list) and len(reg["hypotheses"]) >= 4
    assert isinstance(reg["conventions"], list) and len(reg["conventions"]) >= 4
    assert isinstance(reg["presets"], list) and len(reg["presets"]) >= 3
    assert isinstance(reg["citations"], list) and len(reg["citations"]) >= 8

    citations_by_id = {c["id"]: c for c in reg["citations"]}
    unverified_tokens = (
        "Thornton",
        "Mehta",
        "Choi",
        "Sorokin",
        "Bosch",
        "Nesbit",
        "two-thirds",
        "15-40 ms",
        "MDPI",
    )

    hyp_ids: set[str] = set()
    for hyp in reg["hypotheses"]:
        assert {
            "id",
            "statement",
            "parameters",
            "source",
            "verification_status",
        } <= set(hyp.keys())
        assert hyp["id"] not in hyp_ids
        hyp_ids.add(hyp["id"])
        assert hyp["verification_status"] in {"VERIFIED", "UNVERIFIED", "REMOVED"}
        for tok in unverified_tokens:
            assert tok.lower() not in hyp["statement"].lower(), (
                f"Unverified source token {tok!r} found in hypothesis statement {hyp['id']}"
            )
        if hyp["source"] in citations_by_id:
            assert (
                hyp["verification_status"]
                == citations_by_id[hyp["source"]]["verification_status"]
            )

    conv_ids = {c["id"] for c in reg["conventions"]}
    assert {
        "C-FRAME-GRIP",
        "C-METRIC-CAPACITY",
        "C-METRIC-IDENTITY",
        "C-METRIC-MINV",
    }.issubset(conv_ids)

    preset_names = {p["name"] for p in reg["presets"]}
    assert {"capacity_min_effort", "envelope_only", "smooth_capacity"}.issubset(
        preset_names
    )
    for preset in reg["presets"]:
        for hid in preset["hypotheses"]:
            assert hid in hyp_ids, f"Preset {preset['name']} references unknown {hid}"
        for cid in preset["conventions"]:
            assert cid in conv_ids, f"Preset {preset['name']} references unknown {cid}"


def test_contract_appendix_and_documents_complete() -> None:
    """Verify contract_appendix.md covers NSX-10..NSX-62 and LaTeX/ADR/DV files exist."""
    assert LATEX_SPEC_PATH.is_file(), f"Missing LaTeX spec: {LATEX_SPEC_PATH}"
    assert CONTRACT_APPENDIX_PATH.is_file(), (
        f"Missing contract appendix: {CONTRACT_APPENDIX_PATH}"
    )
    assert ADR_PATH.is_file(), f"Missing ADR-0054: {ADR_PATH}"
    assert DV_RECORD_PATH.is_file(), f"Missing DV-12554.md: {DV_RECORD_PATH}"
    assert DV_SOURCE_PATH.is_file(), f"Missing source-12554.json: {DV_SOURCE_PATH}"

    appendix_text = CONTRACT_APPENDIX_PATH.read_text(encoding="utf-8")
    required_children = [
        "NSX-01",
        "NSX-02",
        "NSX-03",
        "NSX-04",
        "NSX-10",
        "NSX-11",
        "NSX-12",
        "NSX-13",
        "NSX-14",
        "NSX-15",
        "NSX-16",
        "NSX-17",
        "NSX-20",
        "NSX-21",
        "NSX-22",
        "NSX-23",
        "NSX-24",
        "NSX-25",
        "NSX-26",
        "NSX-30",
        "NSX-31",
        "NSX-40",
        "NSX-41",
        "NSX-42",
        "NSX-43",
        "NSX-44",
        "NSX-50",
        "NSX-51",
        "NSX-52",
        "NSX-53",
        "NSX-54",
        "NSX-55",
        "NSX-56",
        "NSX-60",
        "NSX-61",
        "NSX-62",
    ]
    for child_key in required_children:
        assert child_key in appendix_text, (
            f"Contract appendix missing section for {child_key}"
        )
