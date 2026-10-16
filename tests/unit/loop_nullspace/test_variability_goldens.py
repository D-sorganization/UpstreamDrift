"""NSX-80 (#12709) variability-sensitivity goldens.

The goldens come from exact sympy derivations
(``docs/research/loop_nullspace/derivations/variability_goldens.py``) that import
no production code.  These tests (a) check byte-identical regeneration, (b) check
the tangent linearization against an independent finite-difference of the
reduced-coordinate dynamics, so dropping the geometric preload term
``sum_r lambda_r grad^2 Phi_r`` is detected, (c) assert the additive-noise
gradient is exactly zero and (d) check the saltation and minimum-variance
results independently.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

sp = pytest.importorskip("sympy")

ROOT = Path(__file__).resolve().parents[3]
DERIV = ROOT / "docs" / "research" / "loop_nullspace" / "derivations"
GOLDEN = ROOT / "tests" / "fixtures" / "loop_nullspace" / "goldens"
GOLDEN_FILE = GOLDEN / "variability_sensitivity.json"


@pytest.fixture(scope="module")
def vg() -> Any:
    sys.path.insert(0, str(DERIV))
    try:
        spec = importlib.util.spec_from_file_location(
            "variability_goldens_under_test", DERIV / "variability_goldens.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DERIV))
    return module


@pytest.fixture(scope="module")
def gold() -> dict[str, Any]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def _f(text: str) -> float:
    return float(Fraction(text))


def test_golden_regenerates_byte_identically(vg: Any) -> None:
    fresh = json.dumps(vg.generate(), indent=2, sort_keys=True, allow_nan=False)
    assert GOLDEN_FILE.read_text(encoding="utf-8") == fresh + "\n"


def test_check_mode_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(DERIV / "variability_goldens.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=250,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _alpha(theta: float, u: np.ndarray, m: float, length: float, g: float) -> float:
    """Independent reduced-coordinate tangential acceleration, ``u`` fixed."""
    t_hat = np.array([np.cos(theta), np.sin(theta)])
    return float((t_hat @ u - m * g * np.sin(theta)) / (m * length))


def test_tangent_linearization_matches_finite_difference(gold: dict[str, Any]) -> None:
    p = gold["model"]["parameters"]
    m, length, g = _f(p["m"]), _f(p["L"]), _f(p["g"])
    theta0 = float(np.arcsin(_f(p["sin_theta"])))
    for row in gold["linearization"]["allocation_table"]:
        u = np.array([_f(x) for x in row["u"]])
        h = 1e-6
        fd = (
            _alpha(theta0 + h, u, m, length, g) - _alpha(theta0 - h, u, m, length, g)
        ) / (2 * h)
        assert _f(row["F10_with_preload"]) == pytest.approx(fd, abs=1e-7)
        assert _f(row["F10_without_preload"]) != pytest.approx(fd, abs=1e-3)


def test_preload_makes_linearization_depend_on_allocation(
    gold: dict[str, Any],
) -> None:
    table = gold["linearization"]["allocation_table"]
    with_m2 = {_f(r["F10_with_preload"]) for r in table}
    without_m2 = {_f(r["F10_without_preload"]) for r in table}
    assert len(with_m2) == len(table)
    assert len(without_m2) == 1
    omega2 = _f(gold["model"]["parameters"]["omega"]) ** 2
    mass = _f(gold["model"]["parameters"]["m"])
    for row in table:
        expected = omega2 + _f(row["lambda"]) / mass
        assert _f(row["F10_with_preload"]) == pytest.approx(expected, abs=1e-14)
        assert row["F10_with_preload"] == row["F10_expected_omega2_plus_lambda_over_m"]


def test_dropping_preload_changes_the_derivation(vg: Any) -> None:
    keep = vg.linearization(include_preload=True)["F"][1, 0]
    drop = vg.linearization(include_preload=False)["F"][1, 0]
    assert sp.diff(keep, vg.ZZ) != 0
    assert sp.diff(drop, vg.ZZ) == 0
    assert sp.simplify(keep - drop) != 0


def test_kkt_linearization_equals_reduced_coordinates(gold: dict[str, Any]) -> None:
    lin = gold["linearization"]
    assert lin["kkt_equals_reduced_coordinate_symbolic"] is True
    assert lin["G_times_allocation_direction_is_zero"] is True
    assert lin["F11"] == "0"


def test_additive_noise_gradient_is_exactly_zero(vg: Any, gold: dict[str, Any]) -> None:
    flat = vg.additive_flat_case()
    assert flat["gradient_no_preload"] == 0
    assert gold["additive_noise_flat"]["gradient_no_preload"] == "0"
    assert gold["additive_noise_flat"]["gradient_no_preload_is_zero"] is True
    # Flatness is conditional on declaring no mechanism: with the preload
    # mechanism M2 the same additive noise is NOT flat.
    assert flat["gradient_with_preload"] != 0
    assert gold["additive_noise_flat"]["gradient_with_preload_is_zero"] is False


def _var_numeric(gold: dict[str, Any], z: float) -> float:
    """Independent numpy re-evaluation of ``Var(z)`` for the M1-only model."""
    noise = gold["model"]["noise_declared"]
    dt, c = _f(noise["dt"]), np.array([_f(x) for x in noise["c_outcome"]])
    sig0 = np.array([_f(x) for x in noise["sigma0"]])
    k = np.array([_f(x) for x in noise["k"]])
    p0 = np.diag([_f(x) for x in noise["P0_diag"]])
    f10 = _f(gold["min_variance"]["F10_frozen_without_preload"])
    a = np.eye(2) + dt * np.array([[0.0, 1.0], [f10, 0.0]])
    b = (
        dt
        * c[1]
        * np.array([_f(x) for x in gold["linearization"]["G_nominal_om_row"][0]])
    )
    p = gold["model"]["parameters"]
    q = np.array([_f(p["sin_theta"]), -_f(p["cos_theta"])]) * _f(p["L"])
    u = np.array([_f(x) for x in gold["model"]["u_particular"]]) - q * z
    return float(c @ a @ p0 @ a.T @ c + np.sum(b**2 * (sig0**2 + k**2 * u**2)))


def test_min_variance_zstar_is_the_closed_form_minimum(gold: dict[str, Any]) -> None:
    mv = gold["min_variance"]
    assert mv["z_star_matches_closed_form"] is True
    z_star = _f(mv["z_star"])
    assert _f(mv["second_derivative"]) > 0
    grid = np.linspace(z_star - 5.0, z_star + 5.0, 20001)
    values = np.array([_var_numeric(gold, float(z)) for z in grid])
    assert grid[int(np.argmin(values))] == pytest.approx(z_star, abs=1e-3)
    assert _var_numeric(gold, z_star) == pytest.approx(
        _f(mv["var_at_zstar"]), rel=1e-12
    )


def test_saltation_matrix_properties(gold: dict[str, Any]) -> None:
    salt = gold["saltation"]
    f_minus = np.array([[_f(x[0])] for x in salt["f_minus"]])
    for case in salt["cases"]:
        s_mat = np.array([[_f(x) for x in r] for r in case["S"]])
        f_plus = np.array([[_f(x[0])] for x in case["f_plus"]])
        assert s_mat @ f_minus == pytest.approx(f_plus, abs=1e-14)
        assert case["S_f_minus_equals_f_plus"] is True
    assert salt["cases"][1]["S_differs_from_reset_jacobian"] is True
    pi = salt["event_projection"]
    assert pi["grad_g_T_Pi_is_zero"] and pi["Pi_f_minus_is_zero"]
    assert pi["Pi_differs_from_identity"] is True


def test_saltation_agrees_with_closed_form_flow(gold: dict[str, Any]) -> None:
    chk = gold["saltation"]["independent_closed_form_check"]
    assert chk["identity_holds"] is True
    assert chk["event_state_identity_holds"] is True
    # Removing the timing correction must break the identity.
    assert chk["identity_holds_without_timing_term"] is False
    assert chk["event_state_identity_holds_fixed_time"] is False


def test_golfer_preload_depends_on_allocation(gold: dict[str, Any]) -> None:
    g = gold["golfer_preload"]
    assert g["tangent_dimension"] == 3
    assert g["hessian_symmetry_error"] < 1e-12
    assert g["preload_changes_with_allocation"] is True
    k0, k1 = (np.array(a["K_geom_projected"]) for a in g["allocations"])
    assert np.allclose(k0, k0.T, atol=1e-12)
    assert np.linalg.norm(k1 - k0) > 1e-3
