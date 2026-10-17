"""Global sensitivity screening tests (NSX-96, #12725).

The oracle is the analytic Ishigami function (a = 7, b = 0.1) with inputs
``U(-pi, pi)``: ``S1 = (0.3139, 0.4424, 0)``, ``ST = (0.5576, 0.4424, 0.2437)``.
The planar oracle and low-rank basis tests check structure and evidence typing
only; indices from them are screening, never a validated ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.global_sensitivity import (
    InputGroup,
    InputSpace,
    SensitivityInput,
    prcc,
    sobol_convergence,
    sobol_indices,
)
from src.shared.python.loop_nullspace.sensitivity_records import (
    SCREENING_CONVENTION,
    basis_response_model,
    planar_oracle_model,
    prcc_records,
    sobol_records,
)

pytestmark = pytest.mark.unit

A, B = 7.0, 0.1
VAR = A**2 / 8 + B * np.pi**4 / 5 + B**2 * np.pi**8 / 18 + 0.5
S1 = np.array(
    [
        (0.5 * (1 + B * np.pi**4 / 5) ** 2) / VAR,
        (A**2 / 8) / VAR,
        0.0,
    ]
)
V13 = 8 * B**2 * np.pi**8 / 225
ST = np.array([S1[0] + V13 / VAR, S1[1], V13 / VAR])


def ishigami(x: np.ndarray) -> np.ndarray:
    return (
        np.sin(x[:, 0]) + A * np.sin(x[:, 1]) ** 2 + B * x[:, 2] ** 4 * np.sin(x[:, 0])
    )


def ishigami_space() -> InputSpace:
    return InputSpace(
        tuple(SensitivityInput(f"x{i}", -np.pi, np.pi) for i in range(1, 4))
    )


def test_ishigami_first_and_total_recovered_at_n2048() -> None:
    res = sobol_indices(ishigami, ishigami_space(), n=2048, seed=7, n_boot=100)
    assert np.max(np.abs(res.first_order - S1)) < 0.02
    assert np.max(np.abs(res.total - ST)) < 0.02


def test_same_seed_is_bit_identical() -> None:
    a = sobol_indices(ishigami, ishigami_space(), n=256, seed=3, n_boot=20)
    b = sobol_indices(ishigami, ishigami_space(), n=256, seed=3, n_boot=20)
    assert np.array_equal(a.first_order, b.first_order)
    assert np.array_equal(a.total_ci, b.total_ci)


def test_convergence_reports_index_versus_n_and_error_shrinks() -> None:
    conv = sobol_convergence(
        ishigami, ishigami_space(), ns=(64, 256, 1024, 2048), seed=5, n_boot=50
    )
    assert [r.n for r in conv] == [64, 256, 1024, 2048]
    err = [float(np.max(np.abs(r.total - ST))) for r in conv]
    assert err[-1] < err[0]
    assert all(r.total_ci.shape == (3, 2) for r in conv)


def test_correlated_inputs_refuse_ordinary_indices() -> None:
    space = InputSpace(
        tuple(SensitivityInput(f"x{i}", -np.pi, np.pi) for i in range(1, 4)),
        groups=(InputGroup(("x1", "x2"), np.array([[1.0, 0.8], [0.8, 1.0]])),),
    )
    assert space.is_correlated
    with pytest.raises(ValueError, match="grouped"):
        sobol_indices(ishigami, space, n=64, seed=1)


def test_grouped_indices_are_allowed_and_group_sums_consistent() -> None:
    space = InputSpace(
        tuple(SensitivityInput(f"x{i}", -np.pi, np.pi) for i in range(1, 4)),
        groups=(InputGroup(("x1", "x2"), np.array([[1.0, 0.6], [0.6, 1.0]])),),
    )
    res = sobol_indices(ishigami, space, n=1024, seed=2, level="groups", n_boot=20)
    assert res.names == ("x1+x2", "x3")
    # x3 only acts through the x1 interaction, so its total is small but > 0.
    assert 0.0 < res.total[1] < res.total[0]
    assert np.all(res.total >= res.first_order - 0.05)


def test_independent_groups_match_ordinary_indices() -> None:
    sp = ishigami_space()
    a = sobol_indices(ishigami, sp, n=512, seed=4, level="groups", n_boot=10)
    b = sobol_indices(ishigami, sp, n=512, seed=4, n_boot=10)
    assert np.allclose(a.total, b.total)


def test_group_validation() -> None:
    inputs = tuple(SensitivityInput(f"x{i}", 0.0, 1.0) for i in range(2))
    with pytest.raises(ValueError, match="unknown"):
        InputSpace(inputs, groups=(InputGroup(("x0", "zz"), np.eye(2)),))
    with pytest.raises(ValueError, match="positive definite"):
        InputSpace(
            inputs,
            groups=(InputGroup(("x0", "x1"), np.array([[1.0, 1.0], [1.0, 1.0]])),),
        )
    with pytest.raises(ValueError, match="twice"):
        InputSpace(
            inputs,
            groups=(
                InputGroup(("x0", "x1"), np.eye(2)),
                InputGroup(("x1", "x0"), np.eye(2)),
            ),
        )


def test_prcc_ranks_monotone_drivers() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return 4.0 * x[:, 0] ** 3 - 1.0 * x[:, 1] + 0.0 * x[:, 2]

    sp = InputSpace(tuple(SensitivityInput(f"x{i}", 0.0, 1.0) for i in range(3)))
    res = prcc(f, sp, n=1000, seed=11, n_boot=50)
    assert res.prcc[0] > 0.9
    assert res.prcc[1] < -0.3
    assert abs(res.prcc[2]) < 0.1
    assert res.ci.shape == (3, 2)


def test_input_validation() -> None:
    sp = ishigami_space()
    with pytest.raises(ValueError, match="power of two"):
        sobol_indices(ishigami, sp, n=100, seed=1)
    with pytest.raises(ValueError, match="level"):
        sobol_indices(ishigami, sp, n=64, seed=1, level="x")
    with pytest.raises(ValueError, match="shape"):
        sobol_indices(lambda x: x, sp, n=64, seed=1)
    with pytest.raises(ValueError, match="finite"):
        sobol_indices(lambda x: np.full(len(x), np.nan), sp, n=64, seed=1)
    with pytest.raises(ValueError, match="low < high"):
        SensitivityInput("a", 1.0, 1.0)
    with pytest.raises(ValueError, match="zero variance"):
        sobol_indices(lambda x: np.ones(len(x)), sp, n=64, seed=1)


def test_records_are_screening_and_unvalidated() -> None:
    res = sobol_indices(ishigami, ishigami_space(), n=128, seed=1, n_boot=10)
    recs = sobol_records(res, model_name="ishigami", model_version="1")
    assert set(recs) == {"first_order", "total", "first_order_ci", "total_ci"}
    for rec in recs.values():
        assert SCREENING_CONVENTION in rec.conventions
        assert rec.validation_status is ValidationStatus.UNVALIDATED
        assert rec.allocation_dependence is AllocationDependence.INVARIANT
        assert rec.units == "1"
        assert rec.values.shape[0] == 3
    p = prcc_records(
        prcc(ishigami, ishigami_space(), n=128, seed=1, n_boot=10),
        model_name="ishigami",
        model_version="1",
    )
    assert SCREENING_CONVENTION in p["prcc"].conventions


def test_planar_oracle_screening_runs_and_orders_mass_over_dummy() -> None:
    model, space = planar_oracle_model()
    res = sobol_indices(model, space, n=256, seed=9, n_boot=20)
    assert res.names == space.input_names
    assert np.all(res.total >= -0.05)
    assert float(res.total.max()) > 0.1
    rec = sobol_records(res, model_name="planar", model_version="NSX-80")["total"]
    assert SCREENING_CONVENTION in rec.conventions


def test_low_rank_basis_screening_matches_quadratic_form() -> None:
    model, space = basis_response_model(n_coefficients=4, n_frames=40)
    res = sobol_indices(model, space, n=512, seed=2, n_boot=20)
    # response is additive in the coefficients: first order ~ total, sums to ~1.
    assert abs(res.first_order.sum() - 1.0) < 0.1
    assert np.max(np.abs(res.total - res.first_order)) < 0.1
