"""Functional identifiability and attribution quantities (NSE-UD-1, #13461).

``classify_functional`` labels a linear output ``s . x`` of the allocation
``x = x_p + N z`` as numerically invariant or family dependent from the
normalised annihilation residual ``rho = ||N^T s|| / ||s_x * s||``.  The
attribution factory builds per-source quantities ``J L E_k`` on
``x_functional_quantities`` so the intervals come from ``static_envelopes``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    CharacteristicScales,
    decompose,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    BoundStatus,
    static_envelopes,
)
from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow  # noqa: E402
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    generic_planar_config,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.functional_identifiability import (  # noqa: E402
    FunctionalIdentifiability,
    FunctionalStatus,
    attribution_quantities,
    classify_functional,
    functional_record,
)
from src.shared.python.loop_nullspace.hit_and_run import (  # noqa: E402
    TrajectoryDomain,
    hit_and_run,
)

pytestmark = pytest.mark.unit

TOL = 1e-9
MODEL = ModelProvenance("planar-golfer", "test")
SOURCE = InputProvenance("fixture", "generic_planar_config")


def _dec(scales: CharacteristicScales | None = None):
    snap = planar_golfer_snapshot(generic_planar_config(), actuation="full").snapshot
    return decompose(snap, weights=np.eye(snap.topology.n_actuated), scales=scales)


def _invariant_selector(dec, seed: int = 0) -> np.ndarray:
    """A selector in ``range(A^T)``: the part of a random ``w`` orthogonal to ``N``."""
    w = np.random.default_rng(seed).standard_normal(dec.dim_x)
    coef = np.linalg.lstsq(dec.N, w, rcond=None)[0]
    return w - dec.N @ coef


def _variable_selector(dec) -> np.ndarray:
    s = np.zeros(dec.dim_x)
    s[dec.n_actuated] = 1.0  # a closure multiplier: varies along the fibre
    return s


def _box_set(dec, half: float = 1.0) -> FeasibleSet:
    n = dec.nullity
    rows = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        rows.append(FeasibleRow(e, half, f"B:{i}:up", "B", "torque"))
        rows.append(FeasibleRow(-e, half, f"B:{i}:lo", "B", "torque"))
    return dataclasses.replace(FeasibleSet.from_rows(n, rows), decomposition=dec)


# ---------------------------------------------------------- classification
def test_invariant_selector_is_numerically_invariant() -> None:
    dec = _dec()
    res = classify_functional(dec, _invariant_selector(dec), tol=1e-8)
    assert isinstance(res, FunctionalIdentifiability)
    assert res.status is FunctionalStatus.NUMERICALLY_INVARIANT
    assert res.residual <= 1e-8
    assert res.tol == 1e-8


def test_variable_selector_is_family_dependent() -> None:
    dec = _dec()
    res = classify_functional(dec, _variable_selector(dec), tol=1e-8)
    assert res.status is FunctionalStatus.FAMILY_DEPENDENT
    assert 1e-8 < res.residual <= 1.0 + 1e-12


def test_zero_selector_is_invariant_with_zero_residual() -> None:
    dec = _dec()
    res = classify_functional(dec, np.zeros(dec.dim_x), tol=1e-8)
    assert res.status is FunctionalStatus.NUMERICALLY_INVARIANT
    assert res.residual == 0.0


def test_residual_matches_definition_without_second_scaling() -> None:
    dec = _dec()
    s = _variable_selector(dec) + 0.3 * _invariant_selector(dec, 1)
    res = classify_functional(dec, s, tol=1e-8)
    want = np.linalg.norm(dec.N.T @ s) / np.linalg.norm(dec.variable_scale * s)
    assert res.residual == pytest.approx(want, rel=1e-12)
    assert res.annihilation_norm == pytest.approx(np.linalg.norm(dec.N.T @ s))


def test_selector_rescaling_keeps_status_and_rho_and_scales_bound() -> None:
    dec = _dec()
    fs = _box_set(dec)
    radius = float(np.sqrt(dec.nullity))
    s = _variable_selector(dec) + 0.2 * _invariant_selector(dec, 2)
    base = classify_functional(
        dec, s, tol=1e-8, feasible_set=fs, enclosing_radius=radius
    )
    for c in (1e-3, 7.0, 1e4):
        res = classify_functional(
            dec, c * s, tol=1e-8, feasible_set=fs, enclosing_radius=radius
        )
        assert res.status is base.status
        assert res.residual == pytest.approx(base.residual, rel=1e-10)
        assert res.bound is not None and base.bound is not None
        assert res.bound.deviation_bound == pytest.approx(
            c * base.bound.deviation_bound, rel=1e-10
        )


def test_units_rescaling_via_characteristic_scales() -> None:
    k = 1000.0
    d0 = _dec()
    d1 = _dec(CharacteristicScales(force=100.0 * k, moment=5.0 * k))
    s = _variable_selector(d0)
    r0 = classify_functional(d0, s, tol=1e-8)
    r1 = classify_functional(d1, s, tol=1e-8)
    assert r1.residual == pytest.approx(r0.residual, rel=1e-8)
    assert r1.annihilation_norm == pytest.approx(k * r0.annihilation_norm, rel=1e-8)
    inv = classify_functional(d1, _invariant_selector(d1), tol=1e-8)
    assert inv.status is FunctionalStatus.NUMERICALLY_INVARIANT


# ------------------------------------------------------------ bounded bound
def test_bound_contains_every_hit_and_run_deviation() -> None:
    dec = _dec()
    fs = _box_set(dec)
    radius = float(np.sqrt(dec.nullity))
    s = _variable_selector(dec) + 0.1 * _invariant_selector(dec, 3)
    res = classify_functional(
        dec, s, tol=1e-8, feasible_set=fs, enclosing_radius=radius
    )
    assert res.bound is not None
    assert res.bound.deviation_bound == pytest.approx(
        res.annihilation_norm * radius, rel=1e-12
    )
    assert res.bound.width_bound == pytest.approx(
        2.0 * res.bound.deviation_bound, rel=1e-12
    )
    dom = TrajectoryDomain([fs], np.ones((1, 1)))
    draws = hit_and_run(dom, np.zeros(dom.dimension), n_draws=200, seed=5)
    x0 = dec.x(np.zeros(dec.nullity))
    dev = np.array([s @ (dec.x(z) - x0) for z in draws.coefficients])
    assert np.all(np.abs(dev) <= res.bound.deviation_bound + 1e-12)
    assert dev.max() - dev.min() <= res.bound.width_bound + 1e-12


def test_bound_absent_without_feasible_set_and_radius() -> None:
    dec = _dec()
    assert classify_functional(dec, _variable_selector(dec), tol=1e-8).bound is None


def test_bound_requires_both_set_and_radius() -> None:
    dec = _dec()
    s = _variable_selector(dec)
    with pytest.raises(ValueError, match="together"):
        classify_functional(dec, s, tol=1e-8, enclosing_radius=2.0)
    with pytest.raises(ValueError, match="together"):
        classify_functional(dec, s, tol=1e-8, feasible_set=_box_set(dec))


def test_unbounded_set_is_rejected() -> None:
    dec = _dec()
    open_set = dataclasses.replace(
        FeasibleSet.from_rows(dec.nullity, ()), decomposition=dec
    )
    with pytest.raises(ValueError, match="bounded"):
        classify_functional(
            dec,
            _variable_selector(dec),
            tol=1e-8,
            feasible_set=open_set,
            enclosing_radius=10.0,
        )


def test_radius_smaller_than_the_set_is_rejected() -> None:
    dec = _dec()
    with pytest.raises(ValueError, match="enclos"):
        classify_functional(
            dec,
            _variable_selector(dec),
            tol=1e-8,
            feasible_set=_box_set(dec, half=2.0),
            enclosing_radius=1.0,
        )


# --------------------------------------------------------------- contracts
@pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_tolerance_raises(tol) -> None:
    dec = _dec()
    with pytest.raises(ValueError, match="tol"):
        classify_functional(dec, _variable_selector(dec), tol=tol)


def test_invalid_inputs_raise() -> None:
    dec = _dec()
    with pytest.raises(TypeError):
        classify_functional(object(), np.zeros(3), tol=1e-8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selector"):
        classify_functional(dec, np.zeros(dec.dim_x + 1), tol=1e-8)
    bad = np.zeros(dec.dim_x)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        classify_functional(dec, bad, tol=1e-8)


def test_decomposition_without_variable_scale_is_rejected() -> None:
    dec = dataclasses.replace(_dec(), variable_scale=None)
    with pytest.raises(ValueError, match="variable_scale"):
        classify_functional(dec, np.ones(dec.dim_x), tol=1e-8)


# ------------------------------------------------------------------ records
def test_record_tags_invariant_with_conventions_and_no_hypotheses() -> None:
    dec = _dec()
    res = classify_functional(dec, _invariant_selector(dec), tol=1e-8)
    rec = functional_record(
        res, [1.5], units="N m", model=MODEL, source=SOURCE, hypotheses=("H-X",)
    )
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.hypotheses == ()
    assert rec.conventions


def test_record_tags_family_dependent_with_hypotheses() -> None:
    dec = _dec()
    res = classify_functional(dec, _variable_selector(dec), tol=1e-8)
    rec = functional_record(
        res, [1.5], units="N", model=MODEL, source=SOURCE, hypotheses=("H-X",)
    )
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert rec.hypotheses == ("H-X",)
    with pytest.raises(ValueError, match="hypothes"):
        functional_record(res, [1.5], units="N", model=MODEL, source=SOURCE)


# -------------------------------------------------------------- attribution
def _partition(dec, k: int = 2):
    """Coordinate partition of ``x`` into ``k`` sources: diagonal 0/1 matrices."""
    owner = np.arange(dec.dim_x) % k
    return {f"src{j}": np.diag((owner == j).astype(float)) for j in range(k)}


def _task_map(dec, rows: int = 2) -> np.ndarray:
    """Rows in ``range(A^T)`` (a total induced acceleration is balance-determined)."""
    return np.stack([_invariant_selector(dec, 10 + r) for r in range(rows)])


def _sources(dec, task_map, partition):
    return {name: task_map @ e for name, e in partition.items()}


def test_total_attribution_is_invariant_across_the_fibre() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    fs = _box_set(dec)
    factory = attribution_quantities(
        ("a0", "a1"), sel, partition=part, units="m/s^2", scale=1.0
    )
    quantities = factory(fs)
    assert [q.name for q in quantities] == [
        f"{t}|{s}" for s in sel for t in ("a0", "a1")
    ]
    coeff = sum(q.coeff for q in quantities[0::2])
    assert np.max(np.abs(coeff)) < TOL
    for t in range(2):
        total = classify_functional(dec, tmap[t], tol=1e-8)
        assert total.status is FunctionalStatus.NUMERICALLY_INVARIANT
    z = np.random.default_rng(0).uniform(-1, 1, dec.nullity)
    for t in range(2):
        value = sum(q.const + q.coeff @ z for q in quantities[t::2])
        assert value == pytest.approx(tmap[t] @ dec.x(z), abs=1e-9)


def test_individual_sources_are_family_dependent() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    status = [classify_functional(dec, sel[name][0], tol=1e-8).status for name in sel]
    assert FunctionalStatus.FAMILY_DEPENDENT in status


def test_attribution_envelopes_contain_hit_and_run_draws() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    fs = _box_set(dec)
    factory = attribution_quantities(
        ("a0", "a1"), sel, partition=part, units="m/s^2", scale=1.0
    )
    series = static_envelopes([fs], factory)
    assert all(s is BoundStatus.BOUNDED for s in series.status_lower[0])
    dom = TrajectoryDomain([fs], np.ones((1, 1)))
    draws = hit_and_run(dom, np.zeros(dom.dimension), n_draws=200, seed=11)
    quantities = factory(fs)
    for j, q in enumerate(quantities):
        vals = np.array([q.const + q.coeff @ z for z in draws.coefficients])
        assert vals.min() >= series.lower[0, j] - 1e-9
        assert vals.max() <= series.upper[0, j] + 1e-9


def test_task_map_consistency_is_checked_when_given() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    attribution_quantities(
        ("a0", "a1"), sel, partition=part, task_map=tmap, units="m/s^2", scale=1.0
    )
    with pytest.raises(ValueError, match="task_map"):
        attribution_quantities(
            ("a0", "a1"),
            sel,
            partition=part,
            task_map=2.0 * tmap,
            units="m/s^2",
            scale=1.0,
        )


def test_partition_not_summing_to_identity_raises() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    broken = dict(part)
    broken["src0"] = 0.5 * broken["src0"]
    with pytest.raises(ValueError, match="identity"):
        attribution_quantities(
            ("a0", "a1"), sel, partition=broken, units="m/s^2", scale=1.0
        )


def test_attribution_argument_contracts() -> None:
    dec = _dec()
    part, tmap = _partition(dec), _task_map(dec)
    sel = _sources(dec, tmap, part)
    with pytest.raises(ValueError, match="source"):
        attribution_quantities(
            ("a0", "a1"), sel, partition={"other": np.eye(dec.dim_x)},
            units="m/s^2", scale=1.0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="names"):
        attribution_quantities(("a0",), sel, partition=part, units="m/s^2", scale=1.0)
    with pytest.raises(ValueError, match="at least"):
        attribution_quantities((), {}, partition={}, units="m/s^2", scale=1.0)
    with pytest.raises(TypeError):
        attribution_quantities(
            ("a0", "a1"),
            [tmap],
            partition=part,
            units="m/s^2",
            scale=1.0,  # type: ignore[arg-type]
        )
