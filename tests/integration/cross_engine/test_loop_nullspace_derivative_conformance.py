"""NSX-98: two-track conformance of closed-loop sensitivities and variance (#12727).

Extends the NSX-56 harness (``test_loop_nullspace_conformance.py``) from the
snapshot arrays to the *derivatives* of the rigid closed-loop solve.  Every
engine whose derivative capability is at least ``PARTIAL`` (Pinocchio ANALYTIC
reference; Drake AUTODIFF; MuJoCo, MyoSuite, OpenSim finite difference) is
compared with the Pinocchio reference at the same projected state, load and
closure tangent space.

* **Track N** compares ``d a/d tau``, ``d a/d xi``, ``d a/d eta`` and the three
  ``d lambda`` blocks as basis-independent operators.  The tolerance of each
  engine-pair metric is the entry registered in ``divergence_registry.yaml``
  (check ``closed_loop_derivative_conformance``), written from the bound
  derived in ``loop_nullspace_derivative_support.sensitivity_bounds``.
* **Track S** compares the first-order covariance maps those sensitivities
  induce under a *declared* unit noise (scientific comparison, not a measured
  noise level), with the bound propagated from the Track N bounds.

Engines without the capability are reported ``NONE`` with the registry reason.
Mixing soft (compliant) and hard (rigid KKT) dynamics is refused.  Agreement
between engines is software conformance, not physical validation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
)
from src.shared.python.engine_core.cross_engine_validator import DivergenceRegistry
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
    DerivativeMethod,
    default_registry,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    create_engine_force_adapter,
)
from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    SPEC_PATH,
    NamedSnapshot,
    build_snapshot,
    scaled_error,
    seeded_effort,
    to_reference_order,
)
from tests.integration.cross_engine.loop_nullspace_derivative_support import (
    DERIV_CHECK_NAME,
    DERIV_METRICS,
    SENSITIVITY_METRICS,
    VARIANCE_METRICS,
    NamedDerivatives,
    covariance_operators,
    finite_difference_resolution,
    named_derivatives,
    require_hard_kkt,
    scaled_bound,
    sensitivity_bounds,
    variance_bounds,
)
from tests.integration.cross_engine.test_loop_nullspace_conformance import (  # noqa: F401
    OTHERS,
    REFERENCE,
    SDK,
    plant,
    projected,
    reference_adapter,
)

pytestmark = pytest.mark.integration
# ruff: noqa: F811 - the NSX-56 module fixtures are imported by name

REGISTRY = DivergenceRegistry.from_yaml(
    Path(__file__).with_name("divergence_registry.yaml")
)
#: Reference finite-difference step for the exact-engine resolution (NSX-86).
RESOLUTION_STEP = 1e-3


@dataclasses.dataclass(frozen=True)
class DerivativePair:
    """Reference and other-engine operators with their derived bounds."""

    engine: EngineType
    ref: NamedDerivatives
    other: NamedDerivatives
    result: ClosedLoopDerivatives
    bounds: dict[str, float]
    variance_bounds: dict[str, float]
    order: tuple[str, ...]
    own_order: tuple[str, ...]
    actuated: frozenset[str]
    ref_snap: NamedSnapshot
    other_snap: NamedSnapshot
    tau: np.ndarray
    tangent_dim: int


def _derivatives(adapter, named_q, named_v, tau, order, actuated):
    own = tuple(adapter.coordinate_order)
    q = np.array([named_q[n] for n in own])
    v = np.array([named_v[n] for n in own])
    t = np.array([tau[order.index(n)] for n in own])
    res = adapter.constrained_acceleration_derivatives(q, v, t)
    return res, named_derivatives(res, own, order, actuated), own


@pytest.fixture(scope="module", params=OTHERS)
def dpair(request, reference_adapter, projected):
    """``DerivativePair`` of one engine against Pinocchio at one projected state."""
    engine = request.param
    pytest.importorskip(SDK[engine])
    adapter = create_engine_force_adapter(engine, SPEC_PATH, allow_synthetic=False)
    named_q, named_v, frame = projected
    order = tuple(reference_adapter.coordinate_order)
    source = f"dynamics_record frame {frame}"
    ref_snap = to_reference_order(
        build_snapshot(reference_adapter, named_q, named_v, source), order
    )
    other_snap = to_reference_order(
        build_snapshot(adapter, named_q, named_v, source), order
    )
    tau = seeded_effort(order, ref_snap.actuated)
    ref_res, ref, _ = _derivatives(
        reference_adapter, named_q, named_v, tau, order, ref_snap.actuated
    )
    res, other, own = _derivatives(
        adapter, named_q, named_v, tau, order, ref_snap.actuated
    )
    resolution = None
    if res.step is None:
        q = np.array([named_q[n] for n in order])
        v = np.array([named_v[n] for n in order])
        resolution = finite_difference_resolution(
            reference_adapter, ref_res.basis, (q, v, tau), RESOLUTION_STEP
        )
    bounds = sensitivity_bounds(
        ref_snap,
        other_snap,
        tau,
        tangent_dim=ref_res.tangent_dimension,
        step=res.step,
        resolution=resolution,
    )
    return DerivativePair(
        engine,
        ref,
        other,
        res,
        bounds,
        variance_bounds(bounds, ref),
        order,
        own,
        ref_snap.actuated,
        ref_snap,
        other_snap,
        tau,
        ref_res.tangent_dimension,
    )


def _entry(engine: EngineType, metric: str):
    entry = REGISTRY.find(
        check_name=DERIV_CHECK_NAME,
        metric_name=metric,
        engine1=REFERENCE.value,
        engine2=engine.value,
    )
    assert entry is not None, f"no registry entry: {engine.value} {metric}"
    return entry


def _errors(pair: DerivativePair) -> dict[str, float]:
    errs = {
        k: scaled_error(pair.other.sensitivities[k], pair.ref.sensitivities[k])
        for k in SENSITIVITY_METRICS
    }
    cov_ref, cov_other = (
        covariance_operators(pair.ref),
        covariance_operators(pair.other),
    )
    errs.update({k: scaled_error(cov_other[k], cov_ref[k]) for k in VARIANCE_METRICS})
    return errs


# -- Registry ---------------------------------------------------------------


def test_every_pair_metric_is_registered_with_a_rationale() -> None:
    for param in OTHERS:
        for metric in DERIV_METRICS:
            entry = _entry(param.values[0], metric)
            assert entry.tolerance > 0 and entry.rationale


# -- Track N: sensitivities --------------------------------------------------


def test_track_n_sensitivities_agree_within_the_registered_tolerance(dpair) -> None:
    errs = _errors(dpair)
    for metric in SENSITIVITY_METRICS:
        tol = _entry(dpair.engine, metric).tolerance
        assert errs[metric] <= tol, (
            f"{dpair.engine.value} {metric}: {errs[metric]:.3g} exceeds {tol:.3g}"
        )


def test_track_n_difference_is_inside_the_derived_bound(dpair) -> None:
    """The measured spectral difference stays inside the per-run derived bound."""
    for metric in SENSITIVITY_METRICS:
        diff = dpair.other.sensitivities[metric] - dpair.ref.sensitivities[metric]
        assert float(np.linalg.norm(diff, 2)) <= dpair.bounds[metric], (
            f"{dpair.engine.value} {metric}: measured difference exceeds the "
            "derived bound -- register it as a divergence, do not widen"
        )


def test_registered_tolerances_cover_the_derived_bound(dpair) -> None:
    """A registry row may not be tighter than the derivation it records."""
    scaled = {
        k: scaled_bound(dpair.bounds[k], dpair.ref.sensitivities[k])
        for k in SENSITIVITY_METRICS
    }
    scaled.update(
        {
            k: scaled_bound(
                dpair.variance_bounds[k], covariance_operators(dpair.ref)[k]
            )
            for k in VARIANCE_METRICS
        }
    )
    for metric, derived in scaled.items():
        assert _entry(dpair.engine, metric).tolerance >= derived, metric


def test_track_n_is_not_vacuous(dpair) -> None:
    ops = dpair.ref.sensitivities
    assert float(np.abs(ops["d_acc_d_tau"]).max()) > 1.0
    assert float(np.abs(ops["d_acc_d_xi"]).max()) > 1e-3
    assert float(np.abs(ops["d_lambda_d_tau"]).max()) > 1e-3
    assert dpair.result.method in (
        DerivativeMethod.AUTODIFF,
        DerivativeMethod.FINITE_DIFFERENCE,
    )


# -- Track S: variance ---------------------------------------------------------


def test_track_s_variance_agrees_within_the_registered_tolerance(dpair) -> None:
    errs = _errors(dpair)
    for metric in VARIANCE_METRICS:
        assert errs[metric] <= _entry(dpair.engine, metric).tolerance, (
            f"{dpair.engine.value} {metric}: {errs[metric]:.3g}"
        )


def test_track_s_variance_difference_is_inside_the_propagated_bound(dpair) -> None:
    cov_ref, cov_other = (
        covariance_operators(dpair.ref),
        covariance_operators(dpair.other),
    )
    for metric in VARIANCE_METRICS:
        diff = float(np.linalg.norm(cov_other[metric] - cov_ref[metric], 2))
        assert diff <= dpair.variance_bounds[metric], metric


def test_track_s_covariances_are_symmetric_positive_semidefinite(dpair) -> None:
    for side in (dpair.ref, dpair.other):
        for metric, cov in covariance_operators(side).items():
            np.testing.assert_allclose(
                cov, cov.T, rtol=1e-9, atol=1e-9 * np.abs(cov).max()
            )
            assert float(np.linalg.eigvalsh(0.5 * (cov + cov.T)).min()) >= -1e-9 * max(
                1.0, float(np.abs(cov).max())
            ), metric


# -- Harness self-checks: planted errors must fail ------------------------------


def _plant(named: NamedDerivatives, **changes) -> NamedDerivatives:
    ops = dict(named.sensitivities)
    ops.update(changes)
    return dataclasses.replace(named, sensitivities=ops)


def _fails_registry(pair: DerivativePair, bad: NamedDerivatives, metric: str) -> bool:
    err = scaled_error(bad.sensitivities[metric], pair.ref.sensitivities[metric])
    return err > _entry(pair.engine, metric).tolerance


def test_planted_sign_error_fails_track_n(dpair) -> None:
    ops = dpair.other.sensitivities
    for metric in ("d_acc_d_tau", "d_acc_d_xi", "d_lambda_d_tau", "d_lambda_d_xi"):
        assert _fails_registry(
            dpair, _plant(dpair.other, **{metric: -ops[metric]}), metric
        )


def test_planted_wrench_frame_error_fails_track_n(dpair) -> None:
    """Swapping the force and torque halves of the grip wrench is a frame error."""
    swap = [3, 4, 5, 0, 1, 2]
    ops = dpair.other.sensitivities
    bad = _plant(
        dpair.other,
        **{
            k: ops[k][swap]
            for k in ("d_lambda_d_tau", "d_lambda_d_xi", "d_lambda_d_eta")
        },
    )
    for metric in ("d_lambda_d_tau", "d_lambda_d_xi"):
        assert _fails_registry(dpair, bad, metric)


def test_planted_coordinate_permutation_error_fails_track_n(dpair) -> None:
    """A wrong coordinate order (two names swapped) is caught, not absorbed."""
    own = list(dpair.own_order)
    own[0], own[1] = own[1], own[0]
    wrong = named_derivatives(dpair.result, tuple(own), dpair.order, dpair.actuated)
    assert _fails_registry(dpair, wrong, "d_acc_d_tau")


def test_planted_error_fails_track_s(dpair) -> None:
    bad = _plant(
        dpair.other,
        d_acc_d_tau=dpair.other.sensitivities["d_acc_d_tau"] * 1.001,
    )
    cov_ref, cov_bad = covariance_operators(dpair.ref), covariance_operators(bad)
    err = scaled_error(cov_bad["acc_load_variance"], cov_ref["acc_load_variance"])
    assert err > _entry(dpair.engine, "acc_load_variance").tolerance


def test_soft_and_hard_dynamics_cannot_be_mixed(dpair) -> None:
    soft = dataclasses.replace(dpair.result, scheme="mjd_transitionFD_soft")
    with pytest.raises(ValueError, match="soft and hard"):
        require_hard_kkt(soft)
    with pytest.raises(ValueError, match="soft and hard"):
        named_derivatives(soft, dpair.own_order, dpair.order, dpair.actuated)
    require_hard_kkt(dpair.result)  # the genuine result is hard-KKT


def test_vocabulary_mismatch_is_refused(dpair) -> None:
    with pytest.raises(ValueError, match="coordinate vocabulary"):
        named_derivatives(dpair.result, dpair.own_order, ("nope",), dpair.actuated)


def test_exact_engine_needs_a_resolution(dpair) -> None:
    with pytest.raises(ValueError, match="resolution"):
        sensitivity_bounds(
            dpair.ref_snap,
            dpair.other_snap,
            dpair.tau,
            tangent_dim=dpair.tangent_dim,
            step=None,
            resolution=None,
        )


# -- Capability matrix ---------------------------------------------------------

EXPECTED = {
    EngineType.PINOCCHIO: ("pinocchio", DerivativeMethod.ANALYTIC, "FULL"),
    EngineType.DRAKE: ("drake", DerivativeMethod.AUTODIFF, "FULL"),
    EngineType.MUJOCO: ("mujoco", DerivativeMethod.FINITE_DIFFERENCE, "PARTIAL"),
    EngineType.MYOSUITE: ("myosuite", DerivativeMethod.FINITE_DIFFERENCE, "PARTIAL"),
    EngineType.OPENSIM: ("opensim", DerivativeMethod.FINITE_DIFFERENCE, "PARTIAL"),
}


@pytest.mark.parametrize("engine", list(EngineType), ids=lambda e: e.value)
def test_every_engine_is_reported_with_its_capability(engine) -> None:
    """Capable engines carry registry rows; the rest report NONE with the reason."""
    reg = default_registry()
    decl = reg.require(engine.value, minimum=CapabilityLevel.NONE)
    if engine in EXPECTED:
        _, method, level = EXPECTED[engine]
        assert decl.method is method and decl.level.name == level
        if engine is not REFERENCE:
            for metric in DERIV_METRICS:
                assert _entry(engine, metric).rationale
    else:
        assert decl.method is DerivativeMethod.NONE
        assert decl.level is CapabilityLevel.NONE
        assert decl.reason.strip()  # the registry reason is reported, not silent


def test_adapter_declarations_match_the_registry() -> None:
    reg = default_registry()
    for engine in (EngineType.PINOCCHIO, EngineType.MUJOCO):
        pytest.importorskip(SDK.get(engine, "pinocchio"))
        adapter = create_engine_force_adapter(engine, SPEC_PATH, allow_synthetic=False)
        level = adapter.capability(Capability.CLOSED_LOOP_DERIVATIVES)
        assert level is reg.require(engine.value, minimum=CapabilityLevel.NONE).level
