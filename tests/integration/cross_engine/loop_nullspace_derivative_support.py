"""Support for the NSX-98 derivative conformance rows (#12727).

The NSX-56 harness compares the canonical arrays ``M, h, J_c, Jdot_v``.  This
module extends it to the *sensitivities* of the rigid closed-loop solve
(``d a/d tau``, ``d a/d xi``, ``d a/d eta`` and the same three for the reaction
``lambda``) that every engine with a derivative capability of at least
``PARTIAL`` returns from ``constrained_acceleration_derivatives``.

Every sensitivity is compared as a *basis-independent operator* in the
reference coordinate order: ``D P^T`` for the state blocks (``P`` is the
orthonormal basis of ``ker J_c`` the engine used), the plain matrix for the load
block.  Rows of the acceleration blocks and the columns of every block are
permuted by coordinate name.

Variance conformance is a *first-order* propagation of the sensitivities under a
**declared** isotropic unit noise (load noise on the actuated coordinates; state
noise on the tangent coordinates).  It is the covariance map the NSX-80
variability chapter builds on, not a measured noise level, and it uses no
face-angle or allocation module (those arrive with later children).

Agreement between engines is software conformance, not physical validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.adapter_derivatives import AdapterClosedLoopModel
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
)
from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    NamedSnapshot,
    kkt_bound,
    kkt_matrix,
)

DERIV_CHECK_NAME = "closed_loop_derivative_conformance"
SENSITIVITY_METRICS = (
    "d_acc_d_tau",
    "d_acc_d_xi",
    "d_acc_d_eta",
    "d_lambda_d_tau",
    "d_lambda_d_xi",
    "d_lambda_d_eta",
)
VARIANCE_METRICS = (
    "acc_load_variance",
    "lambda_load_variance",
    "acc_state_variance",
    "lambda_state_variance",
)
DERIV_METRICS = SENSITIVITY_METRICS + VARIANCE_METRICS
#: Declared isotropic noise amplitude (unit load / unit tangent coordinate).
DECLARED_SIGMA = 1.0


@dataclass(frozen=True)
class NamedDerivatives:
    """Basis-independent operators of one engine, in the reference order."""

    sensitivities: dict[str, np.ndarray]
    method: str
    acc_load_noise: np.ndarray  # actuated selector (diagonal, 0/1)


def named_derivatives(
    res: ClosedLoopDerivatives,
    own_order: tuple[str, ...],
    reference_order: tuple[str, ...],
    actuated: frozenset[str],
) -> NamedDerivatives:
    """Operators of ``res`` permuted into ``reference_order`` by name.

    Raises ``ValueError`` when the coordinate vocabularies differ.
    """
    if sorted(own_order) != sorted(reference_order):
        raise ValueError("derivatives must share one coordinate vocabulary")
    require_hard_kkt(res)
    perm = np.array([own_order.index(n) for n in reference_order])
    ix = np.ix_(perm, perm)
    basis_t = res.basis.T
    ops = {
        "d_acc_d_tau": res.d_acc_d_tau[ix],
        "d_acc_d_xi": (res.d_acc_d_xi @ basis_t)[ix],
        "d_acc_d_eta": (res.d_acc_d_eta @ basis_t)[ix],
        "d_lambda_d_tau": res.d_lambda_d_tau[:, perm],
        "d_lambda_d_xi": (res.d_lambda_d_xi @ basis_t)[:, perm],
        "d_lambda_d_eta": (res.d_lambda_d_eta @ basis_t)[:, perm],
    }
    selector = np.diag([1.0 if n in actuated else 0.0 for n in reference_order])
    return NamedDerivatives(ops, res.method.value, selector)


def covariance_operators(named: NamedDerivatives) -> dict[str, np.ndarray]:
    """First-order covariances under the declared unit noise (see module doc).

    ``acc_load_variance = s^2 A S A^T`` with ``S`` the actuated selector;
    ``acc_state_variance = s^2 (Dxi Dxi^T + Deta Deta^T)`` (``P`` orthonormal, so
    the operator form needs no basis).  The ``lambda_*`` entries use the
    reaction blocks.  Postcondition: every result is symmetric PSD.
    """
    s2 = DECLARED_SIGMA**2
    ops, sel = named.sensitivities, named.acc_load_noise
    out = {}
    for tag, key in (("acc", "d_acc"), ("lambda", "d_lambda")):
        load = ops[f"{key}_d_tau"]
        out[f"{tag}_load_variance"] = s2 * load @ sel @ load.T
        xi, eta = ops[f"{key}_d_xi"], ops[f"{key}_d_eta"]
        out[f"{tag}_state_variance"] = s2 * (xi @ xi.T + eta @ eta.T)
    return out


def operator_bound(delta: float, norm: float) -> float:
    """``|A A^T - B B^T|_2`` bound from ``|A - B|_2 <= delta``, ``|B|_2 = norm``.

    ``A = B + E``: ``A A^T - B B^T = E B^T + B E^T + E E^T`` so the norm is at
    most ``delta (2 norm + delta)``.
    """
    return delta * (2.0 * norm + delta)


#: Provenance fragments of derivatives of MuJoCo's *soft* constraint stepping.
SOFT_SCHEME_MARKERS = ("soft", "mjd_transition", "solref", "solimp")


def require_hard_kkt(res: ClosedLoopDerivatives) -> None:
    """Refuse derivatives that are not those of the rigid hard-KKT solve.

    Soft (compliant) and hard dynamics are different systems; comparing them
    would report their modelling difference as an engine disagreement.  The
    scheme tag is the only provenance a result carries, so a soft marker in it
    is a hard failure (``ValueError``).
    """
    scheme = res.scheme.lower()
    for marker in SOFT_SCHEME_MARKERS:
        if marker in scheme:
            raise ValueError(
                f"derivatives with scheme {res.scheme!r} are not hard-KKT "
                "derivatives; soft and hard dynamics must not be mixed"
            )


def finite_difference_resolution(
    adapter: Any, basis: np.ndarray, state: tuple[np.ndarray, ...], step: float
) -> dict[str, float]:
    """Spread of the NSX-84 finite-difference state operators under step x0.5, x2.

    Evaluated on the adapter's own model in its own order and tangent ``basis``.
    The spread (spectral norm of ``D P^T`` differences against the nominal
    step) is the smallest difference between two exact routes that a finite
    difference reference can resolve (NSX-86's criterion).  Keys are the four
    state metrics.
    """
    q, v, tau = state
    model = AdapterClosedLoopModel(adapter)

    def run(h: float) -> ClosedLoopDerivatives:
        return tl.FiniteDifferenceDerivatives(
            model,
            tl.EuclideanManifold(model.nv),
            step=h,
            basis_field=lambda _q: basis,
        ).constrained_acceleration_derivatives(q, v, tau)

    nominal = run(step)
    others = [run(step * f) for f in (0.5, 2.0)]
    out = {}
    for name in ("d_acc_d_xi", "d_acc_d_eta", "d_lambda_d_xi", "d_lambda_d_eta"):
        want = getattr(nominal, name) @ basis.T
        out[name] = max(
            float(np.linalg.norm(getattr(o, name) @ basis.T - want, 2)) for o in others
        )
    return out


def sensitivity_bounds(
    ref: NamedSnapshot,
    other: NamedSnapshot,
    tau: np.ndarray,
    *,
    tangent_dim: int,
    step: float | None,
    resolution: dict[str, float] | None,
) -> dict[str, float]:
    """Derived spectral-norm bound on ``|D_other - D_ref|_2`` per sensitivity.

    * Load blocks (every engine): the linear response ``K^-1 [e_j; 0]`` differs
      by ``kkt_bound`` with a zero right-hand side per column, so by
      ``sqrt(nv)`` times that in the spectral norm.
    * State blocks of a finite-difference engine (``step`` given): a central
      difference of two solutions, each within ``kkt_bound`` of the reference
      solution, differs by at most ``kkt_bound / step`` per column
      (``sqrt(k)`` columns) -- the NSX-88 derivation.
    * State blocks of an exact engine (``step is None``): the NSX-56 arrays do
      not bound the differentials, so the bound is the finite-difference
      resolution (``resolution``, required) -- the NSX-86 derivation.

    Raises ``ValueError`` if an exact engine is given without ``resolution``.
    """
    n, m = ref.M.shape[0], ref.J_c.shape[0]
    norm_x = float(np.linalg.norm(np.linalg.solve(kkt_matrix(ref), _rhs(ref, tau))))
    response = np.linalg.solve(
        kkt_matrix(ref), np.vstack([np.eye(n), np.zeros((m, n))])
    )
    load = math.sqrt(n) * kkt_bound(ref, other, float(np.linalg.norm(response, 2)), 0.0)
    bounds = dict.fromkeys(("d_acc_d_tau", "d_lambda_d_tau"), load)
    if step is not None:
        column = math.sqrt(tangent_dim) * kkt_bound(ref, other, norm_x, 1.0) / step
        for k in ("d_acc_d_xi", "d_acc_d_eta", "d_lambda_d_xi", "d_lambda_d_eta"):
            bounds[k] = column
    else:
        if resolution is None:
            raise ValueError("an exact engine needs the finite-difference resolution")
        bounds.update(resolution)
    return bounds


def _rhs(snap: NamedSnapshot, tau: np.ndarray) -> np.ndarray:
    return np.concatenate((tau - snap.h, -snap.Jdot_v))


def variance_bounds(
    bounds: dict[str, float], ref: NamedDerivatives
) -> dict[str, float]:
    """Spectral bounds on the covariance differences from the operator bounds.

    ``|A S A^T - B S B^T| <= d (2 |B| + d)`` with ``S`` a 0/1 selector (see
    :func:`operator_bound`); the state covariance sums the ``xi`` and ``eta``
    contributions.
    """
    ops = ref.sensitivities
    out = {}
    for tag, key in (("acc", "d_acc"), ("lambda", "d_lambda")):
        out[f"{tag}_load_variance"] = operator_bound(
            bounds[f"{key}_d_tau"], float(np.linalg.norm(ops[f"{key}_d_tau"], 2))
        )
        out[f"{tag}_state_variance"] = sum(
            operator_bound(
                bounds[f"{key}_d_{blk}"],
                float(np.linalg.norm(ops[f"{key}_d_{blk}"], 2)),
            )
            for blk in ("xi", "eta")
        )
    return out


def scaled_bound(bound: float, reference: np.ndarray) -> float:
    """``bound / max(1, max|reference|)``: the bound in ``scaled_error`` units."""
    return bound / max(1.0, float(np.max(np.abs(reference))))


def round_up(value: float, digits: int = 2) -> float:
    """Round ``value`` up to ``digits`` significant digits (registry precision)."""
    if value <= 0.0:
        raise ValueError("value must be positive")
    exp = math.floor(math.log10(value)) - digits + 1
    return math.ceil(value / 10**exp) * 10**exp
