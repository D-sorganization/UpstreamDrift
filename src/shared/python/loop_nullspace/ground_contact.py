"""Ground contact as a second unilateral null space, with GRF observations (NSX-60).

Specification section "Lower-Body Loop".  With rigid unilateral feet the load
balance is ``A x = b``, ``A = [S^T J_c^T J_g^T]``, ``x = (tau, lambda, f_g)``,
where ``f_g`` stacks one three-vector per active contact point and ``J_g``
holds the world-axis point Jacobians.  The decomposition (NSX-15), the feasible
set (NSX-21) and the envelopes (NSX-22) already consume ``J_g``; this module
supplies what is specific to the feet:

* the *active set*: which candidate contact points carry a load (a point whose
  signed gap to the ground is within a declared tolerance is a rigid contact);
* Coulomb cones over each contact's load entries, reusing ``FrictionCone``;
* optional *ground-reaction observations* ``E f_g = g`` with a declared
  1-sigma uncertainty per row (hypothesis ``H-GRF-OBSERVATION``).

An observation never changes the equality balance.  In the zero-uncertainty
limit it adds ``rank(E_x N)`` equality rows and removes that many null
directions (:func:`restrict_to_observation`); with uncertainty it enters the
feasible set as the two-sided band ``|E f_g - g| <= k sigma``
(:func:`observation_rows`).  :func:`observation_accounting` reports how many
null directions the data can see and how consistent it is with the balance.

Non-claims: a rigid unilateral contact is the only contact model (no
compliance, no slip dynamics); a friction coefficient is a declared hypothesis,
not a measurement; an observation of the net ground wrench is already fixed by
the unactuated rows and adds nothing; contact loads are not physiological
quantities.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import RANK_RTOL, Decomposition
from src.shared.python.loop_nullspace.evidence import (
    ROTATION_ATOL,
    Array,
    FootContact,
    check_shape,
    readonly_float,
)
from src.shared.python.loop_nullspace.feasible_rows import (
    GRF_ID,
    ContactCone,
    FeasibleRow,
)
from src.shared.python.motion_matching.force_nullspace.constraints import (
    FrictionCone,
)

__all__ = [
    "DEFAULT_COVERAGE",
    "GRF_ID",
    "ContactCandidate",
    "GroundObservation",
    "ObservationAccounting",
    "ObservationInconsistentError",
    "contact_frame",
    "foot_friction_cones",
    "observation_accounting",
    "observation_rows",
    "restrict_to_observation",
    "select_active_contacts",
]

#: Coverage factor ``k`` of the observation band ``|E f - g| <= k sigma``.
DEFAULT_COVERAGE = 3.0
_AXES = "xyz"


class ObservationInconsistentError(ValueError):
    """The observation cannot be met inside its declared uncertainty band.

    ``max_normalised_residual`` is the largest ``|E f - g| / sigma`` left at the
    weighted least-squares member of the balance family.  Never repaired.
    """

    def __init__(self, message: str, max_normalised_residual: float) -> None:
        super().__init__(message)
        self.max_normalised_residual = float(max_normalised_residual)


# ------------------------------------------------------------------ frames --
def contact_frame(normal: ArrayLike) -> Array:
    """Right-handed frame with columns ``(t1, t2, n)`` for a unit ``normal``.

    ``t1`` is the world x axis (the y axis when ``normal`` is near x) projected
    onto the tangent plane, ``t2 = n x t1``.  Raises ``ValueError`` unless
    ``normal`` is a unit 3-vector.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise ValueError(f"normal must have shape (3,) and be finite, got {n.shape}")
    if abs(float(np.linalg.norm(n)) - 1.0) > ROTATION_ATOL:
        raise ValueError("normal must be a unit vector")
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = ref - float(ref @ n) * n
    t1 /= float(np.linalg.norm(t1))
    return np.column_stack([t1, np.cross(n, t1), n])


# ------------------------------------------------------------- active set ---
@dataclass(frozen=True)
class ContactCandidate:
    """A potential ground contact: its geometry and signed gap to the ground.

    ``gap_m`` is the distance from the contact surface to the ground plane
    along its normal (negative when penetrating).  ``contact.position`` is the
    point the Jacobian rows are evaluated at.
    """

    contact: FootContact
    gap_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.gap_m):
            raise ValueError(f"gap_m must be finite, got {self.gap_m}")


def select_active_contacts(
    candidates: Sequence[ContactCandidate],
    jacobian: ArrayLike,
    *,
    gap_tol_m: float,
) -> tuple[tuple[FootContact, ...], Array | None]:
    """Rigid contacts of ``candidates`` and their stacked ``J_g`` rows.

    A candidate is a rigid contact when ``gap_m <= gap_tol_m``; it contributes
    three Jacobian rows (rows ``3k..3k+2`` of ``jacobian``) in candidate
    order.  A looser tolerance only adds unilateral freedom; it never removes
    a load the balance may need.  Returns ``((), None)`` when nothing touches.

    Preconditions: ``gap_tol_m`` finite and non-negative; ``jacobian`` has
    ``3 * len(candidates)`` rows (``ValueError``).  Postcondition: the rows are
    read-only and ``len(rows) == 3 * len(contacts)``.
    """
    if not math.isfinite(gap_tol_m) or gap_tol_m < 0.0:
        raise ValueError(f"gap_tol_m must be finite and non-negative, got {gap_tol_m}")
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[0] != 3 * len(candidates):
        raise ValueError(
            f"jacobian has {jac.shape[0] if jac.ndim == 2 else '?'} rows, expected "
            f"3 x {len(candidates)} candidate contacts"
        )
    keep = [k for k, c in enumerate(candidates) if c.gap_m <= gap_tol_m]
    if not keep:
        return (), None
    rows = np.vstack([jac[3 * k : 3 * k + 3] for k in keep])
    return tuple(candidates[k].contact for k in keep), readonly_float(rows, "J_g")


def foot_friction_cones(snapshot: ClosedLoopSnapshot) -> tuple[ContactCone, ...]:
    """One Coulomb cone per ground contact of ``snapshot``, indexing ``x``.

    The cone of contact ``k`` acts on ``x[n_a + m + 3k : n_a + m + 3k + 3]``,
    in the contact frame ``(t1, t2, n)`` of its declared normal, with the
    declared friction coefficient ``FootContact.mu`` (hypothesis
    ``H-FRICTION-CONE``; pass the result to ``FeasibleSet.from_hypotheses``).

    Raises ``ValueError`` if the snapshot carries no foot contacts or their
    count does not match ``J_g``.
    """
    contacts = snapshot.foot_contacts
    n_rows = 0 if snapshot.J_g is None else snapshot.J_g.shape[0]
    if not contacts:
        raise ValueError("snapshot has no foot contacts to build friction cones")
    if n_rows != 3 * len(contacts):
        raise ValueError(
            f"J_g has {n_rows} rows for {len(contacts)} foot contacts (expected 3 each)"
        )
    lead = snapshot.actuated_rows().shape[0] + snapshot.closure_rows().shape[0]
    return tuple(
        ContactCone(
            c.name,
            FrictionCone(
                (lead + 3 * k, lead + 3 * k + 1, lead + 3 * k + 2),
                contact_frame(c.normal),
                c.mu,
            ),
        )
        for k, c in enumerate(contacts)
    )


# ------------------------------------------------------------ observation ---
@dataclass(frozen=True)
class GroundObservation:
    """Linear observation ``selector @ f_g = values`` with declared uncertainty.

    ``f_g`` is the stacked contact-load vector of the snapshot (three world-axis
    components per active contact).  ``sigma`` is the declared 1-sigma
    measurement uncertainty of every row in the row's units (N or N m); it is
    mandatory and strictly positive.  ``coverage`` is the band factor ``k``.
    ``source`` records where the numbers came from (a force plate, or a
    simulation standing in for one).  Arrays are read-only copies.
    """

    selector: Array
    values: Array
    sigma: Array
    labels: tuple[str, ...]
    source: str
    coverage: float = DEFAULT_COVERAGE

    def __post_init__(self) -> None:
        sel = readonly_float(self.selector, "GroundObservation.selector")
        val = readonly_float(self.values, "GroundObservation.values")
        sig = readonly_float(self.sigma, "GroundObservation.sigma")
        if sel.ndim != 2 or sel.shape[0] == 0:
            raise ValueError("selector must be a non-empty 2-D matrix")
        k = sel.shape[0]
        if val.shape != (k,):
            raise ValueError(f"values must have shape ({k},), got {val.shape}")
        if sig.shape != (k,) or np.any(sig <= 0.0):
            raise ValueError(
                f"sigma must be declared as {k} strictly positive values, got {sig}"
            )
        labels = tuple(self.labels)
        if len(labels) != k or len(set(labels)) != k or not all(labels):
            raise ValueError(f"labels must be {k} unique non-empty names")
        if not self.source:
            raise ValueError("source must name where the observation came from")
        if not math.isfinite(self.coverage) or self.coverage <= 0.0:
            raise ValueError(
                f"coverage must be positive and finite, got {self.coverage}"
            )
        for name, arr in (("selector", sel), ("values", val), ("sigma", sig)):
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "coverage", float(self.coverage))

    @property
    def n_rows(self) -> int:
        """Number of observation rows."""
        return int(self.selector.shape[0])

    @classmethod
    def from_point_loads(
        cls, snapshot: ClosedLoopSnapshot, *, sigma_n: float, coverage: float = 3.0
    ) -> GroundObservation:
        """Observe every contact-load entry: ``snapshot.grf_observed`` itself.

        Raises ``ValueError`` if the snapshot carries no ``grf_observed``.
        """
        if snapshot.grf_observed is None:
            raise ValueError(
                "snapshot has no grf_observed to build an observation from"
            )
        n = snapshot.grf_observed.shape[0]
        names = [c.name for c in snapshot.foot_contacts]
        labels = (
            tuple(f"{nm}:F{ax}" for nm in names for ax in _AXES)
            if len(names) * 3 == n
            else tuple(f"f_g[{i}]" for i in range(n))
        )
        return cls(
            np.eye(n),
            snapshot.grf_observed,
            np.full(n, _positive(sigma_n, "sigma_n")),
            labels,
            "snapshot.grf_observed",
            coverage,
        )

    @classmethod
    def from_contact_groups(
        cls,
        snapshot: ClosedLoopSnapshot,
        groups: Mapping[str, Sequence[str]],
        forces: Mapping[str, ArrayLike],
        *,
        sigma_n: float,
        coverage: float = DEFAULT_COVERAGE,
    ) -> GroundObservation:
        """Net force of each named group of contacts (a force plate per foot).

        ``groups`` maps a label to contact names; ``forces`` maps the same
        labels to the measured world-axis force (3,).  Three rows per group.
        """
        sigma = _positive(sigma_n, "sigma_n")
        names = [c.name for c in snapshot.foot_contacts]
        rows, values, labels = [], [], []
        for label, members in groups.items():
            unknown = sorted(set(members) - set(names))
            if unknown:
                raise ValueError(f"group {label!r} has unknown contacts {unknown}")
            force = np.asarray(forces[label], dtype=np.float64)
            check_shape(force, (3,), f"forces[{label!r}]")
            for axis in range(3):
                row = np.zeros(3 * len(names))
                for nm in members:
                    row[3 * names.index(nm) + axis] = 1.0
                rows.append(row)
                values.append(force[axis])
                labels.append(f"{label}:F{_AXES[axis]}")
        return cls(
            np.vstack(rows),
            np.array(values),
            np.full(len(rows), sigma),
            tuple(labels),
            "contact-group net force",
            coverage,
        )

    @classmethod
    def from_normal_load_and_cop(
        cls,
        snapshot: ClosedLoopSnapshot,
        *,
        normal_load_n: float,
        cop_m: ArrayLike,
        sigma_force_n: float,
        sigma_cop_m: float,
        coverage: float = DEFAULT_COVERAGE,
    ) -> GroundObservation:
        """Total normal load and centre of pressure as three linear rows.

        Row 0 is ``sum_k n . f_k = N``; rows 1 and 2 are the load-weighted
        offsets ``sum_k ((p_k - c) . t_i)(n . f_k) = 0`` (zero by definition of
        the centre of pressure ``c``), with uncertainty ``sigma_cop_m * N``.
        All contacts must share one normal.
        """
        load = _positive(normal_load_n, "normal_load_n")
        sig_f = _positive(sigma_force_n, "sigma_force_n")
        sig_c = _positive(sigma_cop_m, "sigma_cop_m")
        contacts = snapshot.foot_contacts
        if not contacts:
            raise ValueError("snapshot has no foot contacts")
        normal = contacts[0].normal
        if any(np.max(np.abs(c.normal - normal)) > ROTATION_ATOL for c in contacts):
            raise ValueError("contacts must share one ground normal")
        cop = np.asarray(cop_m, dtype=np.float64)
        check_shape(cop, (3,), "cop_m")
        frame = contact_frame(normal)
        sel = np.zeros((3, 3 * len(contacts)))
        for k, c in enumerate(contacts):
            sel[0, 3 * k : 3 * k + 3] = normal
            for i in (0, 1):
                sel[1 + i, 3 * k : 3 * k + 3] = (
                    float((c.position - cop) @ frame[:, i]) * normal
                )
        return cls(
            sel,
            np.array([load, 0.0, 0.0]),
            np.array([sig_f, sig_c * load, sig_c * load]),
            ("normal_load", "cop_moment_t1", "cop_moment_t2"),
            "total normal load and centre of pressure",
            coverage,
        )


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return float(value)


# -------------------------------------------------------------- accounting --
@dataclass(frozen=True)
class ObservationAccounting:
    """What an observation can see in a decomposition's null space.

    ``rank_visible = rank(E_x N)`` is the number of null directions removed in
    the zero-uncertainty limit; ``singular_values_over_sigma`` are the singular
    values of ``diag(1/sigma) E_x N`` (response of a unit scaled null step in
    units of the declared uncertainty, so a value below one is unresolved).
    ``chi2`` and ``max_normalised_residual`` measure the disagreement left
    between the data and the balance at the weighted least-squares member.
    """

    n_rows: int
    rank_selector: int
    nullity_free: int
    rank_visible: int
    nullity_observed: int
    singular_values_over_sigma: Array
    chi2: float
    max_normalised_residual: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {
            "n_rows": self.n_rows,
            "rank_selector": self.rank_selector,
            "nullity_free": self.nullity_free,
            "rank_visible": self.rank_visible,
            "nullity_observed": self.nullity_observed,
            "singular_values_over_sigma": self.singular_values_over_sigma.tolist(),
            "chi2": float(self.chi2),
            "max_normalised_residual": float(self.max_normalised_residual),
        }


@dataclass(frozen=True)
class _Fit:
    """Scaled observation response, rank and weighted least-squares step."""

    response: Array  # E_x N (rows in observation units)
    offset: Array  # E_x x_p - g
    rank: int
    right: Array  # right singular vectors of W E_x N (nullity, nullity)
    sv: Array
    z_star: Array
    residual: Array  # W (offset + response z_star)


def _check_against(dec: Decomposition, obs: GroundObservation) -> int:
    if not isinstance(dec, Decomposition):
        raise TypeError(f"dec must be a Decomposition, got {type(dec).__name__}")
    if not isinstance(obs, GroundObservation):
        raise TypeError(f"obs must be a GroundObservation, got {type(obs).__name__}")
    n_g = dec.counts.n_g
    if obs.selector.shape[1] != n_g:
        raise ValueError(
            f"observation selector has {obs.selector.shape[1]} columns, but the "
            f"snapshot has {n_g} contact-load entries"
        )
    return dec.n_actuated + dec.m


def _rank(sv: Array, rtol: float) -> int:
    if sv.size == 0:
        return 0
    return int(np.count_nonzero(sv > rtol * max(float(sv[0]), 1.0)))


def _fit(dec: Decomposition, obs: GroundObservation, rank_rtol: float) -> _Fit:
    lead = _check_against(dec, obs)
    n_basis = dec.N[lead:]
    response = obs.selector @ n_basis
    offset = obs.selector @ dec.x_p[lead:] - obs.values
    w = 1.0 / obs.sigma
    left, sv, right_t = np.linalg.svd(w[:, None] * response, full_matrices=True)
    rank = _rank(sv, rank_rtol)
    rhs = -(w * offset)
    z_star = (
        right_t[:rank].T @ ((left[:, :rank].T @ rhs) / sv[:rank])
        if rank
        else np.zeros(response.shape[1])
    )
    residual = w * offset + (w[:, None] * response) @ z_star
    return _Fit(response, offset, rank, right_t.T, sv, z_star, residual)


def observation_accounting(
    dec: Decomposition, obs: GroundObservation, *, rank_rtol: float = RANK_RTOL
) -> ObservationAccounting:
    """Count the null directions ``obs`` can see in ``dec``.

    Preconditions: types as named and the selector columns equal the number of
    contact-load entries (``ValueError``); ``0 < rank_rtol < 1``.
    Postconditions: ``nullity_observed = nullity_free - rank_visible`` and
    ``rank_visible <= min(n_rows, nullity_free)``.
    """
    if not 0.0 < rank_rtol < 1.0:
        raise ValueError(f"rank_rtol must lie in (0, 1), got {rank_rtol}")
    fit = _fit(dec, obs, rank_rtol)
    sel_sv = np.linalg.svd(obs.selector / obs.sigma[:, None], compute_uv=False)
    keep = min(obs.n_rows, dec.nullity)
    return ObservationAccounting(
        n_rows=obs.n_rows,
        rank_selector=_rank(sel_sv, rank_rtol),
        nullity_free=dec.nullity,
        rank_visible=fit.rank,
        nullity_observed=dec.nullity - fit.rank,
        singular_values_over_sigma=readonly_float(fit.sv[:keep], "sv"),
        chi2=float(fit.residual @ fit.residual),
        max_normalised_residual=float(np.max(np.abs(fit.residual), initial=0.0)),
    )


def restrict_to_observation(
    dec: Decomposition, obs: GroundObservation, *, rank_rtol: float = RANK_RTOL
) -> Decomposition:
    """The family of ``dec`` that also meets ``obs`` (zero-uncertainty limit).

    Moves the particular solution to the weighted least-squares member and
    drops the ``rank(E_x N)`` visible null directions, so the result has
    ``nullity_free - rank_visible`` null directions.  The result is a
    :class:`Decomposition` for ``FeasibleSet.from_hypotheses`` and the
    envelopes.  Its structural counts (``multiplier_gauge``, ...) still describe
    the unobserved snapshot; ``x_p`` is not re-minimised (the metric label says
    so) and ``records`` are cleared.

    Raises :class:`ObservationInconsistentError` when the data and the balance
    disagree by more than ``obs.coverage`` sigma; the disagreement is never
    absorbed silently.
    """
    fit = _fit(dec, obs, rank_rtol)
    worst = float(np.max(np.abs(fit.residual), initial=0.0))
    if worst > obs.coverage:
        raise ObservationInconsistentError(
            f"observation disagrees with the balance by {worst:.3g} sigma "
            f"(coverage {obs.coverage:g}); not restricting",
            worst,
        )
    keep_dirs = fit.right[:, fit.rank :]
    counts = dataclasses.replace(
        dec.counts, rank_A=dec.rank_A + fit.rank, nullity=dec.nullity - fit.rank
    )
    return dataclasses.replace(
        dec,
        x_p=readonly_float(dec.x_p + dec.N @ fit.z_star, "x_p"),
        N=readonly_float(dec.N @ keep_dirs, "N"),
        counts=counts,
        metric_label=f"{dec.metric_label}; restricted by {GRF_ID} (x_p not re-minimised)",
        records={},
    )


def observation_rows(
    dec: Decomposition, obs: GroundObservation
) -> tuple[FeasibleRow, ...]:
    """Two-sided band rows ``|E f_g(z) - g| <= k sigma`` in null coordinates.

    One upper and one lower :class:`FeasibleRow` per observation row, kind
    ``grf``, hypothesis ``H-GRF-OBSERVATION``.  The null dimension is kept.
    """
    lead = _check_against(dec, obs)
    response = obs.selector @ dec.N[lead:]
    offset = obs.selector @ dec.x_p[lead:] - obs.values
    out: list[FeasibleRow] = []
    for i, label in enumerate(obs.labels):
        slack = obs.coverage * float(obs.sigma[i])
        base = f"{GRF_ID}:{label}"
        out.append(
            FeasibleRow(
                response[i], slack - float(offset[i]), f"{base}:upper", GRF_ID, "grf"
            )
        )
        out.append(
            FeasibleRow(
                -response[i], slack + float(offset[i]), f"{base}:lower", GRF_ID, "grf"
            )
        )
    return tuple(out)
