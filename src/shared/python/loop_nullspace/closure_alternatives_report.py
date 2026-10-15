"""Per-grip-model dynamics, envelopes and comparison (NSX-72, #12590).

For each grip model (``closure_alternatives``) the trajectory is admitted under
the model's own residuals; only an admitted trajectory is decomposed and its
envelopes computed (reusing the NSX-26 machinery with the single closure kind of
the model, so there is no second solver).  The result records what differs from
a rows-only restriction of the welded kinematics: which trajectories the model
admits, how many motions it admits (kinematic nullity), the allocation
nullity, the feasibility counts and the envelope widths of the joint torques
and closure components.

Nothing here is evidence about the true grip mechanics, and a refusal is a
finding, never repaired.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.admission import AdmittedTrajectory
from src.shared.python.loop_nullspace.closure_alternatives import (
    GRIP_MODEL_SPECS,
    GripModel,
    ModelAdmission,
)
from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.envelopes import EnvelopeQuantity
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.restriction import (
    QuantityBuilder,
    RestrictionReport,
    restriction_sensitivity,
)

__all__ = [
    "GRIP_MODEL_LABEL",
    "GRIP_MODEL_NON_CLAIMS",
    "GripModelResult",
    "closure_component_quantities",
    "box_feasible_set_of",
    "evaluate_grip_model",
    "median_width_ratio",
]

GRIP_MODEL_LABEL = "CLOSURE_ALTERNATIVE_COMPARISON"
GRIP_MODEL_NON_CLAIMS = (
    "exploratory; not evidence about the true grip mechanics",
    "each grip model is a declared alternative, none is validated against a golfer",
    "capacities and priors are declared hypotheses and are identical for every model",
    "a refused or balance-inconsistent trajectory is a finding, never repaired",
)

FeasibleSetBuilder = Callable[[Decomposition, ClosedLoopSnapshot], FeasibleSet]
_RANK_RTOL = 1e-9


@dataclass(frozen=True, eq=False)
class GripModelResult:
    """Everything one grip model says about one trajectory.

    ``report`` is ``None`` when the trajectory was refused (``admission``
    says why).  ``kinematic_nullity`` is ``(min, max)`` of ``n - rank(J_c)``
    over frames, the dimension of the admitted velocities; ``allocation_nullity``
    is the ``(min, max)`` nullity of the decomposed frames (``None`` if none).
    """

    model: GripModel
    admission: ModelAdmission
    n_coordinates: int
    kinematic_nullity: tuple[int, int]
    report: RestrictionReport | None
    allocation_nullity: tuple[int, int] | None
    label: str = GRIP_MODEL_LABEL
    non_claims: tuple[str, ...] = GRIP_MODEL_NON_CLAIMS

    @property
    def kind(self) -> Any:
        """The model's closure kind."""
        return GRIP_MODEL_SPECS[self.model].closure_kind

    def counts(self) -> dict[str, int]:
        """Frames per outcome (all zero for a refused trajectory)."""
        if self.report is None:
            return {"feasible": 0, "infeasible": 0, "balance_inconsistent": 0}
        r, k = self.report, self.kind
        return {
            "feasible": r.feasible_counts()[k],
            "infeasible": r.infeasible_counts()[k],
            "balance_inconsistent": r.balance_inconsistent_counts()[k],
        }

    def widths(self, prefix: str) -> np.ndarray | None:
        """Envelope widths ``(frames, quantities)`` of quantities named ``prefix*``."""
        if self.report is None:
            return None
        series = self.report.series(self.kind)
        cols = [i for i, n in enumerate(series.names) if n.startswith(prefix)]
        return None if not cols else series.width()[:, cols]

    def median_width(self, prefix: str) -> float | None:
        """Median bounded width of the ``prefix*`` quantities (``None`` if none)."""
        width = self.widths(prefix)
        if width is None or not np.isfinite(width).any():
            return None
        return float(np.nanmedian(width))

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record (no arrays)."""
        adm = self.admission
        return {
            "model": self.model.value,
            "closure_kind": self.kind.value,
            "rows": list(GRIP_MODEL_SPECS[self.model].components),
            "n_coordinates": self.n_coordinates,
            "admitted": adm.admitted,
            "n_frames": adm.n_frames,
            "max_phi": _num(adm.max_phi),
            "max_phi_dot": _num(adm.max_phi_dot),
            "max_phi_ddot": _num(adm.max_phi_ddot),
            "reasons": list(adm.reasons[:3]),
            "failing_frames": list(adm.failing_frames),
            "kinematic_nullity": list(self.kinematic_nullity),
            "allocation_nullity": None
            if self.allocation_nullity is None
            else list(self.allocation_nullity),
            "counts": self.counts(),
            "median_torque_width_nm": self.median_width("tau_"),
            "median_closure_width": self.median_width("lambda_"),
            "label": self.label,
        }


def box_feasible_set_of(cap_nm: float) -> FeasibleSetBuilder:
    """Feasible-set builder with a symmetric torque box ``|tau_i| <= cap_nm``.

    The box is a declared hypothesis (``H-TORQUE-CAPACITY``, every slot ASSUMED)
    on a synthetic fixture, not a golfer capacity; it is identical for every
    grip model.  Raises ``ValueError`` for a non-positive ``cap_nm``.
    """
    if not math.isfinite(cap_nm) or cap_nm <= 0.0:
        raise ValueError(f"cap_nm must be positive and finite, got {cap_nm}")
    hypotheses = registry.preset("envelope_only")

    def build(dec: Decomposition, snap: ClosedLoopSnapshot) -> FeasibleSet:
        n_a = snap.topology.n_actuated
        names = tuple(f"tau_{i + 1}" for i in range(n_a))
        capacity = CapacityVector(
            coordinate_order=names,
            actuated_indices=tuple(range(n_a)),
            actuated_names=names,
            tau_max_negative_nm=np.full(n_a, cap_nm),
            tau_max_positive_nm=np.full(n_a, cap_nm),
            assumed_slots=names,
            verified_slots=(),
            warnings=(),
            force_velocity={},
        )
        return FeasibleSet.from_hypotheses(dec, hypotheses, capacity=capacity)

    return build


def _num(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def closure_component_quantities(components: Sequence[str]) -> QuantityBuilder:
    """Builder of the closure multiplier components ``lambda_<name>`` of a model.

    ``components`` are the model's kept rows in order; the multiplier block
    follows the actuated torques in the decomposition.
    """
    names = tuple(components)

    def build(fs: FeasibleSet, _snapshot: ClosedLoopSnapshot) -> list[EnvelopeQuantity]:
        dec = fs.decomposition
        if dec is None:
            raise ValueError("this FeasibleSet carries no decomposition")
        if dec.m != len(names):
            raise ValueError(f"{len(names)} component names for {dec.m} multipliers")
        out = []
        for k, name in enumerate(names):
            force = name.startswith("f")
            out.append(
                EnvelopeQuantity(
                    f"lambda_{name}",
                    dec.N[dec.n_actuated + k],
                    float(dec.x_p[dec.n_actuated + k]),
                    "N" if force else "N m",
                    dec.scales.force if force else dec.scales.moment,
                )
            )
        return out

    return build


def _rank_nullity(snaps: Sequence[ClosedLoopSnapshot]) -> tuple[int, int]:
    values = []
    for snap in snaps:
        sv = np.linalg.svd(snap.J_c, compute_uv=False)
        rank = int(np.count_nonzero(sv > _RANK_RTOL * max(float(sv[0]), 1.0)))
        values.append(snap.topology.n - rank)
    return min(values), max(values)


def evaluate_grip_model(
    admission: ModelAdmission,
    *,
    quantities: Sequence[QuantityBuilder],
    feasible_set_of: FeasibleSetBuilder,
) -> GripModelResult:
    """Dynamics and envelopes of an admitted trajectory under its grip model.

    ``admission`` is the :class:`ModelAdmission` of the trajectory under the
    model's own residuals (``admission_outcome`` or ``admit_model``).  A refused
    trajectory returns a result with ``report=None``.  The priors in
    ``feasible_set_of`` are the caller's and are identical for every model; the
    balance tolerance is never relaxed.  Raises ``ValueError`` for empty
    ``quantities``.
    """
    if not list(quantities):
        raise ValueError("quantities must contain at least one builder")
    snaps = admission.report.snapshots
    kinds = _rank_nullity(snaps)
    n = snaps[0].topology.n
    if not admission.admitted:
        return GripModelResult(admission.model, admission, n, kinds, None, None)
    traj = AdmittedTrajectory.from_report(admission.report)
    kind = GRIP_MODEL_SPECS[admission.model].closure_kind
    report = restriction_sensitivity(
        traj, (kind,), list(quantities), feasible_set_of=feasible_set_of
    )
    nullities = [
        f.nullity for f in report.outcomes[kind].frames if f.nullity is not None
    ]
    alloc = (min(nullities), max(nullities)) if nullities else None
    return GripModelResult(admission.model, admission, n, kinds, report, alloc)


def median_width_ratio(
    reference: GripModelResult, other: GripModelResult, prefix: str
) -> float | None:
    """Median over frames and quantities of ``width(other) / width(reference)``.

    Only entries bounded in both with a positive reference width count.  Returns
    ``None`` when either result was refused, the frame counts differ or no entry
    qualifies.  A value below one means the other model's envelope is narrower.
    """
    ref, oth = reference.widths(prefix), other.widths(prefix)
    if ref is None or oth is None or ref.shape != oth.shape:
        return None
    ok = np.isfinite(ref) & np.isfinite(oth) & (ref > 0.0)
    return float(np.median(oth[ok] / ref[ok])) if ok.any() else None
