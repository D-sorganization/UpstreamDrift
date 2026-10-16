"""Evidence typing and oracle adapters for global sensitivity (NSX-96, #12725).

Indices become :class:`~evidence.QuantityRecord` objects labelled with the
``EVIDENCE:SCREENING`` convention and ``UNVALIDATED``.  They are conditional on
the declared input ranges and correlation (the ``conditioning`` axis); they do
not depend on an allocation choice, so they are ``INVARIANT`` given the model.

Adapters run only on the NSX-80 planar oracle and the NSX-70 low-rank temporal
basis.  No full-body model is evaluated.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.fixtures import load_golden
from src.shared.python.loop_nullspace.global_sensitivity import (
    InputGroup,
    InputSpace,
    PrccResult,
    SensitivityInput,
    SobolResult,
)
from src.shared.python.loop_nullspace.hit_and_run import bspline_temporal_basis
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    planar_dynamics,
)

__all__ = [
    "NON_CLAIMS",
    "SCREENING_CONVENTION",
    "basis_response_model",
    "planar_oracle_model",
    "prcc_records",
    "sobol_records",
]

SCREENING_CONVENTION = "EVIDENCE:SCREENING"
H_INPUT_DISTRIBUTION = "H-SCREENING-INPUT-DISTRIBUTION"
NON_CLAIMS = (
    "Screening evidence on a model under declared input ranges and correlation; "
    "never a validated ranking of parameters and not a statement about golfers."
)


def _record(
    values: Array,
    note: str,
    model_name: str,
    model_version: str,
    conventions: tuple[str, ...],
) -> QuantityRecord:
    return QuantityRecord(
        values=values,
        units="1",
        allocation_dependence=AllocationDependence.INVARIANT,
        model_provenance=ModelProvenance(model_name, model_version),
        input_provenance=InputProvenance("sensitivity-screening", model_name, note),
        validation_status=ValidationStatus.UNVALIDATED,
        conventions=(SCREENING_CONVENTION, *conventions),
        conditioning=(H_INPUT_DISTRIBUTION,),
    )


def sobol_records(
    result: SobolResult, *, model_name: str, model_version: str
) -> dict[str, QuantityRecord]:
    """``first_order``, ``total`` and their ``*_ci`` records, all SCREENING."""
    note = (
        f"Sobol {result.level} indices, N={result.n}, seed={result.seed}, "
        f"units: {', '.join(result.names)}. {NON_CLAIMS}"
    )
    conv = ("SALTELLI-2010-FIRST-ORDER", "JANSEN-TOTAL")
    make = lambda v: _record(v, note, model_name, model_version, conv)  # noqa: E731
    return {
        "first_order": make(result.first_order),
        "total": make(result.total),
        "first_order_ci": make(result.first_order_ci),
        "total_ci": make(result.total_ci),
    }


def prcc_records(
    result: PrccResult, *, model_name: str, model_version: str
) -> dict[str, QuantityRecord]:
    """``prcc`` and ``prcc_ci`` records, SCREENING and UNVALIDATED."""
    note = (
        f"PRCC on a Latin hypercube, N={result.n}, seed={result.seed}, "
        f"inputs: {', '.join(result.names)}. {NON_CLAIMS}"
    )
    conv = ("PRCC-RANK-PARTIAL-CORRELATION",)
    return {
        "prcc": _record(result.prcc, note, model_name, model_version, conv),
        "prcc_ci": _record(result.ci, note, model_name, model_version, conv),
    }


def planar_oracle_model(
    *, correlated_masses: float | None = None
) -> tuple[object, InputSpace]:
    """Planar-oracle response ``h[0]`` (lead-shoulder bias torque) and its inputs.

    Inputs are +/-20 % on the arm and club masses, +/-10 % on the club COM
    distance and a lead-shoulder rate in ``[0, 3]`` rad/s, about the NSX-00
    golden state.  With ``correlated_masses = rho`` the two arm masses form a
    correlated group (grouped indices are then the only route).
    """
    gold = load_golden("planar_two_arm_full")
    base = PlanarGolferParams.from_golden(gold)
    q = gold.array("state", "q")
    v0 = gold.array("state", "v")
    bounds = {
        "upper_arm_mass": 0.8 * base.upper_arm_mass,
        "forearm_mass": 0.8 * base.forearm_mass,
        "club_mass": 0.8 * base.club_mass,
        "club_com_from_lead_hand": 0.9 * base.club_com_from_lead_hand,
    }
    inputs = [
        SensitivityInput(k, lo, lo * (1.2 / 0.8 if "mass" in k else 1.1 / 0.9))
        for k, lo in bounds.items()
    ]
    inputs.append(SensitivityInput("lead_shoulder_rate", 0.0, 3.0))
    groups: tuple[InputGroup, ...] = ()
    if correlated_masses is not None:
        rho = float(correlated_masses)
        groups = (
            InputGroup(
                ("upper_arm_mass", "forearm_mass"),
                np.array([[1.0, rho], [rho, 1.0]]),
            ),
        )
    names = [i.name for i in inputs]

    def model(x: Array) -> Array:
        out = np.empty(x.shape[0])
        for r, row in enumerate(x):
            vals = dict(zip(names, row, strict=True))
            params = dataclasses.replace(base, **{k: float(vals[k]) for k in bounds})
            vel = v0.copy()
            vel[0] = vals["lead_shoulder_rate"]
            out[r] = planar_dynamics(q, vel, params)["h"][0]
        return out

    return model, InputSpace(tuple(inputs), groups)


def basis_response_model(
    *, n_coefficients: int, n_frames: int, seed: int = 0
) -> tuple[object, InputSpace]:
    """Response through the NSX-70 low-rank temporal basis, ``z(t) = B c``.

    Output is the time mean of a fixed weighting of ``z(t)`` (linear in the
    coefficients ``c_k``), so the exact first-order shares are
    ``(w_k s_k)^2 / sum`` with ``s_k`` the coefficient range; used to exercise
    the screening on the basis only.
    """
    if n_coefficients < 2 or n_frames < n_coefficients:
        raise ValueError("need n_coefficients >= 2 and n_frames >= n_coefficients")
    basis = bspline_temporal_basis(np.linspace(0.0, 1.0, n_frames), n_coefficients)
    weight = 1.0 + np.sin(np.linspace(0.0, np.pi, n_frames))
    loading = basis.T @ weight / n_frames

    def model(x: Array) -> Array:
        return np.asarray(x @ loading)

    inputs = tuple(
        SensitivityInput(f"c{k}", -1.0 - 0.5 * k, 1.0 + 0.5 * k)
        for k in range(n_coefficients)
    )
    return model, InputSpace(inputs)
