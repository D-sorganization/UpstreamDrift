"""Evidence typing and assessment of variance results (NSX-90, #12719).

Turns a :class:`variance_propagation.VarianceProblem` into
:class:`~evidence.QuantityRecord` objects typed per NSX-81
(``contracts.md``, "conditioning"):

* **Kernels** ``S_{y,tau}``, ``S_{y,x0}`` are ``INVARIANT`` given the model while
  the linearization ``A_t`` does not depend on the allocation, i.e. neither
  mechanism M2 (geometric preload, ``H-GEOMETRIC-PRELOAD``) nor M3 (muscle
  impedance, ``H-MUSCLE-SHORT-RANGE-STIFFNESS``) is declared.  ``B_t`` is
  allocation-independent (``G_t N = 0``).  If M2 or M3 is declared the kernels
  move with ``z`` and become ``FAMILY_DEPENDENT`` under that hypothesis.
* **Variance** under purely additive noise with no M2/M3 is ``INVARIANT`` and
  carries the noise hypothesis on the orthogonal ``conditioning`` axis; any
  declared mechanism (M1 signal-dependent noise, M2, M3) makes it
  ``FAMILY_DEPENDENT`` with that mechanism's hypothesis (V1, V7).

The nonlinearity gate (:mod:`nonlinearity_gate`) sets ``linear_validity``;
without a gate the result is ``NOT_ASSESSED``, never silently valid.  Values are
always ``UNVALIDATED`` here: validation needs an external measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.shared.python.core.contracts import ensure
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import (
    DEFAULT_RESPONSE_FLOOR,
    LinearValidity,
    NonlinearityReport,
    Rollout,
    assess_nonlinearity,
    probe_directions,
)
from src.shared.python.loop_nullspace.variance_propagation import (
    VarianceBudget,
    VarianceProblem,
    adjoint_covariance,
)
from src.shared.python.perturbation.disturbance import DisturbanceSpec

__all__ = [
    "GateSpec",
    "Mechanism",
    "NON_CLAIMS",
    "VarianceAssessment",
    "VarianceDeclaration",
    "assess_variance",
    "declared_mechanisms",
    "noise_hypotheses",
]

NON_CLAIMS = (
    "Linear prediction in the tangent basis at the nominal motion; it does not "
    "cover saturation, slip or missed impacts. Noise amplitudes are declared "
    "hypotheses, not measurements."
)
H_NOISE_ADDITIVE = "H-NOISE-ADDITIVE"


class Mechanism(str, Enum):
    """Declared mechanisms by which the allocation can change variance (V1)."""

    SIGNAL_DEPENDENT_NOISE = "H-NOISE-SIGNAL-DEPENDENT"
    GEOMETRIC_PRELOAD = "H-GEOMETRIC-PRELOAD"
    MUSCLE_IMPEDANCE = "H-MUSCLE-SHORT-RANGE-STIFFNESS"


#: Mechanisms that act on ``A_t`` (and so on the kernels).
_DYNAMICS_MECHANISMS = (Mechanism.GEOMETRIC_PRELOAD, Mechanism.MUSCLE_IMPEDANCE)


def _specs(specs: Sequence[DisturbanceSpec]) -> list[DisturbanceSpec]:
    if isinstance(specs, str) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of DisturbanceSpec")
    if len(specs) == 0:
        raise ValueError("at least one DisturbanceSpec is required")
    if not all(isinstance(s, DisturbanceSpec) for s in specs):
        raise TypeError("every spec must be a DisturbanceSpec")
    return list(specs)


def declared_mechanisms(
    specs: Sequence[DisturbanceSpec],
    *,
    geometric_preload: bool,
    muscle_impedance: bool = False,
) -> frozenset[Mechanism]:
    """Mechanisms in force: M1 from the specs' signal law, M2/M3 as declared."""
    out: set[Mechanism] = set()
    if any(s.has_signal_law for s in _specs(specs)):
        out.add(Mechanism.SIGNAL_DEPENDENT_NOISE)
    if geometric_preload:
        out.add(Mechanism.GEOMETRIC_PRELOAD)
    if muscle_impedance:
        out.add(Mechanism.MUSCLE_IMPEDANCE)
    return frozenset(out)


def noise_hypotheses(specs: Sequence[DisturbanceSpec]) -> tuple[str, ...]:
    """Noise-model hypothesis ids the variance is conditioned on (sorted)."""
    ids = {
        Mechanism.SIGNAL_DEPENDENT_NOISE.value if s.has_signal_law else H_NOISE_ADDITIVE
        for s in _specs(specs)
    }
    return tuple(sorted(ids))


@dataclass(frozen=True)
class VarianceDeclaration:
    """What was declared about the disturbance model and where it came from."""

    mechanisms: frozenset[Mechanism]
    noise_hypotheses: tuple[str, ...]
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    conventions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.noise_hypotheses) == 0:
            raise ValueError("noise_hypotheses must name the noise model")
        if not all(isinstance(m, Mechanism) for m in self.mechanisms):
            raise TypeError("mechanisms must be Mechanism members")

    def ids(self, only: tuple[Mechanism, ...] | None = None) -> tuple[str, ...]:
        """Hypothesis ids of the declared mechanisms (optionally a subset)."""
        keep = [m for m in Mechanism if m in self.mechanisms]
        if only is not None:
            keep = [m for m in keep if m in only]
        return tuple(m.value for m in keep)


@dataclass(frozen=True)
class GateSpec:
    """Nonlinear rollout and gate parameters; ``gate_threshold`` is the caller's."""

    rollout: Rollout
    amplitude: float
    gate_threshold: float
    response_floor: float = DEFAULT_RESPONSE_FLOOR
    include_axes: bool = True


@dataclass(frozen=True)
class VarianceAssessment:
    """Typed variance, kernels, budget and gate outcome of one problem."""

    variance: QuantityRecord
    kernel_x0: QuantityRecord
    kernel_input: QuantityRecord
    budget: VarianceBudget
    nonlinearity: NonlinearityReport | None
    linear_validity: LinearValidity
    non_claims: str = NON_CLAIMS


def _record(
    values: np.ndarray,
    units: str,
    decl: VarianceDeclaration,
    ids: tuple[str, ...],
    conditioning: tuple[str, ...],
) -> QuantityRecord:
    family = len(ids) > 0
    return QuantityRecord(
        values=values,
        units=units,
        allocation_dependence=(
            AllocationDependence.FAMILY_DEPENDENT
            if family
            else AllocationDependence.INVARIANT
        ),
        model_provenance=decl.model_provenance,
        input_provenance=decl.input_provenance,
        validation_status=ValidationStatus.UNVALIDATED,
        conventions=decl.conventions,
        hypotheses=ids,
        conditioning=conditioning,
    )


def _run_gate(
    problem: VarianceProblem, gate: GateSpec | None
) -> tuple[NonlinearityReport | None, LinearValidity]:
    if gate is None:
        return None, LinearValidity.NOT_ASSESSED
    dirs = probe_directions(
        problem.kernels(),
        problem.initial_covariance,
        problem.step_covariance,
        include_axes=gate.include_axes,
    )
    report = assess_nonlinearity(
        gate.rollout,
        dirs,
        amplitude=gate.amplitude,
        gate_threshold=gate.gate_threshold,
        response_floor=gate.response_floor,
    )
    return report, report.validity


def assess_variance(
    problem: VarianceProblem,
    declaration: VarianceDeclaration,
    *,
    outcome_units: str,
    input_units: str = "input unit",
    state_units: str = "tangent-state unit",
    gate: GateSpec | None = None,
) -> VarianceAssessment:
    """Forward and adjoint variance with evidence typing and the gate flag.

    Preconditions (``ValueError``/``TypeError``): non-empty unit strings and
    valid problem/declaration.  Postconditions: the adjoint variance equals the
    forward recursion to ``1e-9`` relative; ``variance`` is ``INVARIANT`` only
    when no mechanism is declared; ``linear_validity`` is ``NOT_ASSESSED``
    without a gate and never ``LINEAR_VALID`` unless the gate ran.
    """
    if not (outcome_units and input_units and state_units):
        raise ValueError("unit strings must be non-empty")
    kernels = problem.kernels()
    cov_adj = adjoint_covariance(
        kernels, problem.initial_covariance, problem.step_covariance
    )
    cov_fwd = problem.forward_outcome_covariance()
    ensure(
        bool(np.allclose(cov_adj, cov_fwd, rtol=1e-9, atol=1e-15)),
        "adjoint and forward outcome covariances must agree",
    )
    decl_ids = declaration.ids(_DYNAMICS_MECHANISMS)
    variance = _record(
        np.diag(cov_fwd),
        f"({outcome_units})^2",
        declaration,
        declaration.ids(),
        declaration.noise_hypotheses,
    )
    kernel_x0 = _record(
        kernels.s_y_x0, f"({outcome_units})/({state_units})", declaration, decl_ids, ()
    )
    kernel_in = _record(
        kernels.inputs, f"({outcome_units})/({input_units})", declaration, decl_ids, ()
    )
    report, validity = _run_gate(problem, gate)
    return VarianceAssessment(
        variance=variance,
        kernel_x0=kernel_x0,
        kernel_input=kernel_in,
        budget=problem.budget(),
        nonlinearity=report,
        linear_validity=validity,
    )
