"""Observability of inertial parameters from dynamics-matching data.

Two complementary analyses are provided.

* :func:`structural_observability` uses only the *unactuated* regressor rows
  ``S_perp Y(q, v, a) pi = S_perp J_c^T lambda`` (zero here, no external
  wrench).  These rows contain no unknown input, so their row space is the part
  of ``pi`` that kinematics alone can see (Ayusawa, Venture and Nakamura,
  IJRR 2014).  Because the rows are homogeneous in ``pi``, a global mass scale
  is always a null direction unless anchored.
* :func:`data_information` eliminates the inputs and template from the inner
  normal matrix (Schur complement) and subtracts the prior, leaving the Fisher
  information on ``pi`` contributed by the data *including* the effort prior
  that ties the actuated rows.  :func:`posterior_observability` reports its
  eigen-structure.

Both report per-parameter observable fractions so the estimator can restrict
updates to the observable subspace and leave the rest to the anthropometric
prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.inertial import PLANAR_PARAMETERS_PER_BODY
from src.shared.python.estimation.mosaic.inner_solve import InnerSolution

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
_PLANAR_NAMES = ("m", "hx", "hy", "Io")


@dataclass(frozen=True)
class ObservabilityReport:
    """Eigen/singular structure of an information matrix on ``pi``."""

    parameter_names: tuple[str, ...]
    eigenvalues: FloatArray
    directions: FloatArray
    rank: int
    tolerance: float

    @property
    def observable_directions(self) -> FloatArray:
        return self.directions[:, : self.rank]

    @property
    def null_directions(self) -> FloatArray:
        return self.directions[:, self.rank :]

    @property
    def observable_fraction(self) -> FloatArray:
        """Per parameter, squared loading onto the observable subspace (in [0, 1])."""
        return np.sum(self.observable_directions**2, axis=1)


def default_parameter_names(n_parameters: int) -> tuple[str, ...]:
    """Planar naming ``m_i, hx_i, hy_i, Io_i`` when ``n_parameters`` is 4 per body."""
    if n_parameters % PLANAR_PARAMETERS_PER_BODY == 0:
        bodies = n_parameters // PLANAR_PARAMETERS_PER_BODY
        return tuple(
            f"{name}_{body}" for body in range(bodies) for name in _PLANAR_NAMES
        )
    return tuple(f"pi_{index}" for index in range(n_parameters))


def _report_from_information(
    information: FloatArray, relative_tolerance: float, names: tuple[str, ...] | None
) -> ObservabilityReport:
    require(
        information.ndim == 2 and information.shape[0] == information.shape[1], "square"
    )
    eigenvalues, vectors = np.linalg.eigh(0.5 * (information + information.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    tolerance = relative_tolerance * max(float(eigenvalues[0]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    labels = names or default_parameter_names(information.shape[0])
    ensure(len(labels) == information.shape[0], "one name per parameter")
    return ObservabilityReport(labels, eigenvalues, vectors, rank, tolerance)


def structural_observability(
    regressors: FloatArray,
    unactuated_rows: IntArray,
    relative_tolerance: float = 1e-10,
    parameter_names: tuple[str, ...] | None = None,
) -> ObservabilityReport:
    """Observability from unactuated rows only (no inputs, no priors).

    A fully actuated model has no such rows: the report then has rank zero
    (kinematics alone carry no torque-free information about ``pi``).
    """
    require(regressors.ndim == 3, "regressors must be (N, n_v, n_pi)", regressors.shape)
    rows = np.asarray(unactuated_rows, dtype=np.int64)
    stacked = regressors[:, rows, :].reshape(-1, regressors.shape[2])
    return _report_from_information(
        stacked.T @ stacked, relative_tolerance, parameter_names
    )


def data_information(solution: InnerSolution) -> FloatArray:
    """Fisher information on ``pi`` from data after eliminating inputs/template.

    ``H_pp - H_pw H_ww^{-1} H_wp`` of the inner normal matrix, minus the prior
    block, so that a pure-prior solve yields (numerically) zero information.
    """
    require(solution.parameters_free, "data_information needs free inner parameters")
    n_pi = solution.parameters.size
    design = solution.design
    normal = (design.T @ design).toarray()
    prior_block = design[solution.parameter_prior_rows][:, :n_pi].toarray()
    prior_information = prior_block.T @ prior_block
    h_pp, h_pw, h_ww = normal[:n_pi, :n_pi], normal[:n_pi, n_pi:], normal[n_pi:, n_pi:]
    schur = h_pp - h_pw @ np.linalg.solve(h_ww, h_pw.T) if h_ww.size else h_pp
    information = schur - prior_information
    return 0.5 * (information + information.T)


def posterior_observability(
    solution: InnerSolution,
    relative_tolerance: float = 1e-8,
    parameter_names: tuple[str, ...] | None = None,
) -> ObservabilityReport:
    """Eigen-structure of :func:`data_information` for the solved inner problem."""
    return _report_from_information(
        data_information(solution), relative_tolerance, parameter_names
    )
