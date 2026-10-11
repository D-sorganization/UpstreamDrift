"""Deterministic evidence receipt for the planar MOSAIC reference results.

The methods reference quotes measured numbers.  This module regenerates them
from a fixed-seed fixture and writes a JSON receipt so the quoted values are
verifiable (``python3 -m src.shared.python.estimation.mosaic.reference_evidence``)
and so a test can fail when the implementation drifts away from the document.
Timings are recorded but never compared.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.estimation.mosaic.inertial import (
    PlanarParameterization,
    project_planar_consistent,
)
from src.shared.python.estimation.mosaic.inner_solve import LinearAnchor
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.estimation.mosaic.outer_solve import (
    OuterOptions,
    OuterPriors,
    TrialData,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)
from src.shared.python.estimation.mosaic.subject_fit import (
    ReplayConfig,
    SubjectFitProblem,
    SubjectFitReport,
    fit_subject,
)

RECEIPT_PATH = Path("docs/research/model_aware_matching/reference_results.json")
ACTUATED = np.array([False, True, True])
TRUE_LENGTHS = np.array([0.9, 0.6, 0.4])
TRUE_PI = np.array(
    [1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064, 0.3, 0.09, 0.0, 0.03]
)
MASS_ROW = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.0])
CLUB_PARAMETERS = (8, 9, 11)
MARKERS = PlanarMarkerSet(
    link_index=np.array([0, 0, 1, 1, 2, 2]),
    offsets=np.array(
        [[0.3, 0.02], [0.8, -0.03], [0.2, 0.0], [0.55, 0.04], [0.1, 0.0], [0.35, -0.02]]
    ),
)


@dataclass(frozen=True)
class FixtureSpec:
    """Fixed-seed synthetic experiment definition."""

    n_trials: int = 2
    n_nodes: int = 150
    sample_rate_hz: float = 200.0
    marker_noise_m: float = 5e-4
    data_seed: int = 8
    prior_seed: int = 2
    geometry_scale: float = 1.05
    prior_relative_spread: float = 0.10


def make_chain(lengths: np.ndarray) -> PlanarChain:
    return PlanarChain(link_lengths=lengths, actuated=ACTUATED)


def synthetic_trials(
    spec: FixtureSpec,
) -> tuple[list[TrialData], list[tuple[Any, Any, Any]]]:
    """Simulate the fixture; returns observations and ``(q, v, u)`` truth per trial."""
    rng = np.random.default_rng(spec.data_seed)
    chain = make_chain(TRUE_LENGTHS)
    dt = 1.0 / spec.sample_rate_hz
    times = (np.arange(spec.n_nodes) * dt).astype(np.float64)
    amps = rng.uniform(0.6, 1.2, size=spec.n_trials)
    freq = rng.uniform(0.9, 1.3, size=spec.n_trials)
    phase = 2 * np.pi * freq[:, None] * times[None, :]
    u = np.stack([amps[:, None] * np.sin(phase), 0.3 * np.cos(phase)], axis=-1)
    hanging = np.array([-np.pi / 2 + 0.4, 0.3, -0.2, 0.0, 0.0, 0.0])
    x0 = np.tile(hanging, (spec.n_trials, 1)) + rng.normal(
        scale=0.05, size=(spec.n_trials, 6)
    )
    q, v = chain.rollout(x0, u[:, :-1], TRUE_PI, dt)
    trials, truth = [], []
    for k in range(spec.n_trials):
        clean = chain.marker_positions(q[k], MARKERS)
        observed = clean + rng.normal(scale=spec.marker_noise_m, size=clean.shape)
        basis = BSplineBasis.uniform(times, n_coefficients=spec.n_nodes // 4, degree=5)
        sigma = max(spec.marker_noise_m, 1e-4)
        trials.append(
            TrialData(times, observed, MARKERS, sigma, basis, np.arange(spec.n_nodes))
        )
        truth.append((q[k], v[k], u[k]))
    return trials, truth


def reference_priors(pi_prior: np.ndarray) -> OuterPriors:
    """15 % anthropometric sigma; total mass and club inertia anchored."""
    anchors = [LinearAnchor(MASS_ROW, float(MASS_ROW @ TRUE_PI), 100.0)]
    anchors.extend(
        LinearAnchor(np.eye(12)[i], float(TRUE_PI[i]), 1e3) for i in CLUB_PARAMETERS
    )
    return OuterPriors(
        geometry_mean=TRUE_LENGTHS * 1.05,
        geometry_weight=np.full(3, 1.0 / 0.05),
        parameter_prior_mean=pi_prior,
        parameter_prior_weight=1.0 / (0.15 * np.abs(TRUE_PI) + 0.02),
        dynamics_sigma=np.full(3, 0.2),
        input_effort_weight=0.3,
        input_smoothness_weight=10.0,
        template_weight=0.0,
        anchors=tuple(anchors),
    )


def perturbed_prior(spec: FixtureSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.prior_seed)
    prior = TRUE_PI * (
        1.0
        + rng.uniform(
            -spec.prior_relative_spread, spec.prior_relative_spread, size=TRUE_PI.size
        )
    )
    prior[2::4] = 0.0
    return project_planar_consistent(prior.reshape(-1, 4) * [1, 1, 1, 1.1]).ravel()


def _step_factory(geometry: np.ndarray, parameters: np.ndarray, dt: float):
    chain = make_chain(geometry)

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return chain.step(x, u, parameters, dt)

    return step


def run_reference(
    spec: FixtureSpec = FixtureSpec(),
) -> tuple[SubjectFitReport, dict[str, Any]]:
    """Run the fixture and return the report plus a JSON-serialisable receipt."""
    trials, truth = synthetic_trials(spec)
    pi_prior = perturbed_prior(spec)
    problem = SubjectFitProblem(
        factory=make_chain,
        trials=tuple(trials),
        first_frame_q=tuple(t[0][0] + 0.1 for t in truth),
        initial_geometry=TRUE_LENGTHS * spec.geometry_scale,
        initial_parameters=pi_prior,
        parameterization=PlanarParameterization(),
        priors=reference_priors(pi_prior),
        options=OuterOptions(),
    )
    config = ReplayConfig(
        step_factory=_step_factory,
        state_weight=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
        input_weight=np.eye(2) * 0.01,
        position_tolerance=0.15,
        initial_perturbation=np.zeros(6),
    )
    report = fit_subject(problem, config, n_phase_bins=15)
    return report, _receipt(report, trials, truth, pi_prior, spec)


def _receipt(report, trials, truth, pi_prior, spec) -> dict[str, Any]:
    fit = report.fit
    u_fit = fit.inner.inputs.reshape(spec.n_trials, spec.n_nodes, 2)
    u_true = np.stack([t[2] for t in truth])
    q_err = [
        float(np.sqrt(np.mean((trial.basis.evaluate(c)[0] - q_true) ** 2)))
        for trial, c, (q_true, _, _) in zip(
            trials, fit.coefficients, truth, strict=True
        )
    ]
    comparable = (report.observability.observable_fraction > 0.5) & (
        np.abs(TRUE_PI) > 1e-6
    )

    def relative_error(parameters: np.ndarray) -> float:
        return float(
            np.max(np.abs((parameters - TRUE_PI)[comparable] / TRUE_PI[comparable]))
        )

    return {
        "fixture": spec.__dict__,
        "geometry_error_identifiable_m": float(
            np.max(np.abs(fit.geometry[:2] - TRUE_LENGTHS[:2]))
        ),
        "kinematic_rms_rad": q_err,
        "torque_rms_error": float(np.sqrt(np.mean((u_fit - u_true) ** 2))),
        "torque_rms_scale": float(np.sqrt(np.mean(u_true**2))),
        "inertia_max_relative_error_observable": relative_error(fit.parameters),
        "inertia_max_relative_error_prior": relative_error(pi_prior),
        "structural_rank": int(report.observability.rank),
        "n_parameters": int(TRUE_PI.size),
        "unobservable_parameters": [
            n
            for n, f in zip(
                report.observability.parameter_names,
                report.observability.observable_fraction,
                strict=True,
            )
            if f < 1e-6
        ],
        "replay": [
            {
                "open_loop_rms_rad": r.open_loop_rms,
                "closed_loop_rms_rad": r.closed_loop_rms,
                "feedback_effort_rms": r.feedback_effort_rms,
                "divergence_time_s": r.divergence_time_s,
                "open_loop_accepted": r.open_loop_accepted,
            }
            for r in report.replays
        ],
        "outer_iterations_per_stage": [r.iterations for r in fit.receipts],
        "all_stages_converged": bool(all(r.converged for r in fit.receipts)),
        "timings_s_not_compared": report.timings_s,
    }


def write_receipt(path: Path = RECEIPT_PATH) -> dict[str, Any]:
    """Regenerate the receipt file; returns the receipt."""
    require(path.suffix == ".json", "receipt path must be a .json file", path)
    _, receipt = run_reference()
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else RECEIPT_PATH
    sys.stdout.write(json.dumps(write_receipt(target), indent=2, sort_keys=True) + "\n")
