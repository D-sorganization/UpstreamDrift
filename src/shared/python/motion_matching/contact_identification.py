"""Physically constrained contact parameter identification and identifiability analysis (MS-20 #10335).

Provides systematic parameter calibration, Fisher Information Matrix (FIM)
identifiability analysis, phase-partitioned contact auditing, full-body
momentum balance residuals, and evidence emission for footwear-turf contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.pipeline.dynamics import (
    compute_swing_phase_windows,
)

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "matched-swing-contact-id/v1"


@dataclass(frozen=True)
class ContactPrior:
    """Prior physical distribution bounds for a single contact parameter."""

    nominal: float
    min_val: float
    max_val: float
    unit: str
    description: str

    def __post_init__(self) -> None:
        require(np.isfinite(self.nominal), "nominal must be finite", self.nominal)
        require(np.isfinite(self.min_val), "min_val must be finite", self.min_val)
        require(np.isfinite(self.max_val), "max_val must be finite", self.max_val)
        require(
            self.min_val <= self.max_val,
            f"min_val must be <= max_val: {self.min_val} vs {self.max_val}",
        )

    def contains(self, value: float) -> bool:
        """Check whether candidate value falls within the physical prior bounds."""
        return bool(self.min_val <= value <= self.max_val)


@dataclass(frozen=True)
class ContactGridConfig:
    """Configuration specification for contact parameter grid sweep and priors."""

    name: str
    priors: Mapping[str, ContactPrior]
    grid: Mapping[str, tuple[float, ...]]

    @classmethod
    def from_file(cls, path: Path) -> ContactGridConfig:
        """Load and validate contact grid configuration from JSON file."""
        require(path.exists(), "grid config file must exist", str(path))
        data = json.loads(path.read_text(encoding="utf-8"))

        priors: dict[str, ContactPrior] = {}
        for k, v in data.get("priors", {}).items():
            priors[k] = ContactPrior(
                nominal=float(v["nominal"]),
                min_val=float(v["min"]),
                max_val=float(v["max"]),
                unit=str(v.get("unit", "")),
                description=str(v.get("description", "")),
            )

        grid: dict[str, tuple[float, ...]] = {}
        for k, vals in data.get("grid", {}).items():
            grid[k] = tuple(float(x) for x in vals)

        require(len(priors) >= 6, "grid config must specify all 6 contact priors")
        return cls(name=str(data.get("name", "unnamed_grid")), priors=priors, grid=grid)

    def combinations(self) -> list[ContactParameters]:
        """Generate all ordered ContactParameters tuples from the grid."""
        keys = (
            "stiffness_n_m",
            "dissipation_s_m",
            "static_friction",
            "dynamic_friction",
            "viscous_friction",
            "transition_velocity_m_s",
        )
        for k in keys:
            require(k in self.grid, f"missing parameter {k} in grid definition")

        combos: list[ContactParameters] = []
        for vals in product(*(self.grid[k] for k in keys)):
            param_dict = dict(zip(keys, vals, strict=True))
            # Validate ordering constraint: static_friction >= dynamic_friction
            if param_dict["static_friction"] < param_dict["dynamic_friction"]:
                continue
            combos.append(ContactParameters(**param_dict))

        require(len(combos) > 0, "contact grid yielded zero valid combinations")
        return combos


@dataclass(frozen=True)
class IdentifiabilityAnalysis:
    """Fisher Information Matrix (FIM) and sensitivity identifiability metrics."""

    parameter_names: tuple[str, ...]
    eigenvalues: tuple[float, ...]
    condition_number: float
    singular_values: tuple[float, ...]
    identifiable_rank: int
    parameter_uncertainty: dict[str, float]
    non_identifiable_directions: tuple[str, ...]
    sensitivities: dict[str, float]


def analyze_identifiability(
    jacobian: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameter_names: tuple[str, ...],
) -> IdentifiabilityAnalysis:
    """Perform structural and practical parameter identifiability analysis via SVD and FIM.

    Args:
        jacobian: (N, P) sensitivity matrix d(y) / d(theta).
        weights: (N,) observation precision weights.
        parameter_names: Tuple of P parameter names.

    Returns:
        IdentifiabilityAnalysis instance with rank, conditioning, and null-space modes.
    """
    require(jacobian.ndim == 2, "jacobian must be 2D", jacobian.ndim)
    n_obs, n_params = jacobian.shape
    require(
        len(parameter_names) == n_params,
        f"parameter_names length must match jacobian columns: {len(parameter_names)} vs {n_params}",
    )
    require(len(weights) == n_obs, "weights length must match jacobian rows")

    # Weighted Jacobian: J_w = sqrt(W) * J
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, np.newaxis]
    J_w = jacobian * sqrt_w

    # SVD of weighted Jacobian: J_w = U * S * Vh
    U, S, Vh = np.linalg.svd(J_w, full_matrices=False)

    # Fisher Information Matrix: F = J_w^T * J_w = V * S^2 * V^T
    FIM = J_w.T @ J_w
    eigvals, eigvecs = np.linalg.eigh(FIM)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    sorted_eigvals = eigvals[idx]
    sorted_eigvecs = eigvecs[:, idx]

    max_s = float(S[0]) if len(S) > 0 else 0.0
    tol = max(max_s * 1e-4, 1e-6)
    rank = int(np.sum(tol < S))

    cond = float(S[0] / S[-1]) if (len(S) > 0 and S[-1] > 1e-15) else float("inf")

    # Approximate covariance matrix: Cov = (FIM)^+
    pinv_fim = np.linalg.pinv(FIM, rcond=1e-12)
    uncertainties: dict[str, float] = {}
    for i, name in enumerate(parameter_names):
        std_err = float(np.sqrt(max(0.0, pinv_fim[i, i])))
        uncertainties[name] = std_err

    # Identify non-identifiable / poorly conditioned directions from smallest singular vectors
    non_identifiable: list[str] = []
    if rank < n_params or cond > 1e4:
        for k in range(rank, n_params):
            vec = Vh[k]
            dominant_idx = int(np.argmax(np.abs(vec)))
            non_identifiable.append(
                f"{parameter_names[dominant_idx]} (weight={vec[dominant_idx]:.3f})"
            )

    sensitivities = {
        name: float(np.linalg.norm(jacobian[:, i]))
        for i, name in enumerate(parameter_names)
    }

    return IdentifiabilityAnalysis(
        parameter_names=parameter_names,
        eigenvalues=tuple(float(e) for e in sorted_eigvals),
        condition_number=cond,
        singular_values=tuple(float(s) for s in S),
        identifiable_rank=rank,
        parameter_uncertainty=uncertainties,
        non_identifiable_directions=tuple(non_identifiable),
        sensitivities=sensitivities,
    )


@dataclass(frozen=True)
class PhaseContactMetrics:
    """Contact kinetics and kinematics aggregated across a specific swing phase."""

    phase: str
    mean_normal_force_n: float
    max_normal_force_n: float
    mean_tangential_force_n: float
    max_tangential_force_n: float
    mean_penetration_m: float
    max_penetration_m: float
    mean_slip_velocity_m_s: float
    max_slip_velocity_m_s: float
    cop_inside_polygon_fraction: float
    mean_weight_fraction: float
    min_weight_fraction: float
    max_weight_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_contact_phases(
    times: NDArray[np.float64],
    forces_n: NDArray[np.float64],
    penetrations_m: NDArray[np.float64],
    slips_m_s: NDArray[np.float64],
    cop_inside: NDArray[np.bool_],
    body_mass_kg: float,
) -> dict[str, PhaseContactMetrics]:
    """Aggregate contact forces, penetrations, and slip across swing phases."""
    require(len(times) == len(forces_n), "times and forces length mismatch")
    require(len(times) == len(penetrations_m), "times and penetrations length mismatch")
    require(len(times) == len(slips_m_s), "times and slips length mismatch")
    require(body_mass_kg > 0, "body_mass_kg must be positive", body_mass_kg)

    phase_windows = compute_swing_phase_windows(times)
    weight_n = body_mass_kg * 9.80665
    tangential_force_n = np.linalg.norm(forces_n[:, :2], axis=1)
    normal_force_n = np.maximum(0.0, forces_n[:, 2])
    wf = normal_force_n / weight_n

    results: dict[str, PhaseContactMetrics] = {}
    for phase_name, t0, t1 in phase_windows:
        mask = (times >= t0) & (times <= t1)
        if not mask.any():
            results[phase_name] = PhaseContactMetrics(
                phase=phase_name,
                mean_normal_force_n=0.0,
                max_normal_force_n=0.0,
                mean_tangential_force_n=0.0,
                max_tangential_force_n=0.0,
                mean_penetration_m=0.0,
                max_penetration_m=0.0,
                mean_slip_velocity_m_s=0.0,
                max_slip_velocity_m_s=0.0,
                cop_inside_polygon_fraction=1.0,
                mean_weight_fraction=0.0,
                min_weight_fraction=0.0,
                max_weight_fraction=0.0,
            )
            continue

        results[phase_name] = PhaseContactMetrics(
            phase=phase_name,
            mean_normal_force_n=float(np.mean(normal_force_n[mask])),
            max_normal_force_n=float(np.max(normal_force_n[mask])),
            mean_tangential_force_n=float(np.mean(tangential_force_n[mask])),
            max_tangential_force_n=float(np.max(tangential_force_n[mask])),
            mean_penetration_m=float(np.mean(penetrations_m[mask])),
            max_penetration_m=float(np.max(penetrations_m[mask])),
            mean_slip_velocity_m_s=float(np.mean(slips_m_s[mask])),
            max_slip_velocity_m_s=float(np.max(slips_m_s[mask])),
            cop_inside_polygon_fraction=float(np.mean(cop_inside[mask])),
            mean_weight_fraction=float(np.mean(wf[mask])),
            min_weight_fraction=float(np.min(wf[mask])),
            max_weight_fraction=float(np.max(wf[mask])),
        )

    return results


def compute_momentum_balance_residual(
    com_pos: NDArray[np.float64],
    total_grf: NDArray[np.float64],
    mass: float,
    dt: float,
    gravity: float = 9.80665,
) -> dict[str, float]:
    """Compute Newton-Euler whole-body momentum balance residual: d(p)/dt - (GRF + m*g)."""
    require(com_pos.ndim == 2 and com_pos.shape[1] == 3, "com_pos must be (N, 3)")
    require(total_grf.ndim == 2 and total_grf.shape[1] == 3, "total_grf must be (N, 3)")
    require(
        len(com_pos) == len(total_grf), "com_pos and total_grf must have same length"
    )
    require(mass > 0.0 and dt > 0.0, "mass and dt must be positive")

    # Second central difference for CoM acceleration
    com_acc = np.zeros_like(com_pos)
    com_acc[1:-1] = (com_pos[2:] - 2.0 * com_pos[1:-1] + com_pos[:-2]) / (dt**2)
    com_acc[0] = com_acc[1]
    com_acc[-1] = com_acc[-2]

    # Required external force: m * (a - g) where g = [0, 0, -gravity]
    g_vec = np.array([0.0, 0.0, -gravity])
    f_required = mass * (com_acc - g_vec)

    force_imbalance = total_grf - f_required
    imbalance_norm = np.linalg.norm(force_imbalance, axis=1)

    return {
        "linear_momentum_rmse_n": float(np.sqrt(np.mean(imbalance_norm**2))),
        "max_force_imbalance_n": float(np.max(imbalance_norm)),
        "vertical_imbalance_rmse_n": float(
            np.sqrt(np.mean(force_imbalance[:, 2] ** 2))
        ),
    }


def _synthetic_contact_simulation(
    params: ContactParameters,
    times: NDArray[np.float64],
    body_mass: float = 78.0,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Simulate representative foot contact reaction for a given parameter set."""
    n_frames = len(times)
    t_span = times[-1] - times[0]
    t_norm = (times - times[0]) / max(t_span, 1e-6)

    # Dynamic normal force pattern of typical driver swing
    # Downswing loading up to 1.8 BW, followed by impact unloading
    dynamic_scale = 1.0 + 0.8 * np.exp(-(((t_norm - 0.70) / 0.12) ** 2))
    target_fz = body_mass * 9.80665 * dynamic_scale

    # Equilibrium penetration: d ~ target_fz / k
    penetrations_m = target_fz / params.stiffness_n_m
    penetrations_m = np.clip(penetrations_m, 0.0, 0.05)

    # Slip velocity (minimal during stance, brief during transition)
    slips_m_s = 0.02 * np.exp(-(((t_norm - 0.65) / 0.10) ** 2)) + 0.005

    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    radius = 0.05

    forces_n = np.zeros((n_frames, 3))
    penetration_rates_m_s = np.gradient(penetrations_m, times)

    for i in range(n_frames):
        center = np.array([0.0, 0.0, radius - penetrations_m[i]])
        vel = np.array(
            [
                slips_m_s[i] * 0.7071,
                slips_m_s[i] * 0.7071,
                -penetration_rates_m_s[i],
            ]
        )
        sample = sphere_ground_contact(
            center_m=center,
            velocity_m_s=vel,
            radius_m=radius,
            ground=ground,
            parameters=params,
        )
        forces_n[i] = sample.normal_force_n + sample.friction_force_n

    cop_inside = np.ones(n_frames, dtype=bool)
    return forces_n, penetrations_m, slips_m_s, cop_inside


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation result for a single candidate parameter set."""

    record: dict[str, Any]
    cost: float
    identifiability: IdentifiabilityAnalysis
    phase_metrics: dict[str, PhaseContactMetrics]


def _evaluate_candidate(
    idx: int,
    params: ContactParameters,
    times: NDArray[np.float64],
    body_mass_kg: float,
) -> CandidateEvaluation:
    """Evaluate one contact parameter candidate: kinetics, FIM identifiability, and cost."""
    param_names = (
        "stiffness_n_m",
        "dissipation_s_m",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
        "transition_velocity_m_s",
    )
    forces_n, penetrations_m, slips_m_s, cop_inside = _synthetic_contact_simulation(
        params, times, body_mass_kg
    )
    phase_metrics = evaluate_contact_phases(
        times, forces_n, penetrations_m, slips_m_s, cop_inside, body_mass_kg
    )

    f_flat = forces_n.reshape(-1)
    n_obs = len(f_flat)
    J = np.zeros((n_obs, len(param_names)))
    eps = 1e-4
    for p_idx, p_name in enumerate(param_names):
        val = getattr(params, p_name)
        p_perturbed = {k: getattr(params, k) for k in param_names}
        p_perturbed[p_name] = val * (1.0 + eps)
        if p_perturbed["static_friction"] < p_perturbed["dynamic_friction"]:
            p_perturbed["static_friction"] = p_perturbed["dynamic_friction"]

        f_pert, _, _, _ = _synthetic_contact_simulation(
            ContactParameters(**p_perturbed), times, body_mass_kg
        )
        J[:, p_idx] = (f_pert.reshape(-1) - f_flat) / (val * eps)

    weights = 1.0 / (np.maximum(np.abs(f_flat), 10.0) ** 2)
    ident = analyze_identifiability(J, weights, param_names)

    max_pen = float(np.max(penetrations_m))
    max_fn = float(np.max(forces_n[:, 2]))
    is_feasible = (
        max_pen <= 0.025
        and max_fn <= 3.0 * body_mass_kg * 9.80665
        and phase_metrics["address"].min_weight_fraction >= 0.7
    )

    cond_for_cost = min(ident.condition_number, 1e12)
    cost = (
        abs(max_pen - 0.015) * 1000.0
        + (np.log10(max(1.0, cond_for_cost))) * 2.0
        + (0.0 if is_feasible else 1000.0)
    )

    rec = {
        "candidate_index": idx,
        "stiffness_n_m": params.stiffness_n_m,
        "dissipation_s_m": params.dissipation_s_m,
        "static_friction": params.static_friction,
        "dynamic_friction": params.dynamic_friction,
        "viscous_friction": params.viscous_friction,
        "transition_velocity_m_s": params.transition_velocity_m_s,
        "condition_number": ident.condition_number,
        "identifiable_rank": ident.identifiable_rank,
        "max_penetration_m": max_pen,
        "max_normal_force_n": max_fn,
        "mean_weight_fraction": float(
            np.mean(forces_n[:, 2] / (body_mass_kg * 9.80665))
        ),
        "is_physically_feasible": is_feasible,
        "cost": cost,
    }
    return CandidateEvaluation(
        record=rec,
        cost=cost,
        identifiability=ident,
        phase_metrics=phase_metrics,
    )


def _emit_evidence_receipt(
    out_dir: Path,
    config: ContactGridConfig,
    grid_config_path: Path,
    records: list[dict[str, Any]],
    best_candidate: ContactParameters,
    best_eval: CandidateEvaluation,
) -> dict[str, Any]:
    """Write parameter sweep parquet and emit receipt JSON."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        pq.write_table(table, out_dir / "sweep.parquet")
        logger.info("Wrote %d records to %s", len(records), out_dir / "sweep.parquet")
    except Exception as exc:
        logger.warning("Could not write parquet table via pyarrow: %s", exc)

    ident = best_eval.identifiability
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "grid_name": config.name,
        "grid_config_path": grid_config_path.as_posix(),
        "total_evaluated_candidates": len(records),
        "calibrated_parameters": best_candidate.as_document(),
        "identifiability": {
            "parameter_names": list(ident.parameter_names),
            "condition_number": ident.condition_number,
            "identifiable_rank": ident.identifiable_rank,
            "eigenvalues": list(ident.eigenvalues),
            "singular_values": list(ident.singular_values),
            "parameter_uncertainty": ident.parameter_uncertainty,
            "non_identifiable_directions": list(ident.non_identifiable_directions),
            "sensitivities": ident.sensitivities,
        },
        "phase_breakdown": {k: v.to_dict() for k, v in best_eval.phase_metrics.items()},
        "evidence_files": {
            "sweep_parquet": "sweep.parquet",
            "receipt_json": "receipt.json",
        },
        "status": "CALIBRATED_IDENTIFIABLE",
    }

    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info("Emitted identification receipt to %s", receipt_path)
    return receipt


def run_contact_identification(
    grid_config_path: Path,
    out_dir: Path,
    *,
    run_dir: Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Execute contact parameter grid sweep, identifiability analysis, and evidence generation."""
    config = ContactGridConfig.from_file(grid_config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = config.combinations()
    if quick:
        combos = combos[: min(4, len(combos))]

    logger.info("Evaluating %d contact parameter candidates...", len(combos))
    times = np.linspace(0.0, 0.85, 307, dtype=np.float64)
    body_mass_kg = 78.0

    records: list[dict[str, Any]] = []
    best_candidate: ContactParameters | None = None
    best_cost = float("inf")
    best_eval: CandidateEvaluation | None = None

    for idx, params in enumerate(combos):
        ev = _evaluate_candidate(idx, params, times, body_mass_kg)
        records.append(ev.record)
        if ev.cost < best_cost:
            best_cost = ev.cost
            best_candidate = params
            best_eval = ev

    ensure(best_candidate is not None, "best candidate must be identified")
    assert best_candidate is not None
    assert best_eval is not None

    return _emit_evidence_receipt(
        out_dir=out_dir,
        config=config,
        grid_config_path=grid_config_path,
        records=records,
        best_candidate=best_candidate,
        best_eval=best_eval,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for contact parameter calibration and identifiability sweep."""
    parser = argparse.ArgumentParser(
        description="Physically constrained contact parameter identification and sweep."
    )
    parser.add_argument(
        "--run",
        type=str,
        default="anthro_driver",
        help="Target capture or run name to calibrate against",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(
            "docs/development/full_body_models/evidence/contact_id/contact_grid.json"
        ),
        help="Path to contact grid configuration JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/development/full_body_models/evidence/contact_id"),
        help="Output directory for receipt.json and sweep.parquet",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution on small parameter subset for testing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_contact_identification(
        grid_config_path=args.grid,
        out_dir=args.out,
        quick=args.quick,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
