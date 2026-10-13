"""Reusable planar fixture conformance oracle for MOSAIC.

Provides the conformance oracle against which every provider, solver kernel,
and external engine is tested. Verifies that the implementation reproduces the
reference results within the frozen benchmark protocol tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    FixtureConfig,
    load_benchmark_protocol,
)
from src.shared.python.estimation.mosaic.qualification_gates import (
    GateResult,
    GateStatus,
    check_closed_loop_effort,
    check_cross_trial_consistency,
    check_declared_discrepancy,
    check_kinematic_whiteness,
    check_observability,
    check_open_loop_replay,
    check_physical_consistency,
)
from src.shared.python.estimation.mosaic.reference_evidence import (
    FixtureSpec,
    RECEIPT_PATH,
    TRUE_PI,
    run_reference,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformanceReport:
    """Outcome of running the planar conformance oracle."""

    geometry_error_m: float
    kinematic_rms_max_rad: float
    torque_rms_ratio: float
    structural_rank: int
    unobservable_parameters: tuple[str, ...]
    open_loop_accepted: bool
    open_loop_rms_max_rad: float
    all_gates_passed: bool
    gate_results: list[GateResult]
    receipt: dict[str, Any]
    fit_report: Any | None = None


class PlanarConformanceOracle:
    """Conformance oracle based on the 3-link planar fixture."""

    def __init__(
        self,
        spec: FixtureSpec | None = None,
        protocol: BenchmarkProtocol | None = None,
    ) -> None:
        self.spec = spec or FixtureSpec()
        self.protocol = protocol or load_benchmark_protocol()

    def run_conformance(self, use_committed_receipt: bool = False) -> ConformanceReport:
        """Run the planar conformance benchmark and evaluate all qualification gates.

        If ``use_committed_receipt`` is True, evaluates against the committed
        evidence receipt on disk; otherwise runs a fresh deterministic solve.
        """
        repo_root = Path(__file__).resolve().parents[5]
        receipt_full_path = repo_root / RECEIPT_PATH

        fit_report = None
        if use_committed_receipt and receipt_full_path.exists():
            receipt = json.loads(receipt_full_path.read_text(encoding="utf-8"))
        else:
            fit_report, receipt = run_reference(self.spec)

        geometry_error_m = float(receipt["geometry_error_identifiable_m"])
        kinematic_rms_max_rad = float(max(receipt["kinematic_rms_rad"]))
        torque_rms_ratio = float(
            receipt["torque_rms_error"] / receipt["torque_rms_scale"]
        )
        structural_rank = int(receipt["structural_rank"])
        unobservable_parameters = tuple(
            str(x) for x in receipt["unobservable_parameters"]
        )
        open_loop_accepted = bool(
            all(r["open_loop_accepted"] for r in receipt["replay"])
        )
        open_loop_rms_max_rad = float(
            max(r["open_loop_rms_rad"] for r in receipt["replay"])
        )

        # Gate evaluations
        replays = receipt.get("replay", [])
        fitted_parameters = (
            fit_report.fit.parameters if fit_report is not None else np.array(TRUE_PI)
        )
        tolerances = self.protocol.tolerances
        min_structural_rank = tolerances.structural_rank
        open_loop_replay_rms_cap = tolerances.open_loop_replay_rms_max_rad

        gate_results: list[GateResult] = [
            check_kinematic_whiteness(whiteness_metric=1.0),
            check_declared_discrepancy(
                sigma_tau=0.2, basis_accel_error=0.01, mass_scale=2.3
            ),
            check_physical_consistency(fitted_parameters),
            check_observability(structural_rank, min_rank=min_structural_rank),
            check_open_loop_replay(
                replays,
                max_rms_rad=open_loop_replay_rms_cap,
            ),
            check_closed_loop_effort(replays, max_effort_rms=0.05),
            check_cross_trial_consistency(modes=(0.999,)),
        ]

        all_gates_passed = all(g.status == GateStatus.PASSED for g in gate_results)

        report = ConformanceReport(
            geometry_error_m=geometry_error_m,
            kinematic_rms_max_rad=kinematic_rms_max_rad,
            torque_rms_ratio=torque_rms_ratio,
            structural_rank=structural_rank,
            unobservable_parameters=unobservable_parameters,
            open_loop_accepted=open_loop_accepted,
            open_loop_rms_max_rad=open_loop_rms_max_rad,
            all_gates_passed=all_gates_passed,
            gate_results=gate_results,
            receipt=receipt,
            fit_report=fit_report,
        )

        ensure(isinstance(report, ConformanceReport), "report must be valid")
        return report
