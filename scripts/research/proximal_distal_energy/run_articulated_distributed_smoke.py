"""Single-worker atomic runner for the prospective distributed smoke study."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco  # Initialize native plugin handles cleanly
except ImportError:
    mujoco = None

from scripts.research.proximal_distal_energy.articulated_contact_events import (
    ContactEventRecord,
)
from scripts.research.proximal_distal_energy.articulated_distributed_event_attribution import (
    attribute_distributed_contact_trajectory,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_distributed_smoke_registration import (
    EVALUATOR_REVISION,
    REGISTRATION_PATH,
    build_registration,
    registered_smoke_cases,
    validate_registration,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RESULTS_PATH = DATA_DIR / "articulated_distributed_smoke_results.json"
DEFAULT_CHECKPOINT_DIR = DATA_DIR / ".smoke_checkpoints"


@dataclass(frozen=True, slots=True)
class CaseExecutionResult:
    """Outcome metrics from a single smoke case execution."""

    case_id: str
    engine: str
    time_step_s: float
    status: str
    event_count: int
    event_kinds: tuple[str, ...]
    events_detail: list[dict[str, Any]]
    maximum_absolute_gap_residual_m: float
    maximum_final_bracket_width_s: float
    maximum_force_closure_residual: float
    total_discrete_event_impulse: float
    total_discrete_event_work_j: float
    momentum_change: list[float]
    kinetic_energy_change_j: float
    continuous_work_j: float
    work_closure_residual_j: float
    generalized_work_j: list[float]
    impulse_shares: list[list[float]]
    work_shares: list[float]
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class DistributedSmokeSummary:
    """Aggregated outcome of the complete prospective smoke matrix."""

    registration_path: str
    evaluator_revision: str
    case_results: list[CaseExecutionResult]
    execution_timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.execution_timestamp:
            object.__setattr__(
                self,
                "execution_timestamp",
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )


def _to_serializable(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, tuple):
        return [_to_serializable(x) for x in val]
    if isinstance(val, list):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    return val


def execute_smoke_case(
    case_spec: Mapping[str, Any], root: Path = ROOT
) -> CaseExecutionResult:
    """Execute a single registered smoke case and return structured metrics."""
    root = root.resolve()
    case_id = str(case_spec.get("case_id", "unknown_case"))
    engine = str(case_spec.get("engine", "unknown"))
    time_step_s = float(case_spec.get("time_step_s", 0.001))

    try:
        if engine not in ("mujoco", "pinocchio"):
            raise ValueError(f"unknown or unsupported engine: {engine}")

        model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
        npz_path = (
            root
            / "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz"
        )
        with np.load(npz_path) as source:
            source_case_idx = int(case_spec.get("source_case_index", 0))
            source_sample_idx = int(case_spec.get("source_sample_index", 6))
            q = np.asarray(
                source["solution_q"][source_case_idx, source_sample_idx], dtype=float
            )
            grip_span_m = float(source["case_grip_span_m"][source_case_idx])

        grip = DistributedGripConfig(
            station_count_per_hand=1,
            station_width_m=0.0,
            slack_distance_m=float(case_spec.get("slack_distance_m", 0.0015)),
            friction_coefficient=0.0,
        )

        case = DistributedIntegrationCase(
            q=q,
            qd=np.zeros(model.nq),
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            time_step_s=time_step_s,
            initial_club_displacement_m=float(
                case_spec.get("initial_club_displacement_m", 0.001)
            ),
            initial_club_velocity_m_s=float(
                case_spec.get("initial_club_velocity_m_s", -0.8)
            ),
            engine=engine,
            grip=grip,
        )

        config = DistributedForwardConfig(
            duration_s=0.05,
            time_steps_s=(0.001, 0.0005, 0.00025),
        )

        evidence = attribute_distributed_contact_trajectory(
            model=model,
            case=case,
            config=config,
        )

        events = evidence.events
        event_kinds = tuple(
            e.kind.value if hasattr(e.kind, "value") else str(e.kind) for e in events
        )
        events_detail = [
            {
                "kind": e.kind.value if hasattr(e.kind, "value") else str(e.kind),
                "time_s": float(e.time_s),
                "gap_residual_m": float(e.gap_residual_m),
                "final_bracket_width_s": float(e.final_bracket_width_s),
                "path_model": str(e.path_model),
            }
            for e in events
        ]

        max_gap_residual = max((abs(e.gap_residual_m) for e in events), default=0.0)
        max_bracket_width = max((e.final_bracket_width_s for e in events), default=0.0)
        max_force_closure = float(
            np.max(np.abs(evidence.pointwise_force_closure_residual))
        )

        attr = evidence.attribution
        total_event_impulse = float(np.sum(np.abs(attr.total_event_impulse)))
        total_event_work = float(abs(attr.total_event_work_j))

        return CaseExecutionResult(
            case_id=case_id,
            engine=engine,
            time_step_s=time_step_s,
            status="completed",
            event_count=len(events),
            event_kinds=event_kinds,
            events_detail=events_detail,
            maximum_absolute_gap_residual_m=max_gap_residual,
            maximum_final_bracket_width_s=max_bracket_width,
            maximum_force_closure_residual=max_force_closure,
            total_discrete_event_impulse=total_event_impulse,
            total_discrete_event_work_j=total_event_work,
            momentum_change=[float(x) for x in attr.momentum_change],
            kinetic_energy_change_j=float(attr.kinetic_energy_change_j),
            continuous_work_j=float(attr.continuous_work_j),
            work_closure_residual_j=float(attr.work_closure_residual_j),
            generalized_work_j=[float(x) for x in attr.generalized_work_j],
            impulse_shares=[[float(y) for y in x] for x in attr.impulse_shares],
            work_shares=[float(x) for x in attr.work_shares],
        )
    except Exception as exc:
        logger.exception("Failed to execute smoke case %s: %s", case_id, exc)
        return CaseExecutionResult(
            case_id=case_id,
            engine=engine,
            time_step_s=time_step_s,
            status="failed",
            event_count=0,
            event_kinds=(),
            events_detail=[],
            maximum_absolute_gap_residual_m=float("nan"),
            maximum_final_bracket_width_s=float("nan"),
            maximum_force_closure_residual=float("nan"),
            total_discrete_event_impulse=float("nan"),
            total_discrete_event_work_j=float("nan"),
            momentum_change=[],
            kinetic_energy_change_j=float("nan"),
            continuous_work_j=float("nan"),
            work_closure_residual_j=float("nan"),
            generalized_work_j=[],
            impulse_shares=[],
            work_shares=[],
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


class DistributedSmokeRunner:
    """Atomic, single-worker runner reading prospective registration."""

    def __init__(
        self,
        root: Path = ROOT,
        registration_path: Path = REGISTRATION_PATH,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._root = root.resolve()
        self._registration_path = registration_path
        self._checkpoint_dir = checkpoint_dir or (
            self._root
            / "docs/research/proximal_distal_energy_transfer/data/.smoke_checkpoints"
        )

    def enforce_thread_limits(self) -> None:
        """Enforce single-threaded numerical backend execution."""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    def load_and_validate_registration(
        self, path: Path | None = None
    ) -> dict[str, Any]:
        """Validate registration immutability and return contents."""
        target = path or self._registration_path
        content = json.loads(target.read_text(encoding="utf-8"))
        validate_registration(content, self._root)
        return content

    @property
    def cases(self) -> tuple[dict[str, Any], ...]:
        """Return registered cases in frozen order."""
        registration = self.load_and_validate_registration()
        return registered_smoke_cases(registration, self._root)

    def _load_checkpoint(self, case_id: str) -> CaseExecutionResult | None:
        cp_file = self._checkpoint_dir / f"{case_id}.json"
        if not cp_file.exists():
            return None
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            return CaseExecutionResult(
                case_id=data["case_id"],
                engine=data["engine"],
                time_step_s=float(data["time_step_s"]),
                status=data["status"],
                event_count=int(data["event_count"]),
                event_kinds=tuple(data["event_kinds"]),
                events_detail=data["events_detail"],
                maximum_absolute_gap_residual_m=float(
                    data["maximum_absolute_gap_residual_m"]
                ),
                maximum_final_bracket_width_s=float(
                    data["maximum_final_bracket_width_s"]
                ),
                maximum_force_closure_residual=float(
                    data["maximum_force_closure_residual"]
                ),
                total_discrete_event_impulse=float(
                    data["total_discrete_event_impulse"]
                ),
                total_discrete_event_work_j=float(data["total_discrete_event_work_j"]),
                momentum_change=data["momentum_change"],
                kinetic_energy_change_j=float(data["kinetic_energy_change_j"]),
                continuous_work_j=float(data["continuous_work_j"]),
                work_closure_residual_j=float(data["work_closure_residual_j"]),
                generalized_work_j=data["generalized_work_j"],
                impulse_shares=data["impulse_shares"],
                work_shares=data["work_shares"],
                error_type=data.get("error_type"),
                error_message=data.get("error_message"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            logger.warning("Checkpoint %s corrupt or unreadable; re-running.", cp_file)
            return None

    def _save_checkpoint(self, result: CaseExecutionResult) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp_file = self._checkpoint_dir / f"{result.case_id}.json"
        tmp_file = self._checkpoint_dir / f".tmp_{result.case_id}.json"
        tmp_file.write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )
        tmp_file.replace(cp_file)

    def run_cases(
        self, cases: Sequence[dict[str, Any]] | None = None
    ) -> DistributedSmokeSummary:
        """Execute all target cases with atomic persistence."""
        self.enforce_thread_limits()
        target_cases = tuple(cases) if cases is not None else self.cases
        results: list[CaseExecutionResult] = []

        for case_spec in target_cases:
            case_id = str(case_spec.get("case_id", ""))
            cached = self._load_checkpoint(case_id)
            if cached is not None:
                results.append(cached)
                continue

            result = execute_smoke_case(case_spec, root=self._root)
            self._save_checkpoint(result)
            results.append(result)

        return DistributedSmokeSummary(
            registration_path=self._registration_path.relative_to(
                self._root
            ).as_posix(),
            evaluator_revision=EVALUATOR_REVISION,
            case_results=results,
        )


def qualify_smoke_results(
    summary: DistributedSmokeSummary, root: Path = ROOT
) -> dict[str, Any]:
    """Evaluate completed smoke outcomes against registered acceptance gates."""
    case_results = summary.case_results
    case_count = len(case_results)
    completed = [r for r in case_results if r.status == "completed"]
    failed = [r for r in case_results if r.status != "completed"]

    all_completed = len(completed) == case_count and case_count == 6
    all_events_bracketed = all_completed and all(
        r.maximum_absolute_gap_residual_m <= 1.0e-10
        and r.maximum_final_bracket_width_s <= 1.0e-12
        for r in completed
    )
    all_force_closures = all_completed and all(
        r.maximum_force_closure_residual <= 1.0e-12 for r in completed
    )
    all_discrete_impulses_zero = all_completed and all(
        r.total_discrete_event_impulse == 0.0 for r in completed
    )
    all_discrete_works_zero = all_completed and all(
        r.total_discrete_event_work_j == 0.0 for r in completed
    )

    qualified = bool(
        all_completed
        and all_events_bracketed
        and all_force_closures
        and all_discrete_impulses_zero
        and all_discrete_works_zero
    )

    return {
        "qualified": qualified,
        "case_count": case_count,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "all_events_bracketed": all_events_bracketed,
        "all_force_closures_pass": all_force_closures,
        "all_discrete_impulses_zero": all_discrete_impulses_zero,
        "all_discrete_works_zero": all_discrete_works_zero,
        "promotion_eligible": False,
        "promotion_authority": "none_from_smoke_execution",
    }


def run_distributed_smoke(
    root: Path = ROOT,
    checkpoint_dir: Path | None = None,
    write_results: bool = True,
) -> dict[str, Any]:
    """Execute the registered smoke cases, qualify outcomes, and write results."""
    runner = DistributedSmokeRunner(root=root, checkpoint_dir=checkpoint_dir)
    summary = runner.run_cases()
    qualification = qualify_smoke_results(summary, root=root)

    record = {
        "schema_version": "1.0.0",
        "study_type": "articulated_distributed_smoke",
        "registration_authority": summary.registration_path,
        "evaluator_revision": summary.evaluator_revision,
        "execution_timestamp": summary.execution_timestamp,
        "qualification": qualification,
        "cases": [asdict(r) for r in summary.case_results],
    }

    if write_results:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(
            json.dumps(_to_serializable(record), indent=2) + "\n",
            encoding="utf-8",
        )

    return record


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate", "status"))
    return parser


def main() -> None:
    """CLI entrypoint for smoke study execution and validation."""
    action = _parser().parse_args().action
    if action == "run":
        record = run_distributed_smoke(write_results=True)
        print(f"Results written to: {RESULTS_PATH}")
        print(json.dumps(record["qualification"], indent=2))
    elif action == "validate":
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"results file {RESULTS_PATH} does not exist")
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        case_results = [
            CaseExecutionResult(
                case_id=c["case_id"],
                engine=c["engine"],
                time_step_s=float(c["time_step_s"]),
                status=c["status"],
                event_count=int(c["event_count"]),
                event_kinds=tuple(c["event_kinds"]),
                events_detail=c["events_detail"],
                maximum_absolute_gap_residual_m=float(
                    c["maximum_absolute_gap_residual_m"]
                ),
                maximum_final_bracket_width_s=float(c["maximum_final_bracket_width_s"]),
                maximum_force_closure_residual=float(
                    c["maximum_force_closure_residual"]
                ),
                total_discrete_event_impulse=float(c["total_discrete_event_impulse"]),
                total_discrete_event_work_j=float(c["total_discrete_event_work_j"]),
                momentum_change=c["momentum_change"],
                kinetic_energy_change_j=float(c["kinetic_energy_change_j"]),
                continuous_work_j=float(c["continuous_work_j"]),
                work_closure_residual_j=float(c["work_closure_residual_j"]),
                generalized_work_j=c["generalized_work_j"],
                impulse_shares=c["impulse_shares"],
                work_shares=c["work_shares"],
                error_type=c.get("error_type"),
                error_message=c.get("error_message"),
            )
            for c in data["cases"]
        ]
        summary = DistributedSmokeSummary(
            registration_path=data["registration_authority"],
            evaluator_revision=data["evaluator_revision"],
            case_results=case_results,
            execution_timestamp=data.get("execution_timestamp", ""),
        )
        qual = qualify_smoke_results(summary)
        if not qual["qualified"]:
            raise ValueError(f"smoke qualification failed: {qual}")
        print(json.dumps(qual, indent=2))
    elif action == "status":
        if not RESULTS_PATH.exists():
            print(json.dumps({"status": "not_executed"}, indent=2))
        else:
            data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            print(json.dumps(data["qualification"], indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CaseExecutionResult",
    "DistributedSmokeRunner",
    "DistributedSmokeSummary",
    "RESULTS_PATH",
    "execute_smoke_case",
    "qualify_smoke_results",
    "run_distributed_smoke",
]
