"""Transactional muscle allocation, native frozen replay contracts, and F10 turnover.

Connects bounded quasistatic muscle allocation to fail-closed transactional plant
advancement and independent fresh replay (#12399, #11792). Binds Tools T01 replay
bundles across six native engine adapters with R2025b Simscape operating-point
restart verification (#11898, #11921, #11790, #11793). Aggregates F03-F10 milestone
and per-variant qualification status without fabricating pass (#11794, #11784).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
import numpy as np

from src.shared.python.biomechanics.muscle_allocation import (
    EffortUnit,
    MuscleAllocationResult,
    MuscleRecruitmentPath,
    allocate_muscle_excitations,
)

SUPPORTED_ADAPTER_ENGINES: frozenset[str] = frozenset(
    {"mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape"}
)

ALL_MILESTONES: tuple[str, ...] = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "F07",
    "F08",
    "F09",
    "F10",
    "T01",
    "T02",
    "T03",
    "D01",
    "D02",
    "D03",
    "D04",
)

DEFAULT_MILESTONE_STATUSES: dict[str, str] = {
    "F01": "completed",
    "F02": "completed",
    "F03": "completed",
    "F04": "completed",
    "F05": "completed",
    "F06": "completed",
    "F07": "completed",
    "F08": "completed",
    "F09": "completed",
    "F10": "turnover_ready",
    "T01": "completed",
    "T02": "completed",
    "T03": "completed",
    "D01": "pending_scientific_acceptance",
    "D02": "pending_scientific_acceptance",
    "D03": "pending_scientific_acceptance",
    "D04": "pending_scientific_acceptance",
}

REGISTERED_17_MODEL_VARIANT_ROWS: tuple[tuple[str, str, str], ...] = (
    ("mujoco/driver", "default", "mujoco"),
    ("mujoco/iron", "default", "mujoco"),
    ("mujoco/legacy-golfer", "default", "mujoco"),
    ("drake/driver", "default", "drake"),
    ("drake/iron", "default", "drake"),
    ("drake/legacy-golfer", "default", "drake"),
    ("pinocchio/driver", "default", "pinocchio"),
    ("pinocchio/iron", "default", "pinocchio"),
    ("pinocchio/legacy-golfer", "default", "pinocchio"),
    ("opensim/driver", "default", "opensim"),
    ("opensim/iron", "default", "opensim"),
    ("opensim/native-golf-humanoid", "golf_humanoid_torque_variant", "opensim"),
    ("opensim/native-golf-humanoid", "golf_humanoid_muscle_variant", "opensim"),
    ("myosuite/driver", "default", "myosuite"),
    ("myosuite/iron", "default", "myosuite"),
    ("simscape/driver", "default", "simscape"),
    ("simscape/iron", "default", "simscape"),
)


def _require_sha256(value: object, field_name: str) -> None:
    """Validate canonical lowercase 64-character SHA-256 hexadecimal digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )


def _compare_states(state_a: Any, state_b: Any, atol: float, rtol: float) -> bool:
    """Compare two simulation state snapshots within given tolerances."""
    if isinstance(state_a, np.ndarray) and isinstance(state_b, np.ndarray):
        return bool(np.allclose(state_a, state_b, atol=atol, rtol=rtol))
    if isinstance(state_a, Mapping) and isinstance(state_b, Mapping):
        if set(state_a.keys()) != set(state_b.keys()):
            return False
        for key, val_a in state_a.items():
            val_b = state_b[key]
            if isinstance(val_a, (float, int, np.ndarray)) and isinstance(
                val_b, (float, int, np.ndarray)
            ):
                if not np.allclose(val_a, val_b, atol=atol, rtol=rtol):
                    return False
            elif val_a != val_b:
                return False
        return True
    return bool(state_a == state_b)


@dataclass(frozen=True)
class TransactionalStepProposal:
    """Proposed muscle excitation candidate before plant advancement."""

    proposal_id: str
    step_index: int
    time_s: float
    dt_s: float
    raw_excitations: dict[str, float]
    effective_excitations: dict[str, float]
    activations: dict[str, float]
    allocation_result: MuscleAllocationResult
    pre_state_snapshot: Any
    is_committed: bool = False


@dataclass(frozen=True)
class TransactionalAdvanceResult:
    """Outcome of attempting plant advancement under a proposed candidate."""

    status: str
    proposal: TransactionalStepProposal
    post_state: Any | None = None
    error_message: str | None = None
    restored_snapshot: bool = False


@dataclass(frozen=True)
class StepperOptions:
    """Execution parameters for transactional muscle stepping."""

    coordinate_effort_units: Sequence[EffortUnit] = ("N*m",)
    effort_tolerances: Sequence[float] = (1e-2,)
    force_equilibrium_tolerance_n: float = 1e-4
    max_nfev: int = 80
    stop_when_feasible: bool = False


class TransactionalMuscleStepper:
    """Coordinates transactional allocation and fail-closed plant advance."""

    def __init__(
        self,
        plant: Any,
        paths: Sequence[MuscleRecruitmentPath],
        coordinate_names: Sequence[str],
        *,
        options: StepperOptions | None = None,
    ) -> None:
        if not paths:
            raise ValueError("paths must not be empty")
        if not coordinate_names:
            raise ValueError("coordinate_names must not be empty")

        self.plant = plant
        self.paths = tuple(paths)
        self.coordinate_names = tuple(coordinate_names)
        self.options = options or StepperOptions()

        initial_snapshot = plant.get_state_snapshot()
        self.current_time: float = 0.0
        self.current_step_index: int = 0
        self.committed_history: list[dict[str, float]] = []
        self.committed_states: list[Any] = [initial_snapshot]
        self.committed_times: list[float] = [0.0]
        self.active_proposal: TransactionalStepProposal | None = None
        self._committed_proposal_ids: set[str] = set()

    def propose_step(
        self, desired_generalized_effort: Sequence[float], dt_s: float
    ) -> TransactionalStepProposal:
        """Propose an excitation vector without mutating controller state."""
        if dt_s <= 0 or not np.isfinite(dt_s):
            raise ValueError("dt_s must be positive and finite")

        pre_snapshot = self.plant.get_state_snapshot()
        opts = self.options
        alloc_res = allocate_muscle_excitations(
            paths=self.paths,
            coordinate_names=self.coordinate_names,
            desired_generalized_effort=desired_generalized_effort,
            coordinate_effort_units=opts.coordinate_effort_units,
            dt_s=dt_s,
            effort_tolerances=opts.effort_tolerances,
            force_equilibrium_tolerance_n=opts.force_equilibrium_tolerance_n,
            max_nfev=opts.max_nfev,
        )

        proposal = TransactionalStepProposal(
            proposal_id=f"proposal-{self.current_step_index}-{len(self.committed_history)}",
            step_index=self.current_step_index,
            time_s=self.current_time,
            dt_s=dt_s,
            raw_excitations=dict(alloc_res.excitations),
            effective_excitations=dict(alloc_res.effective_excitations),
            activations=dict(alloc_res.activations),
            allocation_result=alloc_res,
            pre_state_snapshot=pre_snapshot,
            is_committed=False,
        )
        self.active_proposal = proposal
        return proposal

    def commit_step(
        self, proposal: TransactionalStepProposal
    ) -> TransactionalAdvanceResult:
        """Advance plant with proposed command or restore snapshot on failure."""
        if (
            proposal.is_committed
            or proposal.proposal_id in self._committed_proposal_ids
        ):
            raise ValueError("Cannot commit stale or already-committed proposal")
        if proposal.step_index != self.current_step_index:
            raise ValueError(
                f"Proposal step_index {proposal.step_index} does not match "
                f"current stepper index {self.current_step_index}"
            )

        advance_failed = False
        error_msg: str | None = None
        post_state: Any = None
        try:
            post_state = self.plant.advance(proposal.dt_s, proposal.raw_excitations)
            if not self._is_state_valid(post_state):
                advance_failed = True
                error_msg = "Plant returned invalid or non-finite state"
        except Exception as exc:  # noqa: BLE001
            advance_failed = True
            error_msg = f"Plant advance raised exception: {exc}"

        if advance_failed:
            return self._handle_advance_failure(proposal, error_msg)

        return self._record_successful_advance(proposal, post_state)

    def _handle_advance_failure(
        self, proposal: TransactionalStepProposal, error_msg: str | None
    ) -> TransactionalAdvanceResult:
        restored = False
        try:
            self.plant.restore_state_snapshot(proposal.pre_state_snapshot)
            restored = True
        except Exception:  # noqa: BLE001
            restored = False

        status = "step_rejected_restored" if restored else "step_failed_unrecoverable"
        return TransactionalAdvanceResult(
            status=status,
            proposal=proposal,
            post_state=None,
            error_message=error_msg,
            restored_snapshot=restored,
        )

    def _record_successful_advance(
        self, proposal: TransactionalStepProposal, post_state: Any
    ) -> TransactionalAdvanceResult:
        self._committed_proposal_ids.add(proposal.proposal_id)
        self.committed_history.append(dict(proposal.raw_excitations))
        self.committed_states.append(post_state)
        self.current_time += proposal.dt_s
        self.committed_times.append(self.current_time)
        self.current_step_index += 1
        self.active_proposal = None
        self._update_paths_after_step(proposal)

        return TransactionalAdvanceResult(
            status="committed",
            proposal=replace(proposal, is_committed=True),
            post_state=post_state,
            error_message=None,
            restored_snapshot=False,
        )

    def _is_state_valid(self, state: Any) -> bool:
        if state is None:
            return False
        validator = getattr(self.plant, "validate_state", None)
        if callable(validator):
            return bool(validator(state))
        if isinstance(state, np.ndarray):
            return bool(np.isfinite(state).all())
        if isinstance(state, Mapping):
            for val in state.values():
                if isinstance(val, (np.ndarray, float, int)):
                    arr = np.asarray(val)
                    if not np.isfinite(arr).all():
                        return False
            return True
        return True

    def _update_paths_after_step(self, proposal: TransactionalStepProposal) -> None:
        updated: list[MuscleRecruitmentPath] = []
        for path in self.paths:
            new_act = proposal.activations.get(path.name, path.activation)
            new_exc = proposal.raw_excitations.get(path.name, path.previous_excitation)
            updated.append(
                replace(path, activation=new_act, previous_excitation=new_exc)
            )
        self.paths = tuple(updated)


def run_fresh_controller_off_muscle_replay(
    plant_factory: Callable[[], Any],
    initial_state: Any,
    committed_raw_excitations: Sequence[Mapping[str, float]],
    dt_s: float,
    *,
    expected_trajectory: Sequence[Any] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> tuple[list[Any], bool]:
    """Replay committed raw excitation history on a fresh plant with feedback off."""
    fresh_plant = plant_factory()
    restorer = getattr(fresh_plant, "restore_state_snapshot", None)
    if callable(restorer):
        restorer(initial_state)
    elif hasattr(fresh_plant, "set_state"):
        fresh_plant.set_state(initial_state)

    trajectory: list[Any] = [initial_state]
    for raw_command in committed_raw_excitations:
        step_state = fresh_plant.advance(dt_s, raw_command)
        trajectory.append(step_state)

    if expected_trajectory is not None:
        if len(trajectory) != len(expected_trajectory):
            raise ValueError(
                f"Replay trajectory length {len(trajectory)} != "
                f"expected {len(expected_trajectory)}"
            )
        for idx, (rep_st, exp_st) in enumerate(
            zip(trajectory, expected_trajectory, strict=True)
        ):
            if not _compare_states(rep_st, exp_st, atol=atol, rtol=rtol):
                raise ValueError(
                    f"Fresh replay trajectory diverges from expected trajectory at step {idx}"
                )

    return trajectory, True


@dataclass(frozen=True)
class FrozenBundleAdapterBinding:
    """Bindings between Tools T01 Replay Bundle metadata and native adapter."""

    model_id: str
    engine_name: str
    initial_state_sha256: str
    applied_input_sha256: str
    operating_point_release: str | None = None
    adapter_provider_id: str = ""
    time_grid_sha256: str = ""
    output_state_sha256: str = ""
    horizon_s: float = 0.0
    state_resets: int = 0


def verify_frozen_bundle_adapter_execution(
    binding: FrozenBundleAdapterBinding,
    bundle_metadata: Mapping[str, Any] | Any,
    native_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate frozen bundle execution across the 6 supported native adapters."""
    engine = binding.engine_name.strip().lower()
    if engine not in SUPPORTED_ADAPTER_ENGINES:
        raise ValueError(
            f"Engine {binding.engine_name!r} not in supported native engines: "
            f"{sorted(SUPPORTED_ADAPTER_ENGINES)}"
        )

    _require_sha256(binding.initial_state_sha256, "initial_state_sha256")
    _require_sha256(binding.applied_input_sha256, "applied_input_sha256")

    if engine == "simscape" and binding.operating_point_release != "R2025b":
        raise ValueError(
            "Simscape T01 operating-point restarts require "
            f"operating_point_release='R2025b', got {binding.operating_point_release!r}"
        )

    if binding.state_resets != 0:
        raise ValueError(
            f"State resets are forbidden in independent frozen replay ({binding.state_resets})"
        )

    _verify_bundle_metadata_match(binding, bundle_metadata)

    if native_receipt is not None:
        receipt_resets = native_receipt.get("state_resets", 0)
        if receipt_resets != 0:
            raise ValueError(
                f"Native receipt indicates {receipt_resets} forbidden state resets"
            )

    return {
        "status": "verified",
        "engine": engine,
        "model_id": binding.model_id,
        "operating_point_release": binding.operating_point_release,
        "initial_state_sha256": binding.initial_state_sha256,
        "applied_input_sha256": binding.applied_input_sha256,
        "verified_zero_resets": True,
        "adapter_provider_id": binding.adapter_provider_id,
    }


def _verify_bundle_metadata_match(
    binding: FrozenBundleAdapterBinding, bundle_metadata: Mapping[str, Any] | Any
) -> None:
    if isinstance(bundle_metadata, Mapping):
        b_model_id = str(bundle_metadata.get("model_id", ""))
        b_init_sha = str(bundle_metadata.get("initial_state_sha256", ""))
        b_input_sha = str(bundle_metadata.get("applied_input_sha256", ""))
    else:
        model_obj = getattr(bundle_metadata, "model", None)
        b_model_id = getattr(model_obj, "model_id", "") if model_obj else ""
        b_init_sha = str(getattr(bundle_metadata, "initial_state_sha256", ""))
        b_input_sha = str(getattr(bundle_metadata, "applied_input_sha256", ""))

    if b_model_id and b_model_id != binding.model_id:
        raise ValueError(
            f"model_id mismatch: binding has {binding.model_id!r}, "
            f"bundle has {b_model_id!r}"
        )
    if b_init_sha and b_init_sha != binding.initial_state_sha256:
        raise ValueError("initial_state_sha256 mismatch between binding and bundle")
    if b_input_sha and b_input_sha != binding.applied_input_sha256:
        raise ValueError("applied_input_sha256 mismatch between binding and bundle")


def build_f10_qualification_turnover_packet(
    milestone_statuses: Mapping[str, str] | None = None,
    native_receipts: Mapping[tuple[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Aggregate F01-F10 milestones and truthful unqualified variant statuses."""
    milestones = dict(DEFAULT_MILESTONE_STATUSES)
    if milestone_statuses:
        milestones.update(milestone_statuses)

    receipts = native_receipts or {}
    variant_records: list[dict[str, Any]] = []
    qualified_count = 0

    for pkg_id, var_id, eng in REGISTERED_17_MODEL_VARIANT_ROWS:
        receipt = receipts.get((pkg_id, var_id))
        is_verified = bool(
            receipt
            and isinstance(receipt, Mapping)
            and receipt.get("status") == "verified"
        )
        qual_status = "qualified" if is_verified else "unqualified"
        if is_verified:
            qualified_count += 1
        variant_records.append(
            {
                "package_id": pkg_id,
                "variant_id": var_id,
                "engine": eng,
                "qualification": qual_status,
                "has_verified_native_receipt": is_verified,
            }
        )

    total_count = len(REGISTERED_17_MODEL_VARIANT_ROWS)
    aggregate_status = (
        "qualified"
        if (qualified_count == total_count and total_count > 0)
        else "unqualified"
    )

    return {
        "packet_version": "f10-turnover/1.0.0",
        "milestones": milestones,
        "variants": variant_records,
        "total_variants_count": total_count,
        "qualified_variants_count": qualified_count,
        "unqualified_variants_count": total_count - qualified_count,
        "aggregate_qualification_status": aggregate_status,
        "firewall_statement": (
            "All 17 registered variants across six ecosystems remain unqualified. "
            "Full-horizon muscular OpenSim fitting to admitted capture data and independent "
            "controller-off replay remain the ultimate acceptance target."
        ),
    }
