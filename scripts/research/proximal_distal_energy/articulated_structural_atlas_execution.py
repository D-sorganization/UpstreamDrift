"""Checkpointed execution of governed structural authorities through both atlases."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy import articulated_ground_atlas as ground
from scripts.research.proximal_distal_energy import articulated_shaft_atlas as shaft
from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_atlas_runtime_authority import (
    resolve_atlas_states,
)
from scripts.research.proximal_distal_energy.articulated_structural_branch_contract import (
    structural_branch_contracts,
)
from scripts.research.proximal_distal_energy.articulated_structural_checkpoint import (
    Array,
    audit_structural_checkpoint_directory,
    restore_structural_checkpoint_directory,
    structural_checkpoint_path,
    write_structural_checkpoint,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    StructuralExecutionIdentity,
    resolve_structural_execution_identity,
)
from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
    DEFAULT_OUTPUT,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


@dataclass(frozen=True, slots=True)
class StructuralAtlasExecution:
    """One complete, gate-evaluated corner/pathway atlas execution."""

    record: dict[str, Any]
    arrays: dict[str, NDArray[Any]]
    checkpoint_audit: dict[str, Any]


def _buffer_payload(buffer: Any) -> dict[str, Array]:
    """Detach one local state/branch buffer for validated persistence."""

    payload: dict[str, Array] = {}
    for field in fields(buffer):
        array = np.asarray(getattr(buffer, field.name))
        if array.ndim < 2 or array.shape[:2] != (1, 1):
            raise ValueError("local structural branch buffers must begin with (1, 1)")
        payload[field.name] = array.copy()
    return payload


def _merge_payload(
    target: Any,
    *,
    state_slot: int,
    branch_slot: int,
    payload: Mapping[str, Any],
) -> None:
    """Merge one validated local payload into its full atlas coordinates."""

    expected = {field.name for field in fields(target)}
    if set(payload) != expected:
        raise RuntimeError("structural branch payload fields do not reproduce")
    for field in fields(target):
        local = np.asarray(payload[field.name])
        if local.ndim < 2 or local.shape[:2] != (1, 1):
            raise RuntimeError("structural branch payload shape does not reproduce")
        getattr(target, field.name)[state_slot, branch_slot] = local[0, 0]


def _restore_payloads(
    target: Any,
    identity: StructuralExecutionIdentity,
    restored: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> None:
    for (state_slot, _branch_kind, branch_slot), payload in restored.items():
        _merge_payload(
            target,
            state_slot=state_slot,
            branch_slot=branch_slot,
            payload=payload,
        )


def _versions() -> dict[str, str]:
    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    return {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }


def _shaft_properties(
    authority: ArticulatedAtlasAuthority,
    config: shaft.ArticulatedShaftAtlasConfig,
    state: tuple[int, int],
) -> shaft.ArticulatedShaftProperties:
    resolved = authority.resolve_state(*state)
    return shaft.build_articulated_shaft(
        resolved.model,
        shaft.ArticulatedShaftConfig(
            damping_ratio=config.shaft_damping_ratio,
            bending_frequency_scale=config.bending_frequency_scale,
            torsional_stiffness_scale=config.torsional_stiffness_scale,
        ),
    )


def _shaft_jobs(
    authority: ArticulatedAtlasAuthority,
    config: shaft.ArticulatedShaftAtlasConfig,
    identity: StructuralExecutionIdentity,
    pending: tuple[tuple[int, tuple[int, int], str, int], ...],
) -> tuple[tuple[ArticulatedAtlasAuthority, Any, int, tuple[int, int], int], ...]:
    jobs = []
    for state_slot, state, kind, branch_slot in pending:
        if kind != "activation" or (state != identity.registered_states[state_slot]):
            raise RuntimeError("pending shaft descriptor does not reproduce")
        jobs.append((authority, config, state_slot, state, branch_slot))
    return tuple(jobs)


def _shaft_execution_result(
    authority: ArticulatedAtlasAuthority,
    config: shaft.ArticulatedShaftAtlasConfig,
    states: tuple[tuple[int, int], ...],
    buffers: Any,
    audit: dict[str, Any],
) -> StructuralAtlasExecution:
    selection = resolve_atlas_states(
        authority, config.case_indices, config.sample_indices
    )
    properties = _shaft_properties(authority, config, states[-1])
    beam_record = json.loads(
        (DATA / "shaft_beam_reference.json").read_text(encoding="utf-8")
    )
    coarse_probes = shaft._excluded_step_probes(authority, config)
    gates = shaft._gates(buffers, config, properties, beam_record)
    arrays = shaft._arrays(authority, states, buffers, config, gates)
    record = shaft._record(
        authority,
        selection,
        buffers,
        config,
        properties,
        beam_record,
        gates,
        _versions(),
        coarse_probes,
    )
    record["checkpoint_audit"] = audit
    return StructuralAtlasExecution(record, arrays, audit)


def execute_structural_shaft_atlas(
    authority: ArticulatedAtlasAuthority,
    *,
    corner_id: str,
    checkpoint_directory: Path,
    config: shaft.ArticulatedShaftAtlasConfig = shaft.ArticulatedShaftAtlasConfig(),
    plan_path: Path = DEFAULT_OUTPUT,
) -> StructuralAtlasExecution:
    """Execute or exactly resume one structural shaft corner."""

    identity = resolve_structural_execution_identity(
        authority,
        corner_id=corner_id,
        pathway="shaft",
        configuration=config,
        plan_path=plan_path,
    )
    contracts = structural_branch_contracts(identity, config, nq=20)
    inventory = restore_structural_checkpoint_directory(
        checkpoint_directory,
        identity,
        expected_contracts=contracts,
    )
    states = identity.registered_states
    shape = (
        len(states),
        len(config.activations),
        2,
        len(config.forward.time_steps_s),
        2,
        len(config.horizons_s),
    )
    buffers = shaft._buffers(shape, 20)
    _restore_payloads(buffers, identity, inventory.restored)
    jobs = _shaft_jobs(authority, config, identity, inventory.pending)
    executor = None
    results: Iterator[
        tuple[int, tuple[int, int], int, Any, shaft.ArticulatedShaftProperties]
    ]
    if config.worker_count == 1:
        results = map(shaft._run_activation_job, jobs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(config.worker_count, len(jobs)) if jobs else 1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(shaft._run_activation_job, jobs)
    try:
        for state_slot, state, branch_slot, local, _properties in results:
            payload = _buffer_payload(local)
            write_structural_checkpoint(
                structural_checkpoint_path(
                    checkpoint_directory,
                    identity,
                    state_slot=state_slot,
                    branch_kind="activation",
                    branch_slot=branch_slot,
                ),
                identity,
                state_slot=state_slot,
                state=state,
                branch_kind="activation",
                branch_slot=branch_slot,
                arrays=payload,
                expected_contract=contracts[("activation", branch_slot)],
            )
            _merge_payload(
                buffers,
                state_slot=state_slot,
                branch_slot=branch_slot,
                payload=payload,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    audit = audit_structural_checkpoint_directory(
        checkpoint_directory,
        identity,
        expected_contracts=contracts,
    )
    return _shaft_execution_result(authority, config, states, buffers, audit)


def _ground_jobs(
    authority: ArticulatedAtlasAuthority,
    config: ground.ArticulatedGroundAtlasConfig,
    identity: StructuralExecutionIdentity,
    pending: tuple[tuple[int, tuple[int, int], str, int], ...],
) -> tuple[tuple[Any, ...], ...]:
    jobs = []
    for state_slot, state, kind, branch_slot in pending:
        if kind not in {"primary", "control"} or (
            state != identity.registered_states[state_slot]
        ):
            raise RuntimeError("pending ground descriptor does not reproduce")
        jobs.append((authority, config, state_slot, state, kind, branch_slot))
    return tuple(jobs)


def execute_structural_ground_atlas(
    authority: ArticulatedAtlasAuthority,
    *,
    corner_id: str,
    checkpoint_directory: Path,
    config: ground.ArticulatedGroundAtlasConfig = ground.ArticulatedGroundAtlasConfig(),
    plan_path: Path = DEFAULT_OUTPUT,
) -> StructuralAtlasExecution:
    """Execute or exactly resume one structural finite-ground corner."""

    identity = resolve_structural_execution_identity(
        authority,
        corner_id=corner_id,
        pathway="ground",
        configuration=config,
        plan_path=plan_path,
    )
    contracts = structural_branch_contracts(identity, config, nq=20)
    inventory = restore_structural_checkpoint_directory(
        checkpoint_directory,
        identity,
        expected_contracts=contracts,
    )
    states = identity.registered_states
    tail = (2, len(config.forward.time_steps_s), 2, len(config.horizons_s))
    primary = ground._buffers((len(states), len(config.ground_activations), *tail), 20)
    controls = ground._buffers((len(states), len(config.control_names), *tail), 20)
    for key, payload in inventory.restored.items():
        state_slot, kind, branch_slot = key
        _merge_payload(
            primary if kind == "primary" else controls,
            state_slot=state_slot,
            branch_slot=branch_slot,
            payload=payload,
        )
    jobs = _ground_jobs(authority, config, identity, inventory.pending)
    executor = None
    results: Iterator[tuple[int, tuple[int, int], Any, int, Any]]
    if config.worker_count == 1:
        results = map(ground._branch_job, jobs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(config.worker_count, len(jobs)) if jobs else 1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(ground._branch_job, jobs)
    try:
        for state_slot, state, kind, branch_slot, local in results:
            payload = _buffer_payload(local)
            write_structural_checkpoint(
                structural_checkpoint_path(
                    checkpoint_directory,
                    identity,
                    state_slot=state_slot,
                    branch_kind=kind,
                    branch_slot=branch_slot,
                ),
                identity,
                state_slot=state_slot,
                state=state,
                branch_kind=kind,
                branch_slot=branch_slot,
                arrays=payload,
                expected_contract=contracts[(kind, branch_slot)],
            )
            _merge_payload(
                primary if kind == "primary" else controls,
                state_slot=state_slot,
                branch_slot=branch_slot,
                payload=payload,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    audit = audit_structural_checkpoint_directory(
        checkpoint_directory,
        identity,
        expected_contracts=contracts,
    )
    selection = resolve_atlas_states(
        authority, config.case_indices, config.sample_indices
    )
    gates = ground._gates(primary, controls, config)
    arrays = ground._arrays(authority, states, primary, controls, config, gates)
    record = ground._record(
        authority,
        selection,
        primary,
        controls,
        config,
        gates,
        _versions(),
    )
    record["checkpoint_audit"] = audit
    return StructuralAtlasExecution(record, arrays, audit)


__all__ = [
    "StructuralAtlasExecution",
    "execute_structural_ground_atlas",
    "execute_structural_shaft_atlas",
]
