"""Checkpointed execution contracts for structural headline propagation."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from scripts.research.proximal_distal_energy import articulated_ground_atlas as ground
from scripts.research.proximal_distal_energy import articulated_shaft_atlas as shaft
from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_structural_atlas_execution import (
    _buffer_payload,
    _merge_payload,
    _shaft_properties,
    execute_structural_ground_atlas,
    execute_structural_shaft_atlas,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _authority() -> ArticulatedAtlasAuthority:
    return ArticulatedAtlasAuthority.from_scaled(
        load_scaled_authority(
            DATA / "articulated_structural_authority_nominal.json",
            DATA / "articulated_structural_authority_nominal.npz",
        )
    )


def _fill(buffer, value: float) -> None:
    for field in fields(buffer):
        array = getattr(buffer, field.name)
        if array.dtype == np.bool_:
            array.fill(True)
        elif np.issubdtype(array.dtype, np.integer):
            array.fill(int(value))
        else:
            array.fill(value)


def test_branch_payload_is_detached_and_merges_one_exact_slot() -> None:
    local = shaft._buffers((1, 1, 2, 2, 2, 4), 20)
    target = shaft._buffers((3, 4, 2, 2, 2, 4), 20)
    _fill(local, 7.0)
    _fill(target, -1.0)

    payload = _buffer_payload(local)
    _merge_payload(target, state_slot=2, branch_slot=3, payload=payload)
    local.peak_force.fill(99.0)

    assert np.all(target.peak_force[2, 3] == 7.0)
    assert np.all(target.active_set_parity[2, 3])
    assert np.all(target.peak_force[0, 0] == -1.0)


def _native_modules(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(
        sys.modules,
        "pinocchio",
        SimpleNamespace(__version__="test"),
    )


def test_shaft_execution_resumes_without_recomputing_valid_checkpoints(
    tmp_path, monkeypatch
) -> None:
    authority = _authority()
    config = shaft.ArticulatedShaftAtlasConfig(worker_count=1)
    calls: list[tuple[int, int]] = []
    _native_modules(monkeypatch)

    def fake_job(payload):
        governed, configuration, state_slot, state, branch_slot = payload
        calls.append((state_slot, branch_slot))
        local = shaft._buffers((1, 1, 2, 2, 2, 4), 20)
        _fill(local, float(state_slot + branch_slot + 1))
        return (
            state_slot,
            state,
            branch_slot,
            local,
            _shaft_properties(governed, configuration, state),
        )

    monkeypatch.setattr(shaft, "_run_activation_job", fake_job)
    monkeypatch.setattr(shaft, "_excluded_step_probes", lambda *_: [])
    monkeypatch.setattr(shaft, "_gates", lambda *_: {})
    monkeypatch.setattr(shaft, "_arrays", lambda *args: {"complete": np.ones(1)})
    monkeypatch.setattr(shaft, "_record", lambda *args: {"pathway": "shaft"})

    first = execute_structural_shaft_atlas(
        authority,
        corner_id="nominal",
        checkpoint_directory=tmp_path / "shaft",
        config=config,
    )
    assert len(calls) == 48
    assert first.checkpoint_audit["status"] == "complete"
    assert first.checkpoint_audit["checkpoint_count"] == 48
    assert first.checkpoint_audit["release_evidence"] is False

    calls.clear()
    second = execute_structural_shaft_atlas(
        authority,
        corner_id="nominal",
        checkpoint_directory=tmp_path / "shaft",
        config=config,
    )
    assert calls == []
    assert second.checkpoint_audit == first.checkpoint_audit
    assert second.record["checkpoint_audit"] == first.checkpoint_audit


def test_ground_execution_resumes_without_recomputing_valid_checkpoints(
    tmp_path, monkeypatch
) -> None:
    authority = _authority()
    config = ground.ArticulatedGroundAtlasConfig(worker_count=1)
    calls: list[tuple[int, str, int]] = []
    _native_modules(monkeypatch)

    def fake_job(payload):
        _governed, _configuration, state_slot, state, kind, branch_slot = payload
        calls.append((state_slot, kind, branch_slot))
        local = ground._buffers((1, 1, 2, 2, 2, 4), 20)
        _fill(local, float(state_slot + branch_slot + 1))
        return state_slot, state, kind, branch_slot, local

    monkeypatch.setattr(ground, "_branch_job", fake_job)
    monkeypatch.setattr(ground, "_gates", lambda *_: {})
    monkeypatch.setattr(ground, "_arrays", lambda *args: {"complete": np.ones(1)})
    monkeypatch.setattr(ground, "_record", lambda *args: {"pathway": "ground"})

    first = execute_structural_ground_atlas(
        authority,
        corner_id="nominal",
        checkpoint_directory=tmp_path / "ground",
        config=config,
    )
    assert len(calls) == 72
    assert first.checkpoint_audit["status"] == "complete"
    assert first.checkpoint_audit["checkpoint_count"] == 72

    calls.clear()
    second = execute_structural_ground_atlas(
        authority,
        corner_id="nominal",
        checkpoint_directory=tmp_path / "ground",
        config=config,
    )
    assert calls == []
    assert second.checkpoint_audit == first.checkpoint_audit
