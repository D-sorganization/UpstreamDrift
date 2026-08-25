"""Execution-topology and checkpoint contracts for the full ground atlas."""

from __future__ import annotations

from dataclasses import replace
from importlib.machinery import PathFinder

import numpy as np
import pytest

from scripts.research.proximal_distal_energy import articulated_ground_atlas as atlas

pytestmark = pytest.mark.scientific


def _authority() -> atlas._Authority:
    return atlas._Authority(
        time_s=np.linspace(0.0, 0.24, 13),
        profile_index=np.repeat(np.arange(6), 3),
        grip_span_m=np.tile(np.asarray((0.12, 0.18, 0.24)), 6),
        solution_q=np.zeros((18, 13, 20)),
    )


def test_ground_atlas_accepts_twenty_branch_workers() -> None:
    assert atlas.ArticulatedGroundAtlasConfig(worker_count=20).worker_count == 20
    with pytest.raises(ValueError, match="worker_count"):
        atlas.ArticulatedGroundAtlasConfig(worker_count=21)


def test_branch_checkpoint_round_trip_and_design_drift_rejection(tmp_path) -> None:
    config = atlas.ArticulatedGroundAtlasConfig(worker_count=20)
    authority = _authority()
    shape = (1, 1, 2, len(config.forward.time_steps_s), 2, len(config.horizons_s))
    source = atlas._buffers(shape, 20)
    for index, field in enumerate(atlas.fields(atlas._Buffers), 1):
        getattr(source, field.name).fill(index)
    digest = atlas._execution_digest(authority, config)
    path = tmp_path / "branch.npz"

    atlas._save_branch_checkpoint(
        path,
        digest=digest,
        state_slot=3,
        state=(8, 0),
        kind="primary",
        branch_slot=2,
        buffer=source,
    )
    loaded = atlas._load_branch_checkpoint(
        path,
        digest=digest,
        state_slot=3,
        state=(8, 0),
        kind="primary",
        branch_slot=2,
    )
    for field in atlas.fields(atlas._Buffers):
        assert np.array_equal(getattr(loaded, field.name), getattr(source, field.name))

    changed = atlas._execution_digest(
        authority, replace(config, ground_translation_stiffness_scale=1.1)
    )
    with pytest.raises(RuntimeError, match="design digest"):
        atlas._load_branch_checkpoint(
            path,
            digest=changed,
            state_slot=3,
            state=(8, 0),
            kind="primary",
            branch_slot=2,
        )


def test_short_real_atlas_restarts_from_identical_branch_checkpoints(tmp_path) -> None:
    if PathFinder.find_spec("pinocchio") is None:
        pytest.skip("robotics Pinocchio is required for the native restart test")
    config = atlas.ArticulatedGroundAtlasConfig(
        forward=atlas.GroundForwardConfig(
            duration_s=0.001,
            time_steps_s=(0.001, 0.0005),
        ),
        case_indices=(0,),
        sample_indices=(0,),
        horizons_s=(0.001,),
        worker_count=1,
    )

    first_record, first_arrays = atlas.run_articulated_ground_atlas(
        config, state_checkpoint_dir=tmp_path
    )
    second_record, second_arrays = atlas.run_articulated_ground_atlas(
        config, state_checkpoint_dir=tmp_path
    )

    assert len(list(tmp_path.glob("*.npz"))) == 6
    assert second_record == first_record
    assert second_arrays.keys() == first_arrays.keys()
    for name in first_arrays:
        equal_nan = np.issubdtype(first_arrays[name].dtype, np.inexact)
        assert np.array_equal(
            second_arrays[name], first_arrays[name], equal_nan=equal_nan
        ), name
