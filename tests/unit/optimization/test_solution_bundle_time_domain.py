"""Temporal support of persisted replay evidence (Refs #15493)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.optimization.solution_bundle import (
    SolutionBundleError,
    load_solution_bundle,
    save_solution_bundle,
)
from tests.unit.optimization.test_solution_bundle import _bundle

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_rejects_timebase_without_an_interval(
    tmp_path: Path, count: int, operation: str
) -> None:
    bundle = _bundle()
    target = tmp_path / "bundle"
    if operation == "load":
        save_solution_bundle(bundle, target)
    # Truncate all time-indexed channels together, preserving shape consistency.
    for name in (
        "time",
        "q_ref",
        "qd_ref",
        "tau_ff_ref",
        "q_rollout",
        "qd_rollout",
        "tau_cmd",
        "tau_applied",
    ):
        bundle.arrays[name] = bundle.arrays[name][:count]
    if operation == "load":
        np.savez(target / "arrays.npz", **bundle.arrays)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["arrays_sha256"] = hashlib.sha256(
            (target / "arrays.npz").read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SolutionBundleError, match="time.*at least two"):
        if operation == "save":
            save_solution_bundle(bundle, target)
        else:
            load_solution_bundle(target)
    if operation == "save":
        assert not target.exists()


@pytest.mark.parametrize("impact_time", [-0.01, 0.26])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_rejects_impact_outside_recorded_time_support(
    tmp_path: Path, impact_time: float, operation: str
) -> None:
    bundle = _bundle()
    target = tmp_path / "bundle"
    event = {**bundle.impact_event, "t": impact_time}
    if operation == "load":
        save_solution_bundle(bundle, target)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["impact_event"] = event
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SolutionBundleError, match="impact_event.t.*time"):
        if operation == "save":
            save_solution_bundle(replace(bundle, impact_event=event), target)
        else:
            load_solution_bundle(target)
    if operation == "save":
        assert not target.exists()


@pytest.mark.parametrize("impact_time", [4.0, 4.2, 4.25])
def test_accepts_endpoints_and_nonzero_time_origin(
    tmp_path: Path, impact_time: float
) -> None:
    bundle = _bundle()
    bundle.arrays["time"] += 4.0
    event = {**bundle.impact_event, "t": impact_time}
    target = save_solution_bundle(
        replace(bundle, impact_event=event), tmp_path / "bundle"
    )
    loaded = load_solution_bundle(target)
    assert loaded.impact_event["t"] == impact_time
    np.testing.assert_array_equal(loaded.arrays["time"], bundle.arrays["time"])
