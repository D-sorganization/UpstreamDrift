"""Unit tests for the same-input bundle reader (issue #11617, phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_bundle import (
    SpecBundle,
    load_spec_bundle,
)

pytestmark = pytest.mark.unit

ORDER = ["TranslationInputX", "hip_flexion_r", "knee_angle_r"]


def _write(path: Path, *, steps: int = 5, schema: str = "same-input-bundle/v1") -> None:
    spec = {"coordinate_order": ORDER, "schema_version": "full-body-v1"}
    manifest = {"schema": schema, "coordinate_order": ORDER, "dt_s": 0.001}
    manifest["provenance"] = {"capture": "unit"}
    rng = np.random.default_rng(0)
    np.savez(
        path,
        manifest=np.array(json.dumps(manifest)),
        spec=np.frombuffer(json.dumps(spec).encode(), dtype=np.uint8),
        q0=np.zeros(3),
        v0=np.zeros(3),
        efforts=rng.normal(size=(steps, 3)),
        reference_q=rng.normal(size=(steps + 1, 3)),
        reference_v=rng.normal(size=(steps + 1, 3)),
    )


def test_load_round_trip_and_derived_quantities(tmp_path: Path) -> None:
    file = tmp_path / "b.npz"
    _write(file)
    bundle = load_spec_bundle(file)
    assert isinstance(bundle, SpecBundle)
    assert bundle.steps == 5 and bundle.nv == 3 and bundle.capture == "unit"
    assert len(bundle.sha256) == 64
    assert bundle.times().shape == (6,) and bundle.times()[-1] == pytest.approx(0.005)
    acc = bundle.step_accelerations()
    assert acc.shape == (5, 3)
    np.testing.assert_allclose(acc[2], (bundle.v[3] - bundle.v[2]) / 0.001)
    q_mid, v_mid = bundle.step_midpoint_states()
    np.testing.assert_allclose(q_mid[0], 0.5 * (bundle.q[0] + bundle.q[1]))
    assert v_mid.shape == (5, 3)
    assert bundle.index("knee_angle_r") == 2
    with pytest.raises(KeyError):
        bundle.index("nope")


def test_rejects_wrong_schema_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    _write(bad, schema="other/v9")
    with pytest.raises(ValueError, match="schema"):
        load_spec_bundle(bad)
    with pytest.raises(FileNotFoundError):
        load_spec_bundle(tmp_path / "absent.npz")


def test_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        SpecBundle(
            spec_bytes=b"{}",
            coordinate_order=("a", "b"),
            dt_s=0.001,
            q=np.zeros((4, 2)),
            v=np.zeros((4, 2)),
            efforts=np.zeros((4, 2)),  # must be 3 rows
            manifest={},
            sha256="0" * 64,
        )
