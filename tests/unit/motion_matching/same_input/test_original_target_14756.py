"""Original tracking evidence stays distinct from deterministic rollout states."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input.bundle import InputBundle

pytestmark = pytest.mark.unit


def _bundle(**kwargs: object) -> InputBundle:
    return InputBundle(
        spec_bytes=b'{"coordinate_order": ["joint"]}',
        coordinate_order=("joint",),
        dt_s=0.001,
        q0=np.zeros(1),
        v0=np.zeros(1),
        efforts=np.zeros((2, 1)),
        reference_q=np.zeros((3, 1)),
        reference_v=np.zeros((3, 1)),
        reference_engine="mujoco",
        provenance={"run_id": "synthetic-run"},
        **kwargs,
    )


def _target_bundle() -> InputBundle:
    return _bundle(
        target_time_s=np.array([12.0, 12.001, 12.002]),
        target_q=np.ones((3, 1)),
    )


def _rewrite(path: Path, mutate: Callable[[dict], None]) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def test_original_target_round_trip_is_not_the_rollout(tmp_path: Path) -> None:
    bundle = _target_bundle()
    path = tmp_path / "target.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    np.testing.assert_array_equal(loaded.target_time_s, [12.0, 12.001, 12.002])
    np.testing.assert_array_equal(loaded.target_q, np.ones((3, 1)))
    np.testing.assert_array_equal(loaded.reference_q, np.zeros((3, 1)))
    assert loaded.target_sha256 == bundle.target_sha256
    assert loaded.manifest()["target_sha256"] == loaded.target_sha256
    assert loaded.provenance == bundle.provenance


@pytest.mark.parametrize("field", ["target_q", "target_time_s"])
@pytest.mark.parametrize("missing", [False, True])
def test_target_identity_rejects_changed_or_missing_arrays(
    tmp_path: Path, field: str, missing: bool
) -> None:
    path = tmp_path / "target.npz"
    _target_bundle().save(path)

    def mutate(arrays: dict) -> None:
        if missing:
            arrays.pop(field)
        else:
            arrays[field] = arrays[field] + 0.25

    _rewrite(path, mutate)
    with pytest.raises(ValueError, match="target"):
        InputBundle.load(path)


def test_target_arrays_require_manifest_identity(tmp_path: Path) -> None:
    path = tmp_path / "target.npz"
    _target_bundle().save(path)

    def mutate(arrays: dict) -> None:
        manifest = json.loads(str(arrays["manifest"]))
        manifest.pop("target_sha256")
        arrays["manifest"] = np.array(json.dumps(manifest))

    _rewrite(path, mutate)
    with pytest.raises(ValueError, match="target"):
        InputBundle.load(path)


def test_legacy_bundle_has_no_inferred_target(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    _bundle().save(path)
    loaded = InputBundle.load(path)
    assert loaded.target_q is None
    assert loaded.target_time_s is None
    assert loaded.target_sha256 is None


@pytest.mark.parametrize(
    "times,q",
    [
        (None, np.ones((3, 1))),
        (np.arange(3.0), None),
        (np.array([]), np.empty((0, 1))),
        (np.array([1.0, 1.0]), np.ones((2, 1))),
        (np.array([1.0, np.nan]), np.ones((2, 1))),
        (np.arange(3.0), np.ones((3, 2))),
        (np.arange(3.0), np.full((3, 1), np.inf)),
    ],
)
def test_invalid_target_evidence_is_rejected(times: object, q: object) -> None:
    with pytest.raises(ValueError, match="target"):
        _bundle(target_time_s=times, target_q=q)


def test_target_identity_binds_clock_coordinates_and_spec() -> None:
    bundle = _target_bundle()
    identities = {
        bundle.target_sha256,
        replace(bundle, target_time_s=bundle.target_time_s + 1).target_sha256,
        replace(bundle, target_q=bundle.target_q + 1).target_sha256,
        replace(
            bundle, spec_bytes=b'{"coordinate_order": ["joint"], "v": 2}'
        ).target_sha256,
    }
    assert len(identities) == 4


@pytest.mark.parametrize("lead_in_steps", [None, 1])
def test_generation_preserves_original_target(
    monkeypatch: pytest.MonkeyPatch, lead_in_steps: int | None
) -> None:
    from src.shared.python.motion_matching.same_input import reference
    from src.shared.python.motion_matching.same_input.integrator import Rollout

    rollout = Rollout(
        time_s=np.arange(4) * 0.001,
        q=np.zeros((4, 1)),
        v=np.zeros((4, 1)),
        efforts=np.zeros((3, 1)),
        pose_drift=np.zeros(4),
    )
    monkeypatch.setattr(reference, "closed_loop", lambda *a, **kw: rollout)
    times = np.array([12.0, 12.001, 12.002, 12.003])
    target = np.ones((4, 1))
    kwargs = {"duration_s": 0.002}
    generate = reference.generate_reference_bundle
    if lead_in_steps is not None:
        generate = reference.generate_lead_in_bundle
        kwargs["lead_in_steps"] = lead_in_steps
    bundle = generate(_bundle().spec_bytes, times, target, **kwargs)
    times[:] = 0
    target[:] = 0
    np.testing.assert_array_equal(bundle.target_q, np.ones((4, 1)))
    np.testing.assert_array_equal(bundle.target_time_s, [12, 12.001, 12.002, 12.003])
    assert not bundle.reference_q.any()
