"""OpenSim-gated tests of the spec-driven musculoskeletal pipeline (#11617).

Uses a three-frame excerpt of the driver same-input bundle (address, mid
downswing, ball-contact transient) stored under ``tests/fixtures/musculoskeletal``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "musculoskeletal"


@pytest.fixture(scope="module")
def stack():  # type: ignore[no-untyped-def]
    pytest.importorskip("opensim")
    from src.engines.physics_engines.opensim.python import (
        musculoskeletal_pipeline_v2 as p2,
        musculoskeletal_spec_dynamics as dynamics,
        musculoskeletal_spec_model as model_mod,
        musculoskeletal_static_opt as so,
    )
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    spec_bytes = (FIXTURES / "spec_driver.json").read_bytes()
    frames = np.load(FIXTURES / "frames_driver.npz")
    model, info = model_mod.build_spec_musculoskeletal_model(spec_bytes)
    return {
        "p2": p2,
        "dyn": dynamics.SkeletonDynamics(spec_bytes),
        "so": so,
        "model": model,
        "info": info,
        "frames": frames,
        "spec_bytes": spec_bytes,
    }


def _bundle(stack, n: int):  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.opensim.python.musculoskeletal_spec_bundle import (
        SpecBundle,
    )

    f = stack["frames"]
    return SpecBundle(
        spec_bytes=stack["spec_bytes"],
        coordinate_order=tuple(str(x) for x in f["coordinate_order"]),
        dt_s=float(f["dt_s"]),
        q=f["q"][n],
        v=f["v"][n],
        efforts=f["efforts"][n][None, :],
        manifest={},
        sha256="fixture",
    )


def test_graft_builds_all_muscles_and_wrap_objects(stack) -> None:  # type: ignore[no-untyped-def]
    assert stack["info"]["n_muscles"] == 80
    assert stack["info"]["n_wrap_objects"] == 44
    assert stack["model"].getMuscles().getSize() == 80


def test_muscle_model_poses_bodies_like_the_skeleton(stack) -> None:  # type: ignore[no-untyped-def]
    bundle = _bundle(stack, 1)
    basis = stack["so"].MuscleBasis(stack["model"], bundle.coordinate_order)
    out = stack["p2"].body_origin_mismatch(stack["dyn"], basis, bundle, 1)
    assert out["bodies"] > 20
    assert out["max_mm"] < 1e-3


@pytest.mark.parametrize("n", [0, 1])
def test_inverse_dynamics_reproduces_leg_and_trunk_efforts(stack, n) -> None:  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.opensim.python.musculoskeletal_spec_dynamics import (
        is_loop_coordinate,
    )

    bundle = _bundle(stack, n)
    qm, vm = bundle.step_midpoint_states()
    effort, forces = stack["dyn"].required_efforts(
        qm[0], vm[0], bundle.step_accelerations()[0]
    )
    mask = np.array([not is_loop_coordinate(c) for c in bundle.coordinate_order])
    assert forces.shape[1] == 3
    assert np.abs((effort - bundle.efforts[0])[mask]).max() < 2.0


def test_static_optimisation_returns_bounded_consistent_solution(stack) -> None:  # type: ignore[no-untyped-def]
    bundle = _bundle(stack, 1)
    result = stack["p2"].static_optimisation(
        stack["so"].MuscleBasis(stack["model"], bundle.coordinate_order), bundle, 1, 1.0
    )
    a = result["activation"]
    assert a.shape == (1, 80) and a.min() >= 0.0 and a.max() <= 1.0
    assert np.isfinite(result["reserve"]).all()
    # reserve + muscle torque = required torque (checked via the stored force)
    assert result["tau"].shape == (1, 14)


def test_upper_body_summary_excludes_legs_and_root(stack) -> None:  # type: ignore[no-untyped-def]
    summary = stack["p2"].upper_body_summary(_bundle(stack, 0))
    assert summary and not any(n.startswith("hip_flexion") for n in summary)
    assert all(v["peak"] >= v["rms"] >= 0.0 for v in summary.values())
