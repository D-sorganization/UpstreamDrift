"""Bench press same-input bundle on the bench contact body (LIFT-5, #11745).

The lifter lies supine: four back spheres (OSL-2 ``msk_lift_model`` geometry)
press on a raised ``bench_top`` support at the hold's lowest back-sphere point
and the feet stand on the floor (``contact_law.support_planes``).  The bar
path is the Movement Optimizer's bench arm-chain bar model on the model's
shoulders; the 0.2 s MuJoCo window is centred on the chest touch
(``scripts/generate_lift_bundle.py --lift bench_press``).  The IK thresholds
are the residuals measured when the committed bundle was generated, rounded
up to 0.1 mm; they are recorded evidence, not tuning targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bench_press import (
    BACK_SPHERE_RADIUS_M,
    BACK_SPHERES,
    BENCH_SUPPORT,
    GENERIC_STATURE_M,
    back_sphere_documents,
    bench_bar_targets,
    bench_spec,
)
from src.shared.python.lifting.bundles import LIFT_SPEC, load_seed_pose
from src.shared.python.lifting.lift_recipes import recipe, window_centre_time
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalFrame,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

MAX_BUNDLE_BYTES = 300_000
# Measured maxima over all 151 frames (metres); see the module docstring.
IK_RESIDUAL_THRESHOLDS_M: dict[str, float] = {
    "ankle_l": 0.0001,
    "ankle_r": 0.0001,
    "knee_l": 0.0001,
    "knee_r": 0.0001,
    "hip_l": 0.0001,
    "hip_r": 0.0001,
    "shoulder_l": 0.0001,
    "shoulder_r": 0.0001,
    "bar": 0.0148,
}
CLOSURE_THRESHOLD_M = 2.9e-5
FOOT_PENETRATION_THRESHOLD_M = 7.0e-4
# Measured on the supine hold fit (metres): the largest back-sphere gap above
# the bench top, and the hold's bench height (IPF 42-45 cm is not met).
SPHERE_CLEARANCE_THRESHOLD_M = 1.1e-3


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(recipe("bench_press").bundle_path)


def _document(spec_bytes: bytes) -> dict:
    return json.loads(spec_bytes)


# -- the bench contact body ---------------------------------------------------


def test_back_spheres_mirror_the_osl2_geometry() -> None:
    from src.engines.physics_engines.opensim.python import msk_lift_model as osl2

    assert GENERIC_STATURE_M == osl2.GENERIC_STATURE_M
    assert BENCH_SUPPORT == osl2.BENCH_TOP
    assert [m for m, _body, _r in osl2._BACK_SPHERES] == [
        name.removeprefix("back_") for name, _marker in BACK_SPHERES
    ]
    assert {r for _m, _b, r in osl2._BACK_SPHERES} == {BACK_SPHERE_RADIUS_M}


def test_back_spheres_scale_with_stature_and_sit_anterior_of_the_skin() -> None:
    document = _document(LIFT_SPEC.read_bytes())
    spheres = back_sphere_documents(document)
    stature = document["subject"]["stature_m"]
    assert [s["name"] for s in spheres] == [n for n, _m in BACK_SPHERES]
    for sphere in spheres:
        assert sphere["radius_m"] == pytest.approx(
            BACK_SPHERE_RADIUS_M * stature / GENERIC_STATURE_M
        )
    no_stature = {**document, "subject": {}}
    with pytest.raises(ValueError, match="stature"):
        back_sphere_documents(no_stature)


def test_bench_spec_adds_the_back_spheres_on_the_bench_top_only() -> None:
    base = LIFT_SPEC.read_bytes()
    derived = _document(bench_spec(base, 0.45))
    original = _document(base)
    feet = original["contact"]["spheres"]
    assert derived["contact"]["spheres"][: len(feet)] == feet
    (support,) = derived["contact"]["supports"]
    assert support["name"] == BENCH_SUPPORT
    assert support["height_m"] == 0.45
    assert support["spheres"] == [n for n, _m in BACK_SPHERES]
    assert {k: v for k, v in derived.items() if k != "contact"} == {
        k: v for k, v in original.items() if k != "contact"
    }
    with pytest.raises(ValueError, match="already"):
        bench_spec(bench_spec(base, 0.45), 0.45)
    with pytest.raises(ValueError, match="finite"):
        bench_spec(base, float("nan"))
    with pytest.raises(TypeError):
        bench_spec("spec", 0.45)  # type: ignore[arg-type]


def test_bench_bar_targets_scale_the_optimizer_offset_in_the_press_plane() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 3),
        np.zeros((3, 3)),
        np.array([[0.0, 0.7], [0.08, 0.02], [0.0, 0.7]]),
        {},
    )
    frame = SagittalFrame(forward=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0))
    mid = np.array([0.1, 0.4, 0.6])
    targets = bench_bar_targets(solution, mid, frame, 2.0)
    np.testing.assert_allclose(targets[1], mid + [0.0, -0.16, 0.04])
    with pytest.raises(ValueError, match="scale"):
        bench_bar_targets(solution, mid, frame, 0.0)


# -- the committed bundle -----------------------------------------------------


def test_committed_bundle_is_small(bundle: InputBundle) -> None:
    assert recipe("bench_press").bundle_path.stat().st_size <= MAX_BUNDLE_BYTES
    assert bundle.reference_engine == "mujoco"
    assert bundle.steps == 200


def test_bundle_round_trips(bundle: InputBundle, tmp_path: Path) -> None:
    path = tmp_path / "again.npz"
    bundle.save(path)
    again = InputBundle.load(path)
    assert again.manifest() == bundle.manifest()
    assert again.spec_bytes == bundle.spec_bytes
    for name in ("q0", "v0", "efforts", "reference_q", "reference_v"):
        np.testing.assert_array_equal(getattr(again, name), getattr(bundle, name))


def test_bundle_carries_provenance_and_the_bench(bundle: InputBundle) -> None:
    from src.shared.python.lifting.back_bar import apply_grip_width
    from src.shared.python.lifting.lift_spec import lift_grip

    bench = recipe("bench_press")
    prov = bundle.provenance
    assert prov["lift"] == bench.bundle_name
    assert len(prov["source_commit"]) == 40
    assert len(prov["tools_pin"]) == 40
    optimizer = prov["optimizer"]
    assert optimizer["exercise"] == optimizer["bar_convention"] == "bench_press"
    assert optimizer["duration_s"] == bench.duration_s
    assert optimizer["success"] is True
    assert prov["recipe"]["phase"] == "chest_touch"
    assert prov["bar_frame"] == "Grip"
    width = lift_grip("bench_press").width_m
    assert prov["ik"]["hand_bar"]["grip_width_m"] == width
    record = prov["bench"]
    grip_spec = apply_grip_width(LIFT_SPEC.read_bytes(), width)
    assert bundle.spec_bytes == bench_spec(grip_spec, record["height_m"])
    assert max(record["sphere_clearance_m"].values()) <= (SPHERE_CLEARANCE_THRESHOLD_M)
    assert min(record["sphere_clearance_m"].values()) == 0.0
    assert "bench contact body" in prov["limitations"]
    assert "Hand-held bar" in prov["limitations"]


def test_window_is_centred_on_the_chest_touch(bundle: InputBundle) -> None:
    prov = bundle.provenance
    solution = SagittalSolution.from_record(prov["sagittal_solution"])
    centre = window_centre_time(solution, "bottom")
    start = prov["window"]["start_s"]
    assert start <= centre <= start + prov["window"]["duration_s"]
    step = float(np.diff(solution.time_s).max())
    assert abs(start + prov["window"]["duration_s"] / 2 - centre) <= step


def test_recorded_ik_residuals_are_below_thresholds(bundle: InputBundle) -> None:
    ik = bundle.provenance["ik"]
    assert set(IK_RESIDUAL_THRESHOLDS_M) == set(LANDMARK_LABELS)
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert ik["max_residual_m"][label] <= limit, label
    assert ik["max_closure_m"] <= CLOSURE_THRESHOLD_M
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_back_spheres_carry_the_trunk_through_the_window(
    bundle: InputBundle,
) -> None:
    """The bench holds the supine trunk: every back sphere is loaded mid-window
    and the floating root stays within millimetres of its start."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = VectorPlant("mujoco", bundle.spec_bytes)
    reference = bundle.reference_q
    for k in (reference.shape[0] // 2, reference.shape[0] - 1):
        samples = plant.contact_samples(reference[k], np.zeros(reference.shape[1]))
        for name, _marker in BACK_SPHERES:
            assert float(samples[name].normal_force_n[2]) > 0.0, (k, name)
    drift = np.abs(reference[:, :3] - reference[0, :3]).max()
    assert drift <= 2e-3


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_from_the_recorded_solution_meets_thresholds(
    bundle: InputBundle,
) -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.bench_press import lift_bench_to_full_body

    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    bench = lift_bench_to_full_body(LIFT_SPEC.read_bytes(), solution, load_seed_pose())
    full = bench.lift
    assert full.back_bar is not None
    assert full.back_bar.spec_bytes == bundle.spec_bytes
    residual = full.max_residual_by_landmark()
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert residual[label] <= limit, (label, residual[label])
    assert float(full.closure_m.max()) <= CLOSURE_THRESHOLD_M
    assert float(full.lowest_sphere_m.min()) >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_mujoco_replay_reproduces_the_bar_path(bundle: InputBundle) -> None:
    """Open-loop MuJoCo replay of the first 10 steps is bit exact (smoke)."""
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.bundles import bar_path
    from src.shared.python.motion_matching.same_input import VectorPlant, open_loop

    steps = 10
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    rollout = open_loop(
        plant, bundle.q0, bundle.v0, bundle.efforts[:steps], dt_s=bundle.dt_s
    )
    np.testing.assert_array_equal(rollout.q, bundle.reference_q[: steps + 1])
    recorded = np.asarray(bundle.provenance["window_bar_path_m"], dtype=float)
    np.testing.assert_allclose(
        bar_path(plant, rollout.q, frame=bundle.provenance["bar_frame"]),
        recorded[: steps + 1],
        rtol=0,
        atol=1e-9,
    )
