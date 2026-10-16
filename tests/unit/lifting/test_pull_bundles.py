"""Deadlift and clean-and-jerk same-input bundles (LIFT-5, #11745).

Both are pulls built by the back-squat pipeline (``scripts/generate_lift_bundle
.py --lift ...``): a Movement Optimizer solve, full-body marker IK with the bar
held in the hands at the standard's grip width, and a 0.2 s MuJoCo tracking
window centred on the knee pass.  The IK thresholds are the residuals measured
when the committed bundles were generated, rounded up to 0.1 mm; they are
recorded evidence, not tuning targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import (
    LIFT_SPEC,
    lift_hands_to_full_body,
    load_seed_pose,
)
from src.shared.python.lifting.lift_recipes import knee_pass_time, recipe
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

PULLS = ("deadlift", "clean_and_jerk")
MAX_BUNDLE_BYTES = 300_000
# Measured maxima over all 151 frames (metres); see the module docstring.
IK_RESIDUAL_THRESHOLDS_M: dict[str, dict[str, float]] = {
    "deadlift": {
        "ankle_l": 0.0080,
        "ankle_r": 0.0040,
        "knee_l": 0.0227,
        "knee_r": 0.0100,
        "hip_l": 0.0149,
        "hip_r": 0.0144,
        "shoulder_l": 0.0096,
        "shoulder_r": 0.0078,
        "bar": 0.0040,
    },
    "clean_and_jerk": {
        "ankle_l": 0.0078,
        "ankle_r": 0.0037,
        "knee_l": 0.0227,
        "knee_r": 0.0101,
        "hip_l": 0.0094,
        "hip_r": 0.0139,
        "shoulder_l": 0.0087,
        "shoulder_r": 0.0078,
        "bar": 0.0045,
    },
}
CLOSURE_THRESHOLD_M: dict[str, float] = {"deadlift": 1.7e-7, "clean_and_jerk": 6.5e-7}
FOOT_PENETRATION_THRESHOLD_M: dict[str, float] = {
    "deadlift": 7.0e-4,
    "clean_and_jerk": 7.0e-4,
}


@pytest.fixture(scope="module", params=PULLS)
def lift(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def bundle(lift: str) -> InputBundle:
    return InputBundle.load(recipe(lift).bundle_path)


def test_committed_bundle_is_small(lift: str, bundle: InputBundle) -> None:
    assert recipe(lift).bundle_path.stat().st_size <= MAX_BUNDLE_BYTES
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


def test_bundle_carries_provenance(lift: str, bundle: InputBundle) -> None:
    from src.shared.python.lifting.back_bar import apply_grip_width
    from src.shared.python.lifting.lift_spec import lift_grip

    pull = recipe(lift)
    prov = bundle.provenance
    assert prov["lift"] == pull.bundle_name
    assert len(prov["source_commit"]) == 40
    assert len(prov["tools_pin"]) == 40
    optimizer = prov["optimizer"]
    assert optimizer["package"] == "movement_optimizer"
    assert optimizer["exercise"] == pull.exercise
    assert optimizer["bar_convention"] == pull.bar_convention
    assert optimizer["duration_s"] == pull.duration_s
    assert optimizer["success"] is True
    assert prov["recipe"]["phase"] == pull.phase
    assert prov["bar_frame"] == "Grip"
    hand_bar = prov["ik"]["hand_bar"]
    assert hand_bar["lift"] == pull.lift
    width = lift_grip(pull.lift).width_m
    assert hand_bar["grip_width_m"] == width
    assert bundle.spec_bytes == apply_grip_width(LIFT_SPEC.read_bytes(), width)
    assert prov["spec_sha256"] == bundle.spec_sha256
    assert "Hand-held bar" in prov["limitations"]


def test_window_is_centred_on_the_knee_pass(bundle: InputBundle) -> None:
    prov = bundle.provenance
    solution = SagittalSolution.from_record(prov["sagittal_solution"])
    centre = knee_pass_time(solution)
    start = prov["window"]["start_s"]
    assert start <= centre <= start + prov["window"]["duration_s"]
    step = float(np.diff(solution.time_s).max())
    assert abs(start + prov["window"]["duration_s"] / 2 - centre) <= step


def test_recorded_ik_residuals_are_below_thresholds(
    lift: str, bundle: InputBundle
) -> None:
    ik = bundle.provenance["ik"]
    assert set(ik["max_residual_m"]) == set(LANDMARK_LABELS)
    for label, limit in IK_RESIDUAL_THRESHOLDS_M[lift].items():
        assert ik["max_residual_m"][label] <= limit, label
    assert set(IK_RESIDUAL_THRESHOLDS_M[lift]) == set(LANDMARK_LABELS)
    assert ik["max_closure_m"] <= CLOSURE_THRESHOLD_M[lift]
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M[lift]


def test_window_bar_hangs_below_the_shoulders(bundle: InputBundle) -> None:
    """The recorded bar path stays under the shoulder girdle (a pull)."""
    bar = np.asarray(bundle.provenance["window_bar_path_m"], dtype=float)
    assert bar.shape == (bundle.steps + 1, 3)
    from src.shared.python.motion_matching.same_input import VectorPlant

    pytest.importorskip("mujoco")
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    hub = plant.kinematic_frames(bundle.reference_q[0])["Hub"][:3, 3]
    assert bar[0, 2] < hub[2] - 0.4


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_from_the_recorded_solution_meets_thresholds(
    lift: str, bundle: InputBundle
) -> None:
    pytest.importorskip("mujoco")
    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    full = lift_hands_to_full_body(
        LIFT_SPEC.read_bytes(), solution, load_seed_pose(), recipe(lift).lift
    )
    assert full.back_bar is not None
    assert full.back_bar.spec_bytes == bundle.spec_bytes
    residual = full.max_residual_by_landmark()
    for label, limit in IK_RESIDUAL_THRESHOLDS_M[lift].items():
        assert residual[label] <= limit, (label, residual[label])
    assert float(full.closure_m.max()) <= CLOSURE_THRESHOLD_M[lift]
    assert float(full.lowest_sphere_m.min()) >= -FOOT_PENETRATION_THRESHOLD_M[lift]


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


def test_lift_hands_rejects_an_unknown_lift() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5),
        np.zeros((5, 3)),
        np.zeros((5, 2)),
        {"segment_lengths_m": [0.44, 0.44, 0.51]},
    )
    with pytest.raises(ValueError, match="lift"):
        lift_hands_to_full_body(
            LIFT_SPEC.read_bytes(), solution, load_seed_pose(), "curl"
        )
