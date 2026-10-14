"""Back-squat same-input bundle: round trip, IK residuals, MuJoCo replay (#11745).

The IK thresholds below are the residuals measured when the committed bundle
was generated (``scripts/generate_lift_bundle.py``), rounded up to 0.1 mm;
they are recorded evidence, not tuning targets.  The dominant residuals come
from the full-body spec's own left/right segment-length and pelvis asymmetry,
which a symmetric sagittal chain cannot match exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import (
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    build_lift_bundle,
    lift_to_full_body,
    load_seed_pose,
)
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

# Measured maxima over all 151 frames (metres); see module docstring.
IK_RESIDUAL_THRESHOLDS_M = {
    "ankle_l": 0.0083,
    "ankle_r": 0.0054,
    "knee_l": 0.0270,
    "knee_r": 0.0110,
    "hip_l": 0.0149,
    "hip_r": 0.0153,
    "shoulder_l": 0.0098,
    "shoulder_r": 0.0098,
    "bar": 0.0044,
}
CLOSURE_THRESHOLD_M = 1.9e-6
FOOT_PENETRATION_THRESHOLD_M = 5.4e-4


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


def test_committed_bundle_is_small(bundle: InputBundle) -> None:
    assert BACK_SQUAT_BUNDLE.stat().st_size <= 300_000
    assert bundle.reference_engine == "mujoco"


def test_bundle_round_trips(bundle: InputBundle, tmp_path: Path) -> None:
    path = tmp_path / "again.npz"
    bundle.save(path)
    again = InputBundle.load(path)
    assert again.manifest() == bundle.manifest()
    assert again.spec_bytes == bundle.spec_bytes
    for name in ("q0", "v0", "efforts", "reference_q", "reference_v"):
        np.testing.assert_array_equal(getattr(again, name), getattr(bundle, name))


def test_bundle_carries_provenance(bundle: InputBundle) -> None:
    prov = bundle.provenance
    assert prov["lift"] == "back_squat"
    assert len(prov["source_commit"]) == 40
    optimizer = prov["optimizer"]
    assert optimizer["package"] == "movement_optimizer"
    assert optimizer["exercise"] == "full_squat"
    assert optimizer["success"] is True
    for key in ("version", "body_mass_kg", "height_m", "bar_mass_kg", "n_eval"):
        assert key in optimizer
    ik = prov["ik"]
    assert set(ik["max_residual_m"]) == set(LANDMARK_LABELS)
    assert prov["spec_sha256"] == bundle.spec_sha256
    if "back_bar" in ik:
        from src.shared.python.lifting.back_bar import apply_grip_width

        width = ik["back_bar"]["geometry"]["grip_width_m"]
        assert bundle.spec_bytes == apply_grip_width(LIFT_SPEC.read_bytes(), width)
    else:
        assert bundle.spec_bytes == LIFT_SPEC.read_bytes()


def test_recorded_ik_residuals_are_below_thresholds(bundle: InputBundle) -> None:
    ik = bundle.provenance["ik"]
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert ik["max_residual_m"][label] <= limit, label
    assert ik["max_closure_m"] <= CLOSURE_THRESHOLD_M
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_from_the_recorded_solution_meets_thresholds(
    bundle: InputBundle,
) -> None:
    pytest.importorskip("mujoco")
    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    geom_rec = bundle.provenance["ik"].get("back_bar", {}).get("geometry")
    geometry = None
    if geom_rec is not None:
        from src.shared.python.lifting.back_bar import BackBarGeometry

        geometry = BackBarGeometry.from_standard(
            geom_rec["lift"], height_m=geom_rec["height_m"]
        )
    lift = lift_to_full_body(
        bundle.spec_bytes, solution, load_seed_pose(), geometry=geometry
    )
    residual = lift.max_residual_by_landmark()
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert residual[label] <= limit, (label, residual[label])
    assert float(lift.closure_m.max()) <= CLOSURE_THRESHOLD_M
    assert float(lift.lowest_sphere_m.min()) >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_mujoco_replay_reproduces_the_bar_path(bundle: InputBundle) -> None:
    """Open-loop MuJoCo replay of the first 10 steps is bit exact (smoke).

    Ten steps keep the test inside the 60 s budget on a loaded runner; the
    full-window cross-engine bar-path parity is the next LIFT-5 slice.
    """
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.bundles import BAR_FRAME, bar_path
    from src.shared.python.motion_matching.same_input import VectorPlant, open_loop

    steps = 10
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    rollout = open_loop(
        plant, bundle.q0, bundle.v0, bundle.efforts[:steps], dt_s=bundle.dt_s
    )
    np.testing.assert_array_equal(rollout.q, bundle.reference_q[: steps + 1])
    recorded = np.asarray(bundle.provenance["window_bar_path_m"], dtype=float)
    frame = bundle.provenance.get("bar_frame", BAR_FRAME)
    np.testing.assert_allclose(
        bar_path(plant, rollout.q, frame=frame),
        recorded[: steps + 1],
        rtol=0,
        atol=1e-9,
    )


def test_build_rejects_a_window_outside_the_lift() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5), np.zeros((5, 3)), np.zeros((5, 2)), {}
    )
    with pytest.raises(ValueError, match="window"):
        build_lift_bundle(
            b"{}",
            solution,
            None,
            lift_name="back_squat",
            window_start_s=0.9,
            window_s=0.5,
        )
    with pytest.raises(ValueError, match="window_s"):
        build_lift_bundle(
            b"{}",
            solution,
            None,
            lift_name="back_squat",
            window_start_s=0.0,
            window_s=0.0,
        )


def test_lift_to_full_body_rejects_a_bad_seed() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5), np.zeros((5, 3)), np.zeros((5, 2)), {}
    )
    with pytest.raises(ValueError, match="seed_q"):
        lift_to_full_body(LIFT_SPEC.read_bytes(), solution, np.zeros(3))
    with pytest.raises(TypeError, match="spec_bytes"):
        lift_to_full_body("not bytes", solution, np.zeros(41))  # type: ignore[arg-type]
