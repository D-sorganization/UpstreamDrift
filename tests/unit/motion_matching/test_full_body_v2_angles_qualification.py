"""Cross-engine anatomical-angle parity of full-body-v2 (HRG-Q1, #13176).

Every engine is driven through the same seeded, smooth trajectory q(t), and
the HRG-E2 wrist and forearm angles (``anatomical_angles.full_body_v1``) are
extracted from its own forward kinematics. The arm chain bodies carry the spec
frames ``{L,R}S`` (humerus), ``{L,R}F`` (forearm) and ``{L,R}W`` (hand)
rigidly; the neutral calibration at ``q = 0`` absorbs each frame's constant
offset, so these frames stand in for the bodies exactly.

The reference is computed from the specification alone: ``spec_body_poses``
composes the spec joints and each frame is its body pose times the frame
placement. Each engine's frames come from ``VectorPlant.kinematic_frames``.
All twelve angle series (both sides; wrist flexion, ulnar deviation, axial
rotation; forearm flexion, carrying angle, pronation) must equal the
reference to ``TOL`` at every sample. The worst error measured over 40
samples is 5.6e-16 rad on MuJoCo, Pinocchio, Drake and OpenSim, so ``TOL``
(1e-12 rad) keeps the HRG-Q1 precision budget, well inside the issue's 1e-6
rad acceptance, and is never loosened.

MyoSuite is reached by segment-orientation IK on the thorax, not by exporting
the spec (``EXPORT_STATUS["myosuite"]``), so it has no arm chain driven by
the same q(t).
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.anatomical_angles import full_body_v1 as fb
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2_geometry as geo
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
N_SAMPLES = 40
SEED = 1317603
TOL = 1e-12
# Spec frame carried rigidly by each arm-chain body, by side-chain role.
_ROLE_FRAME = {"humerus": "S", "forearm_body": "F", "hand": "W"}
Frames = Callable[[np.ndarray], dict[str, np.ndarray]]


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-Q1 angles fixture")


@pytest.fixture(scope="module")
def trajectory(doc) -> np.ndarray:
    """Seeded smooth q(t): one sinusoid per coordinate, 0.2-0.6 rad or m."""
    n = len(doc["coordinate_order"])
    rng = np.random.default_rng(SEED)
    amplitude = rng.uniform(0.2, 0.6, n)
    frequency = rng.uniform(0.5, 2.0, n)
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    t = np.linspace(0.0, 1.0, N_SAMPLES)
    return amplitude * np.sin(2.0 * np.pi * np.outer(t, frequency) + phase)


def _spec_frames(doc: dict) -> Frames:
    """Spec-only world poses of every spec frame at ``q``."""
    order = doc["coordinate_order"]

    def frames(q: np.ndarray) -> dict[str, np.ndarray]:
        poses = spec_body_poses(doc, dict(zip(order, q, strict=True)))
        return {
            f["name"]: poses[f["body"]] @ np.asarray(f["placement"], dtype=float)
            for f in doc["frames"]
        }

    return frames


def _angles(frames: Frames, trajectory: np.ndarray) -> dict[str, np.ndarray]:
    """Wrist and forearm angle series of both arms along ``trajectory``."""
    neutral = frames(np.zeros(trajectory.shape[1]))
    samples = [frames(q) for q in trajectory]
    out = {}
    for side in "LR":
        chain = fb.side_chain(side)
        frame_of = {getattr(chain, r): side + f for r, f in _ROLE_FRAME.items()}
        calibration = fb.calibrate_neutral(
            side, {b: neutral[f][:3, :3] for b, f in frame_of.items()}
        )
        series = {
            b: np.stack([s[f][:3, :3] for s in samples]) for b, f in frame_of.items()
        }
        arm = fb.full_body_v1_wrist_forearm(
            calibration, series, receipt_path="HRG-Q1 angles qualification"
        )
        for part in ("wrist", "forearm"):
            for name, values in vars(getattr(arm, part)).items():
                if name.endswith("_rad"):
                    out[f"{side}.{part}.{name}"] = np.asarray(values, dtype=float)
    return out


@pytest.fixture(scope="module")
def reference(doc, trajectory) -> dict[str, np.ndarray]:
    return _angles(_spec_frames(doc), trajectory)


def _real_engine(engine: str) -> None:
    """Skip when the engine is missing or a CI ``MagicMock`` stub."""
    if engine == "opensim":
        module, attr = require_opensim(), "Model"
    else:
        name, attr = {
            "mujoco": ("mujoco", "MjModel"),
            "pinocchio": ("pinocchio", "Model"),
            "drake": ("pydrake.all", "MultibodyPlant"),
        }[engine]
        module = pytest.importorskip(name)
    if not (
        isinstance(module, types.ModuleType)
        and isinstance(getattr(module, attr, None), type)
    ):
        pytest.skip(f"{engine} is a stub or not the physics library")


def test_reference_angles_are_finite_and_exercised(reference) -> None:
    """Twelve finite series that move: the parity below is not vacuous."""
    assert len(reference) == 12
    for name, values in reference.items():
        assert values.shape == (N_SAMPLES,), name
        assert np.all(np.isfinite(values)), name
    assert max(np.ptp(v) for v in reference.values()) > 0.1


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("mujoco", marks=pytest.mark.requires_mujoco),
        pytest.param("pinocchio", marks=pytest.mark.requires_pinocchio),
        pytest.param("drake", marks=pytest.mark.requires_drake),
        pytest.param("opensim", marks=pytest.mark.requires_opensim),
    ],
)
def test_anatomical_angles_identical_from_same_q(
    engine, doc, trajectory, reference
) -> None:
    """Each engine's wrist and forearm angles equal the spec's along q(t)."""
    _real_engine(engine)
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = VectorPlant(engine, json.dumps(doc, sort_keys=True).encode("utf-8"))
    got = _angles(plant.kinematic_frames, trajectory)
    errors = {k: float(np.abs(got[k] - reference[k]).max()) for k in reference}
    worst = max(errors, key=errors.get)
    assert errors[worst] <= TOL, f"{engine}: {worst} differs by {errors[worst]:.3g}"


def test_myosuite_has_no_arm_chain_on_the_same_q() -> None:
    """MyoSuite is IK-mapped on the thorax, so its angles cell is exempt."""
    status = geo.EXPORT_STATUS["myosuite"]
    assert status["status"] == "mapped_thorax_only"
    assert "segment-orientation IK" in status["reason"]
