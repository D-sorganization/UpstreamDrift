"""Cross-engine loop-closure qualification of full-body-v2 (HRG-Q1, #13176).

The reference is computed from the specification alone: ``spec_body_poses``
composes the spec's joint transforms, and the weld residual is the relative
pose of the spec's two closure frames,

    T_rel = (T_a P_a)^-1 (T_b P_b),   r = [t_rel; rotvec(R_rel)],

where ``P_a`` and ``P_b`` are the closure placements. Seeded configurations
are projected onto ``r = 0`` over the trail-arm coordinates
(``project_onto_closure``) to ``PROJECT_TOL``.

Engines lay out their native pose residuals differently (row order, frame,
sign, and Pinocchio's se(3) logarithm), so two convention-free properties are
qualified for every engine through ``VectorPlant.closure_pose_residual``:

* the zero set: every engine's residual vanishes on the projected
  configurations (worst measured 2.2e-15 on all four engines);
* the magnitude: off the closure, the weld distance ``|t_rel|`` and relative
  rotation angle ``|rotvec|`` read from each engine's residual equal the
  reference (worst measured 2.7e-15 m and rad). MuJoCo, Drake and OpenSim
  report ``[linear; angular]`` with a point-difference linear part; Pinocchio
  reports ``-log6``, whose translation is recovered with the SE(3) exponential.

``TOL`` (1e-12) keeps the HRG-Q1 precision budget and is never loosened.
MyoSuite is reached by segment-orientation IK on the thorax, not by exporting
the spec (``EXPORT_STATUS["myosuite"]``), so it carries no grip weld to
qualify.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2_geometry as geo
from src.shared.python.motion_matching.pipeline.constants import (
    TRAIL_ARM_WELD_COORDINATES,
)
from src.shared.python.motion_matching.pipeline.weld_projection import (
    project_onto_closure,
)
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
N_CONFIGS = 24
SEED = 1317602
TOL = 1e-12
PROJECT_TOL = 1e-12


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-Q1 closure fixture")


@pytest.fixture(scope="module")
def reference(doc) -> Callable[[np.ndarray], np.ndarray]:
    """Spec-only weld residual ``[t_rel; rotvec(R_rel)]`` at ``q``."""
    order, closure = doc["coordinate_order"], doc["closure"]
    place_a = np.asarray(closure["placement_a"], dtype=float)
    place_b = np.asarray(closure["placement_b"], dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        poses = spec_body_poses(doc, dict(zip(order, q, strict=True)))
        rel = np.linalg.inv(poses[closure["body_a"]] @ place_a) @ (
            poses[closure["body_b"]] @ place_b
        )
        return np.r_[rel[:3, 3], Rotation.from_matrix(rel[:3, :3]).as_rotvec()]

    return residual


@pytest.fixture(scope="module")
def open_configs(doc) -> list[np.ndarray]:
    """Seeded configurations; the weld is open on every one of them."""
    rng = np.random.default_rng(SEED)
    return [
        rng.uniform(-0.6, 0.6, len(doc["coordinate_order"])) for _ in range(N_CONFIGS)
    ]


@pytest.fixture(scope="module")
def closed_configs(doc, reference, open_configs) -> list[np.ndarray]:
    """The open configurations projected onto the spec weld (trail arm only)."""
    order = doc["coordinate_order"]
    columns = [order.index(name) for name in TRAIL_ARM_WELD_COORDINATES]
    return [
        project_onto_closure(q, reference, columns, tol=PROJECT_TOL, max_iter=50)
        for q in open_configs
    ]


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _point_difference(residual: np.ndarray) -> tuple[float, float]:
    """Weld distance and angle of a ``[linear; angular]`` point-difference row."""
    return float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))


def _se3_logarithm(residual: np.ndarray) -> tuple[float, float]:
    """Weld distance and angle of an se(3) logarithm ``[v; w]`` (Pinocchio)."""
    v, w = residual[:3], residual[3:]
    theta = float(np.linalg.norm(w))
    hat = _hat(w)
    left_jacobian = (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta**2 * hat
        + (theta - np.sin(theta)) / theta**3 * hat @ hat
    )
    return float(np.linalg.norm(left_jacobian @ v)), theta


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


ENGINES = [
    pytest.param("mujoco", _point_difference, marks=pytest.mark.requires_mujoco),
    pytest.param("pinocchio", _se3_logarithm, marks=pytest.mark.requires_pinocchio),
    pytest.param("drake", _point_difference, marks=pytest.mark.requires_drake),
    pytest.param("opensim", _point_difference, marks=pytest.mark.requires_opensim),
]


def test_reference_projection_closes_an_open_weld(
    reference, open_configs, closed_configs
) -> None:
    """The reference is non-degenerate: open by centimetres, closed to TOL."""
    assert min(np.linalg.norm(reference(q)[:3]) for q in open_configs) > 1e-2
    assert max(np.abs(reference(q)).max() for q in closed_configs) <= PROJECT_TOL


@pytest.mark.parametrize(("engine", "invariants"), ENGINES)
def test_closure_residual_matches_spec_weld(
    engine, invariants, doc, reference, open_configs, closed_configs
) -> None:
    """Each engine's weld vanishes where the spec's does and opens as far."""
    _real_engine(engine)
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = VectorPlant(engine, json.dumps(doc, sort_keys=True).encode("utf-8"))
    on_manifold = max(
        np.abs(plant.closure_pose_residual(q)).max() for q in closed_configs
    )
    assert on_manifold <= TOL, f"{engine}: weld residual {on_manifold:.3g} on closure"
    for q in open_configs:
        got = invariants(plant.closure_pose_residual(q))
        want = _point_difference(reference(q))
        error = max(abs(g - w) for g, w in zip(got, want, strict=True))
        assert error <= TOL, (
            f"{engine}: weld distance/angle {got} vs spec {want} (error {error:.3g})"
        )


def test_myosuite_has_no_grip_weld_to_qualify() -> None:
    """MyoSuite is IK-mapped on the thorax, so its closure cell is exempt."""
    status = geo.EXPORT_STATUS["myosuite"]
    assert status["status"] == "mapped_thorax_only"
    assert "segment-orientation IK" in status["reason"]
