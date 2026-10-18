"""The OpenSim lift foot carries the bundle's tibial torsion (OSL-18, #13157).

Red tests against the real OpenSim 4.6 runtime, written before the code.

Diagnosis (measured; #13157): the same-input squat bundle grafts the
Rajagopal lower limbs and then turns each foot out by the #12814 default
tibial-torsion offset (10 deg about the shank long axis through the ankle
centre; ``motion_matching.tibial_torsion``, Seber et al. 2000).  The OpenSim
lift model is the same Rajagopal leg without that offset, so its foot cannot
follow the bundle foot without a hip rotation that moves the knee markers,
and the forefoot, heel and ankle markers push the squat marker RMS to
2.45 cm, past the 2 cm gate.

* **Targets.**  ``lift_targets`` carries the bundle spec's recorded torsion.
* **Model.**  ``build_lift_model(tibial_torsion=...)`` rotates each
  ``ankle_<side>`` parent frame on the tibia exactly as the spec does, so the
  talus sits in the tibia as the bundle's talus does.
* **Gate.**  The squat IK then meets the unchanged 2 cm / 4 cm marker gate
  (``test_opensim_lift_bar_reach``, ``test_back_rack``).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES

pytestmark = pytest.mark.unit

#: The OpenSim talus-in-tibia rotation must equal the bundle spec's to this
#: angle.  Declared before measuring: both apply the same rotation to the same
#: generic Rajagopal frames, so only the scaled shank axis may differ.
_FOOT_ROTATION_TOL_RAD = math.radians(0.05)


def _real_opensim() -> None:
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")


def _base_model():
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        return resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(f"muscle-bearing body not available: {exc}")


@pytest.fixture(scope="module")
def squat_spec() -> dict:
    from src.shared.python.motion_matching.same_input import InputBundle

    path = SAME_INPUT_LIFT_BUNDLES["squat"]
    assert path is not None
    return json.loads(InputBundle.load(path).spec_bytes)


def _spec_talus_in_tibia(spec: dict, side: str) -> np.ndarray:
    """Talus orientation in the tibia frame at zero ankle angle (spec)."""
    joint = next(j for j in spec["joints"] if j["name"] == f"ankle_{side}")
    base = np.asarray(joint["parent_to_base"], dtype=float)[:3, :3]
    follower = np.asarray(joint["child_to_follower"], dtype=float)[:3, :3]
    return base @ follower.T


def _model_talus_in_tibia(build, side: str) -> np.ndarray:
    """Talus orientation in the tibia frame at zero ankle angle (OpenSim)."""
    model = build.model
    state = model.initSystem()
    coords = model.getCoordinateSet()
    for name in (f"ankle_angle_{side}", f"subtalar_angle_{side}"):
        coords.get(name).setValue(state, 0.0)
    model.realizePosition(state)
    bodies = model.getBodySet()
    tibia = bodies.get(f"tibia_{side}").getTransformInGround(state).R()
    talus = bodies.get(f"talus_{side}").getTransformInGround(state).R()

    def mat(r) -> np.ndarray:
        return np.array([[r.get(i, j) for j in range(3)] for i in range(3)])

    return mat(tibia).T @ mat(talus)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    return float(Rotation.from_matrix(a.T @ b).magnitude())


def _build(torsion):
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )

    return build_lift_model(
        "squat",
        body_mass_kg=80.0,
        height_m=1.78,
        bar_mass_kg=100.0,
        base_model_path=_base_model(),
        tibial_torsion=torsion,
    )


@pytest.mark.requires_mujoco
@pytest.mark.integration
def test_targets_carry_the_bundle_tibial_torsion(squat_spec: dict) -> None:
    from src.shared.python.lifting.bundle_targets import lift_targets
    from src.shared.python.motion_matching.tibial_torsion import TIBIAL_TORSION_KEY

    path = SAME_INPUT_LIFT_BUNDLES["squat"]
    targets = lift_targets("squat", path, stride=400)
    assert dict(targets.tibial_torsion) == squat_spec[TIBIAL_TORSION_KEY]
    assert targets.tibial_torsion["r"] == targets.tibial_torsion["l"] == 10.0


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("side", ["r", "l"])
def test_model_foot_turns_like_the_bundle_foot(squat_spec: dict, side: str) -> None:
    from src.shared.python.motion_matching.tibial_torsion import TIBIAL_TORSION_KEY

    _real_opensim()
    build = _build(squat_spec[TIBIAL_TORSION_KEY])
    expected = _spec_talus_in_tibia(squat_spec, side)
    # The bundle foot really is turned (10 deg), so the check is not vacuous.
    assert _angle(expected, np.eye(3)) == pytest.approx(math.radians(10.0), abs=1e-6)
    assert _angle(_model_talus_in_tibia(build, side), expected) <= (
        _FOOT_ROTATION_TOL_RAD
    )
    record = build.receipt["tibial_torsion"]
    assert record["r"] == record["l"] == 10.0
    assert record["source"] == squat_spec[TIBIAL_TORSION_KEY]["source"]


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_model_without_torsion_keeps_the_generic_foot() -> None:
    _real_opensim()
    build = _build(None)
    for side in ("r", "l"):
        assert _angle(_model_talus_in_tibia(build, side), np.eye(3)) <= 1e-9
    assert build.receipt["tibial_torsion"] is None


@pytest.mark.requires_opensim
def test_model_rejects_torsion_outside_the_adult_range() -> None:
    _real_opensim()
    with pytest.raises(ValueError, match="tibial torsion"):
        _build({"r": 60.0, "l": 10.0, "source": "out of range"})


#: A foot scale equal to the bundle's own heel-to-forefoot ratio to this
#: relative error.  Declared before measuring: both are ratios of the same
#: rigid calcaneus stations, so only round-off may differ.
_FOOT_SCALE_RTOL = 1e-9


@pytest.mark.requires_opensim
@pytest.mark.requires_mujoco
@pytest.mark.integration
def test_foot_is_scaled_by_the_bundle_foot_not_the_shank() -> None:
    """The bundle's calcaneus stations measure its foot; the shank does not.

    Measured (#13157): the bundle's heel and forefoot stations are 0.150 m
    apart, exactly the generic station separation, while the knee-to-ankle
    ratio is 1.10; scaling the foot by the shank put the model's heel and
    forefoot 0.165 m apart.
    """
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.shared.python.lifting.bundle_targets import lift_targets

    _real_opensim()
    targets = lift_targets("squat", SAME_INPUT_LIFT_BUNDLES["squat"], stride=400)
    scales = ikid.bundle_segment_scales(targets, _base_model())
    points = targets.landmarks(0)
    for side in ("r", "l"):
        heel = np.asarray(targets.foot_stations_m[f"heel_{side}"][1])
        forefoot = np.asarray(targets.foot_stations_m[f"forefoot_{side}"][1])
        foot = np.linalg.norm(points[f"heel_{side}"] - points[f"forefoot_{side}"])
        expected = float(foot / np.linalg.norm(heel - forefoot))
        for body in (f"talus_{side}", f"calcn_{side}", f"toes_{side}"):
            assert scales[body] == pytest.approx(expected, rel=_FOOT_SCALE_RTOL)
        # The shank keeps its knee-to-ankle factor, which differs here.
        assert abs(scales[f"tibia_{side}"] - expected) > 0.05
