"""Engine-free contract tests for the ClubheadSeries adapters (GCV-15/16).

The real-engine parity tests in ``test_adapters.py`` skip on hosts without
Drake, OpenSim or Pinocchio.  These tests drive each adapter with a duck-typed
fake that exposes only the binding calls the adapter makes, and check that the
adapter forwards every sample's pose and twist to :func:`rigid_body_series`
unchanged and in order.  They cover wiring and preconditions only; numerical
agreement with the real bindings stays with the parity tests.
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.adapters import (
    ClubFaceSpec,
    clubhead_series_from_drake,
    clubhead_series_from_opensim,
    clubhead_series_from_pinocchio,
    rigid_body_series,
)
from src.shared.python.impact_parameters.adapters.club_face import (
    check_trajectory,
    empty_pose_twist,
)
from src.shared.python.impact_parameters.panel_model import build_impact_card

pytestmark = pytest.mark.unit

N = 9
NAMES = ("j1", "j2", "j3")
SPEC = ClubFaceSpec(face_center_body_m=(0.02, -0.01, 0.03))


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _kinematics(q: np.ndarray, v: np.ndarray):
    """Toy FK: origin at ``q``, yaw ``q[2]``, twist ``(v, (0, 0, v[2]))``."""
    return (
        np.array(q, dtype=float),
        _rot_z(float(q[2])),
        np.array(v, dtype=float),
        np.array([0.0, 0.0, float(v[2])]),
    )


def _rollout():
    t = np.arange(N) * 0.005
    q = np.column_stack([np.sin(t), np.cos(2 * t), 0.3 * t])
    v = np.column_stack([np.cos(t), -2 * np.sin(2 * t), np.full(N, 0.3)])
    return t, q, v


def _expected(t, q, v) -> ClubheadSeries:
    pos, rot, lin, ang = empty_pose_twist(N)
    for i in range(N):
        pos[i], rot[i], lin[i], ang[i] = _kinematics(q[i], v[i])
    return rigid_body_series(t, pos, rot, lin, ang, SPEC)


def _assert_same(got: ClubheadSeries, want: ClubheadSeries) -> None:
    np.testing.assert_allclose(got.face_center_m, want.face_center_m, atol=1e-12)
    np.testing.assert_allclose(got.velocity_mps, want.velocity_mps, atol=1e-12)
    np.testing.assert_allclose(got.face_normal, want.face_normal, atol=1e-12)
    np.testing.assert_allclose(got.toe_axis, want.toe_axis, atol=1e-12)


# ----------------------------------------------------------------- Drake ----
class _DrakePose:
    def __init__(self, p, rot):
        self._p, self._rot = p, rot

    def translation(self):
        return self._p

    def rotation(self):
        return types.SimpleNamespace(matrix=lambda: self._rot)


class _DrakeTwist:
    def __init__(self, lin, ang):
        self._lin, self._ang = lin, ang

    def translational(self):
        return self._lin

    def rotational(self):
        return self._ang


class _FakeDrakePlant:
    def HasBodyNamed(self, name):  # noqa: N802 - mirrors the pydrake API
        return name == "club"

    def num_positions(self):
        return 3

    def num_velocities(self):
        return 3

    def GetBodyByName(self, name):  # noqa: N802 - mirrors the pydrake API
        return name

    def CreateDefaultContext(self):  # noqa: N802 - mirrors the pydrake API
        return {}

    def SetPositions(self, context, q):  # noqa: N802 - mirrors the pydrake API
        context["q"] = np.array(q)

    def SetVelocities(self, context, v):  # noqa: N802 - mirrors the pydrake API
        context["v"] = np.array(v)

    def EvalBodyPoseInWorld(self, context, body):  # noqa: N802 - pydrake API
        p, rot, _, _ = _kinematics(context["q"], context["v"])
        return _DrakePose(p, rot)

    def EvalBodySpatialVelocityInWorld(self, context, body):  # noqa: N802
        _, _, lin, ang = _kinematics(context["q"], context["v"])
        return _DrakeTwist(lin, ang)


def test_drake_adapter_forwards_each_sample():
    t, q, v = _rollout()
    got = clubhead_series_from_drake(_FakeDrakePlant(), t, q, v, "club", SPEC)
    _assert_same(got, _expected(t, q, v))


def test_drake_adapter_preconditions():
    t, q, v = _rollout()
    with pytest.raises(ValueError, match="no body"):
        clubhead_series_from_drake(_FakeDrakePlant(), t, q, v, "nope")
    with pytest.raises(ValueError, match="widths"):
        clubhead_series_from_drake(_FakeDrakePlant(), t, q[:, :2], v[:, :2], "club")


# ---------------------------------------------------------------- OpenSim ----
class _OsimVec:
    def __init__(self, values):
        self._values = values

    def get(self, *index):
        return self._values[index if len(index) > 1 else index[0]]


class _OsimCoordinate:
    def __init__(self, column):
        self._column = column

    def setValue(self, state, value, enforce):  # noqa: N802 - OpenSim API
        assert enforce is False
        state["q"][self._column] = value

    def setSpeedValue(self, state, value):  # noqa: N802 - OpenSim API
        state["v"][self._column] = value


class _OsimBody:
    def getTransformInGround(self, state):  # noqa: N802 - OpenSim API
        p, rot, _, _ = _kinematics(state["q"], state["v"])
        return types.SimpleNamespace(p=lambda: _OsimVec(p), R=lambda: _OsimVec(rot))

    def getVelocityInGround(self, state):  # noqa: N802 - OpenSim API
        _, _, lin, ang = _kinematics(state["q"], state["v"])
        return _OsimVec((_OsimVec(ang), _OsimVec(lin)))


class _OsimSet:
    def __init__(self, items):
        self._items = items

    def hasComponent(self, name):  # noqa: N802 - OpenSim API
        return name in self._items

    def get(self, name):
        return self._items[name]


class _FakeOsimModel:
    def __init__(self):
        self.realized = 0

    def getBodySet(self):  # noqa: N802 - OpenSim API
        return _OsimSet({"club": _OsimBody()})

    def getCoordinateSet(self):  # noqa: N802 - OpenSim API
        return _OsimSet({n: _OsimCoordinate(i) for i, n in enumerate(NAMES)})

    def initSystem(self):  # noqa: N802 - OpenSim API
        return {"q": np.zeros(3), "v": np.zeros(3)}

    def realizeVelocity(self, state):  # noqa: N802 - OpenSim API
        self.realized += 1


def _named(arr):
    return {n: arr[:, i] for i, n in enumerate(NAMES)}


def test_opensim_adapter_forwards_each_sample():
    t, q, v = _rollout()
    model = _FakeOsimModel()
    got = clubhead_series_from_opensim(model, t, _named(q), _named(v), "club", SPEC)
    _assert_same(got, _expected(t, q, v))
    assert model.realized == N


def test_opensim_adapter_preconditions():
    t, q, v = _rollout()
    model = _FakeOsimModel()
    with pytest.raises(ValueError, match="at least 2"):
        clubhead_series_from_opensim(model, t[:1], {}, {}, "club")
    with pytest.raises(ValueError, match="same keys"):
        clubhead_series_from_opensim(model, t, _named(q), {"j1": v[:, 0]}, "club")
    short = {n: x[:-1] for n, x in _named(q).items()}
    with pytest.raises(ValueError, match="one value per time"):
        clubhead_series_from_opensim(model, t, short, _named(v), "club")
    with pytest.raises(ValueError, match="no body"):
        clubhead_series_from_opensim(model, t, _named(q), _named(v), "nope")


# -------------------------------------------------------------- Pinocchio ----
class _FakePinModel:
    nq = 3
    nv = 3

    def existFrame(self, name):  # noqa: N802 - Pinocchio API
        return name == "club"

    def getFrameId(self, name):  # noqa: N802 - Pinocchio API
        return 7

    def createData(self):  # noqa: N802 - Pinocchio API
        return types.SimpleNamespace(oMf={}, state=None)


def _fake_pinocchio() -> types.ModuleType:
    pin = types.ModuleType("pinocchio")
    pin.LOCAL_WORLD_ALIGNED = "LWA"

    def forward_kinematics(model, data, q, v):
        data.state = _kinematics(q, v)

    def update_frame_placements(model, data):
        p, rot, _, _ = data.state
        data.oMf = {7: types.SimpleNamespace(translation=p, rotation=rot)}

    def get_frame_velocity(model, data, fid, reference):
        assert fid == 7 and reference == "LWA"
        _, _, lin, ang = data.state
        return types.SimpleNamespace(linear=lin, angular=ang)

    pin.forwardKinematics = forward_kinematics
    pin.updateFramePlacements = update_frame_placements
    pin.getFrameVelocity = get_frame_velocity
    return pin


def test_pinocchio_adapter_forwards_each_sample():
    t, q, v = _rollout()
    with patch.dict("sys.modules", {"pinocchio": _fake_pinocchio()}):
        got = clubhead_series_from_pinocchio(_FakePinModel(), t, q, v, "club", SPEC)
    _assert_same(got, _expected(t, q, v))


def test_pinocchio_adapter_preconditions():
    t, q, v = _rollout()
    with patch.dict("sys.modules", {"pinocchio": _fake_pinocchio()}):
        with pytest.raises(ValueError, match="no frame"):
            clubhead_series_from_pinocchio(_FakePinModel(), t, q, v, "nope")
        with pytest.raises(ValueError, match="widths"):
            clubhead_series_from_pinocchio(_FakePinModel(), t, q[:, :2], v, "club")


# -------------------------------------------------------- shared kernel -----
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normal_body": (1.0, 0.0)}, "finite 3-vector"),
        ({"normal_body": (0.0, 0.0, 0.0)}, "nonzero"),
        ({"grip_body": (1.0, 0.0, 0.0)}, "grip_body must be orthogonal"),
    ],
)
def test_face_spec_rejects_bad_axes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ClubFaceSpec(**kwargs)


def test_empty_pose_twist_rejects_negative_count():
    with pytest.raises(ValueError, match="non-negative"):
        empty_pose_twist(-1)


def test_rigid_body_series_rejects_non_finite_and_2d_times():
    t = np.arange(3) * 0.01
    rot = np.tile(np.eye(3), (3, 1, 1))
    pos = np.zeros((3, 3))
    pos[1, 0] = np.nan
    with pytest.raises(ValueError, match="origins_m must be finite"):
        rigid_body_series(t, pos, rot, np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="1-D"):
        rigid_body_series(t[None, :], pos, rot, np.zeros((3, 3)), np.zeros((3, 3)))


def test_check_trajectory_preconditions():
    t, q, v = _rollout()
    with pytest.raises(ValueError, match="at least 2"):
        check_trajectory(t[:1], q[:1], v[:1])
    with pytest.raises(ValueError, match="velocities"):
        check_trajectory(t, q, v[:-1])
    bad_q = q.copy()
    bad_q[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        check_trajectory(t, bad_q, v)
    bad_t = t.copy()
    bad_t[0] = np.nan
    with pytest.raises(ValueError, match="times_s must be finite"):
        check_trajectory(bad_t, q, v)


# ------------------------------------------------------------ card model ----
def _params():
    t = np.arange(41) * 0.002
    vel = np.tile([0.0, -40.0, -3.0], (41, 1))
    series = ClubheadSeries(
        t,
        np.cumsum(vel * 0.002, axis=0),
        vel,
        face_normal=np.tile([0.0, -0.97, 0.22], (41, 1)),
        toe_axis=np.tile([1.0, 0.0, 0.0], (41, 1)),
        grip_axis=np.tile([0.0, 0.22, 0.97], (41, 1)),
    )
    return extract_impact_parameters(series, TargetFrame(), impact_index=30)


def _card_row(card, key):
    return next(r for r in card.rows if r.key == key)


def test_card_shows_impact_location_and_labelled_smash():
    params = dataclasses.replace(
        _params(),
        toe_mm=4.0,
        high_mm=2.5,
        smash_factor=1.45,
        smash_factor_label="impact model calibration: provisional",
    )
    card = build_impact_card(params)
    location = _card_row(card, "impact_location")
    assert location.value == 4.0 and location.note == "high 2.5 mm"
    smash = _card_row(card, "smash_factor")
    assert smash.value == pytest.approx(1.45)
    assert smash.note == "impact model calibration: provisional"


def test_card_marks_missing_value_without_reason_as_not_computed():
    params = dataclasses.replace(_params(), attack_angle_deg=None, unavailable={})
    row = _card_row(build_impact_card(params), "attack_angle_deg")
    assert row.value is None and row.reason == "not computed"
