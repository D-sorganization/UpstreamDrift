"""API contract tests for GET /analysis/impact-parameters (GCV-17, #11723)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.shared.python.impact_parameters import ClubheadSeries

pytestmark = pytest.mark.unit

N = 41


def _series(face=True):
    t = np.arange(N) * 0.002
    vel = np.tile([0.0, -40.0, -3.0], (N, 1))
    pos = np.cumsum(vel * 0.002, axis=0)
    if face:
        return ClubheadSeries(
            t,
            pos,
            vel,
            face_normal=np.tile([0.0, -0.97, 0.22], (N, 1)),
            toe_axis=np.tile([1.0, 0.0, 0.0], (N, 1)),
            grip_axis=np.tile([0.0, 0.22, 0.97], (N, 1)),
        )
    return ClubheadSeries(t, pos, vel, face_unobservable_reason="roll unobservable")


class _Service:
    def __init__(self, runs):
        self._runs = runs

    def get_run(self, run_id=None):
        return self._runs.get(run_id or "active")


def _run(data, engine=None):
    return SimpleNamespace(
        run_id="r1", engine_type="mock", simulation_data=data, engine=engine
    )


def _client(runs):
    app = FastAPI()
    app.include_router(router)
    svc = _Service(runs)
    app.dependency_overrides[get_simulation_service] = lambda: svc
    return TestClient(app)


def test_contract_for_run_with_clubhead_series():
    c = _client({"r1": _run({"clubhead_series": _series()})})
    body = c.get("/analysis/impact-parameters", params={"run_id": "r1"}).json()
    assert body["available"] is True and body["run_id"] == "r1"
    rows = {r["key"]: r for r in body["rows"]}
    assert (
        rows["clubhead_speed"]["unit"] == "mph" and rows["clubhead_speed"]["value"] > 80
    )
    assert rows["attack_angle_deg"]["value"] < 0
    assert rows["smash_factor"]["value"] is None and rows["smash_factor"]["reason"]
    assert body["frame"]["handedness"] == "right"
    assert set(body["d_plane"]) >= {"club_path_deg", "face_angle_deg"}


def test_target_dir_changes_path_and_units_toggle():
    c = _client({"r1": _run({"clubhead_series": _series()})})
    base = c.get("/analysis/impact-parameters", params={"run_id": "r1"}).json()
    rot = c.get(
        "/analysis/impact-parameters",
        params={"run_id": "r1", "target_dir": "1,-1", "units": "m/s"},
    ).json()
    brow = {r["key"]: r for r in base["rows"]}
    rrow = {r["key"]: r for r in rot["rows"]}
    assert rrow["clubhead_speed"]["unit"] == "m/s"
    assert rrow["club_path_deg"]["value"] != pytest.approx(
        brow["club_path_deg"]["value"]
    )
    assert rrow["attack_angle_deg"]["value"] == pytest.approx(
        brow["attack_angle_deg"]["value"]
    )


def test_unobservable_face_is_null_with_reason():
    c = _client({"r1": _run({"clubhead_series": _series(face=False)})})
    body = c.get("/analysis/impact-parameters", params={"run_id": "r1"}).json()
    face = next(r for r in body["rows"] if r["key"] == "face_angle_deg")
    assert face["value"] is None and "unobservable" in face["reason"]


def test_run_without_series_is_unavailable_not_zero():
    c = _client({"r1": _run({})})
    r = c.get("/analysis/impact-parameters", params={"run_id": "r1"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and body["rows"] == [] and body["reason"]


def test_engine_provider_is_used():
    engine = SimpleNamespace(get_clubhead_series=lambda: _series())
    c = _client({"r1": _run({}, engine)})
    assert c.get("/analysis/impact-parameters", params={"run_id": "r1"}).json()[
        "available"
    ]


def test_low_speed_is_unavailable_with_reason():
    s = _series()
    slow = ClubheadSeries(
        s.times_s,
        s.face_center_m,
        s.velocity_mps * 0.001,
        face_normal=s.face_normal,
        toe_axis=s.toe_axis,
        grip_axis=s.grip_axis,
    )
    c = _client({"r1": _run({"clubhead_series": slow})})
    body = c.get("/analysis/impact-parameters", params={"run_id": "r1"}).json()
    assert body["available"] is False and "minimum" in body["reason"]


def test_errors():
    c = _client({"r1": _run({"clubhead_series": _series()})})
    assert (
        c.get("/analysis/impact-parameters", params={"run_id": "nope"}).status_code
        == 404
    )
    assert (
        c.get(
            "/analysis/impact-parameters", params={"run_id": "r1", "target_dir": "x"}
        ).status_code
        == 400
    )
    assert (
        c.get(
            "/analysis/impact-parameters", params={"run_id": "r1", "target_dir": "0,0"}
        ).status_code
        == 400
    )
    assert (
        c.get(
            "/analysis/impact-parameters", params={"run_id": "r1", "units": "kph"}
        ).status_code
        == 422
    )
    assert (
        c.get(
            "/analysis/impact-parameters", params={"run_id": "r1", "impact_index": 999}
        ).status_code
        == 400
    )


def test_mujoco_run_resolves_series_through_adapter():
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><worldbody><body name="clubhead" pos="0 0 1">
      <joint name="s" type="slide" axis="0 -1 0"/><joint name="h" type="hinge" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    t = np.arange(N) * 0.002
    q = np.column_stack([np.zeros(N) + 40.0 * t, 0.1 * t])
    v = np.column_stack([np.full(N, 40.0), np.full(N, 0.1)])
    run = _run(
        {"times": t, "joint_positions": q, "joint_velocities": v},
        SimpleNamespace(model=model),
    )
    body = (
        _client({"r1": run})
        .get("/analysis/impact-parameters", params={"run_id": "r1", "impact_index": 20})
        .json()
    )
    speed = next(r for r in body["rows"] if r["key"] == "clubhead_speed")
    assert body["available"] and speed["value"] == pytest.approx(
        40.0 * 2.2369362920544, rel=1e-6
    )
