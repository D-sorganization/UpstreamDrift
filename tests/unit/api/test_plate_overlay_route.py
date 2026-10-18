"""Measured plate overlay on /analysis/ground-reaction (OF-07, #13087)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.api.services.plate_overlay_service import (
    PLATE_TRIAL_ROOT_ENV,
    PlateRootNotConfiguredError,
    resolve_trial_path,
)
from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROUTE = "/analysis/ground-reaction/plate-overlay"
TIMES = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025]


def _series() -> GroundReactionSeries:
    def stance(fz: float):
        left = ContactSet(np.array([[0.0, 0.0, fz]]), np.array([[0.0, 0.15, 0.0]]))
        return analyze_ground_reaction(
            {"left": left, "right": ContactSet.empty()}, (0.0, 0.0, 0.95)
        )

    return GroundReactionSeries.from_breakdowns(TIMES, [stance(700.0)] * len(TIMES))


class _Service:
    def __init__(self, runs: dict) -> None:
        self._runs = runs

    def get_run(self, run_id=None):
        return self._runs.get(run_id or "active")


def _client(data: dict[str, Any] | None = None) -> TestClient:
    run = SimpleNamespace(
        run_id="r1",
        engine_type="mujoco",
        simulation_data=data
        if data is not None
        else {"ground_reaction_series": _series()},
        engine=None,
    )
    app = FastAPI()
    app.include_router(router)
    service = _Service({"r1": run})
    app.dependency_overrides[get_simulation_service] = lambda: service
    return TestClient(app)


def _body(trial: str, **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "run_id": "r1",
        "trial": trial,
        "handedness": "RH",
        "origin_m": [0.0, 0.0, 0.0],
        "declared": {
            "force_floor_n": 2.0,
            "force_fraction": 0.01,
            "cop_m": 0.01,
            "sensing_depth_sigma_m": 0.002,
            "uncertainty_source": "declared hypothesis for tests",
            "sensing_depth_m": 0.015,
            "capacity_n": 3923.0,
        },
        "alignment": {
            "offset_s": 0.0,
            "sigma_offset_s": 0.001,
            "kind": "manual",
            "note": "test clock",
        },
        "reading": {"skew_s": 0.0, "cop_skew": "unknown"},
    }
    body.update(changes)
    return body


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "plates"
    directory.mkdir()
    monkeypatch.setenv(PLATE_TRIAL_ROOT_ENV, str(directory))
    return directory


def test_overlay_happy_path(root: Path) -> None:
    json_path = w.write_swing(root)
    response = _client().post(ROUTE, json=_body(json_path.name))
    assert response.status_code == 200, response.text
    body = response.json()
    left = body["traces"]["measured_left_force_n"]
    assert left["z"] == pytest.approx(list(w.FZ_L))
    assert left["x"] == [None] * 6 and left["y"] == [None] * 6
    assert body["measured_qualified"] is False
    assert body["observability"]["fx"] is False
    assert "force.impact" in body["events"]
    assert body["traces"]["left_force_n"]["z"][0] == pytest.approx(700.0)


def test_unconfigured_root_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLATE_TRIAL_ROOT_ENV, raising=False)
    response = _client().post(ROUTE, json=_body("x.json"))
    assert response.status_code == 503


@pytest.mark.parametrize(
    "trial", ["/etc/passwd", "../x.json", "a/../x.json", "./x.json", "a\\b.json", ""]
)
def test_paths_outside_the_root_are_refused(root: Path, trial: str) -> None:
    with pytest.raises(ValueError):
        resolve_trial_path(trial, root)
    assert _client().post(ROUTE, json=_body(trial)).status_code == 400


def test_symlinked_file_and_directory_are_refused(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    real = w.write_swing(outside)
    (root / "link.json").symlink_to(real)
    (root / "dir").symlink_to(outside, target_is_directory=True)
    for trial in ("link.json", f"dir/{real.name}"):
        with pytest.raises(ValueError, match="symbolic link"):
            resolve_trial_path(trial, root)
        assert _client().post(ROUTE, json=_body(trial)).status_code == 400


def test_missing_file_and_bad_root(root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        resolve_trial_path("nope.json", root)
    with pytest.raises(PlateRootNotConfiguredError):
        resolve_trial_path("a.json", tmp_path / "missing")


def test_invalid_declarations_and_handedness_are_400(root: Path) -> None:
    name = w.write_swing(root).name
    bad = _body(name)
    bad["declared"]["capacity_n"] = -1.0
    assert _client().post(ROUTE, json=bad).status_code == 400
    mismatch = _body(name, handedness="LH")
    assert _client().post(ROUTE, json=mismatch).status_code == 400
    assert _client().post(ROUTE, json=_body(name, origin_m=[0.0])).status_code == 422


def test_cross_correlation_alignment_needs_its_fitted_model(root: Path) -> None:
    name = w.write_swing(root).name
    fitted = {
        "offset_s": 0.0,
        "sigma_offset_s": 0.001,
        "kind": "cross_correlation",
        "note": "fitted against the model's vertical force",
    }
    named = _body(name, alignment={**fitted, "derived_from_model": "model-a"})
    assert _client().post(ROUTE, json=named).status_code == 200
    assert _client().post(ROUTE, json=_body(name, alignment=fitted)).status_code == 400


def test_run_without_ground_reaction_stays_unavailable(root: Path) -> None:
    name = w.write_swing(root).name
    response = _client({}).post(ROUTE, json=_body(name))
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert os.environ[PLATE_TRIAL_ROOT_ENV] == str(root)


def _cop_cov(root: Path, reading: dict[str, Any]) -> list[Any]:
    json_path = w.write_swing(root)
    response = _client().post(ROUTE, json=_body(json_path.name, reading=reading))
    assert response.status_code == 200, response.text
    return response.json()["traces"]["measured_left_cop_cov_m2"]["x"]


def test_declared_independent_skew_carries_the_cop_covariance(root: Path) -> None:
    cov = _cop_cov(root, {"skew_s": 0.0005, "cop_skew": "independent"})
    assert any(v is not None for v in cov)


def test_undeclared_skew_relation_keeps_only_the_per_axis_bound(root: Path) -> None:
    cov = _cop_cov(root, {"skew_s": 0.0005, "cop_skew": "unknown"})
    assert all(v is None for v in cov)


def test_reading_declarations_are_required_and_validated(root: Path) -> None:
    json_path = w.write_swing(root)
    body = _body(json_path.name)
    del body["reading"]
    assert _client().post(ROUTE, json=body).status_code == 422
    bad = _body(json_path.name, reading={"skew_s": 0.0, "cop_skew": "shared"})
    assert _client().post(ROUTE, json=bad).status_code == 422
    negative = _body(json_path.name, reading={"skew_s": -1.0, "cop_skew": "unknown"})
    assert _client().post(ROUTE, json=negative).status_code in (400, 422)
