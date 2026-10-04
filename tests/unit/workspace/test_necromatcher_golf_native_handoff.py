"""Opt-in synthetic replay → authentic impact → API and Qt local research.

No native, worker, impact, flight, stamp or transport substitutions are allowed.
The API overrides only its temporary Library and local-client admission dependency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

from tests.unit.tools.test_necromatcher_replay_impact_native_handoff import (
    _prepare_offscreen_font,
    _recorded_case,
    _wait,
)
from tests.unit.workspace import test_necromatcher_replay as replay_fixtures

pytest.importorskip("PyQt6")
pytestmark = [pytest.mark.live_simulation, pytest.mark.integration]
replay_case = replay_fixtures.replay_case


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")


def _phase(case: dict[str, Any], name: str) -> None:
    case["timeline"][name] = time.monotonic() - case["started"]
    _save(case["evidence"] / "timeline.json", case["timeline"])


def _metadata(shot: str, session: str) -> Any:
    from src.shared.python.golf_simulator import AimContext, ShotMetadata, SourceKind

    return ShotMetadata(
        shot,
        session,
        AimContext(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1),
        "2026-10-04T12:00:00Z",
        source_kind=SourceKind.MODEL_CONTACT,
    )


def _impact_bundle(case: dict[str, Any], declaration: Path) -> str:
    from src.shared.python import workspace as owner

    value = json.loads(declaration.read_bytes())
    session = owner.NativeImpactSession(case["library"])
    try:
        run = session.submit(
            "native-replay",
            owner.ReplayImpactGeometry.from_record(value["geometry"]),
            owner.ReplayImpactSelection.from_record(value["selection"]),
            60.0,
        )["run_id"]
        deadline = time.monotonic() + 90
        while True:
            terminal = session.view("native-replay", run)
            if terminal["status"] not in {"pending", "running"}:
                break
            assert time.monotonic() < deadline, terminal
            time.sleep(0.05)
        assert terminal["status"] == "succeeded", terminal
        assert terminal["execution_verified"] and terminal["acceptance"] == "rejected"
        _save(case["evidence"] / "terminal.json", terminal)
        with ZipFile(session.download("native-replay", run)) as archive:
            assert set(archive.namelist()) == {
                "request.json",
                "result.json",
                "trajectory.json",
                "impact-receipt.json",
            }
            archive.extractall(case["evidence"] / "impact")
    finally:
        session.close()
    _phase(case, "authentic_impact_closed")
    return run


def _post(client: Any, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/tools/golf-simulator" + endpoint, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _api_path(case: dict[str, Any], run: str) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import golf_research, golf_simulator

    app = FastAPI()
    app.include_router(golf_simulator.router)
    app.include_router(golf_research.router)
    app.dependency_overrides[golf_research.get_library] = lambda: case["library"]
    app.dependency_overrides[golf_research.require_local_client] = lambda: None
    golf_simulator.reset_simulator_state()
    try:
        with TestClient(app) as client:
            _post(
                client,
                "/session",
                {"destination_id": "local", "session_id": "api-local"},
            )
            prepare = _post(
                client,
                "/shot/prepare-research-impact",
                {
                    "replay_id": "native-replay",
                    "run_id": run,
                    "shot_id": "api-shot",
                    "session_id": "api-local",
                    "created_at_utc": "2026-10-04T12:00:00Z",
                    "aim_context": {
                        "source_to_target_rotation": np.eye(3).tolist(),
                        "revision": 1,
                    },
                    "context_revision": 1,
                },
            )
            arm = _post(
                client,
                "/shot/arm",
                {
                    "prepared_shot_id": prepare["prepared_shot_id"],
                    "context_revision": 1,
                },
            )
            delivery = _post(
                client,
                "/shot/submit",
                {
                    "prepared_shot_id": prepare["prepared_shot_id"],
                    "arm_token": arm["arm_token"],
                },
            )
            assert delivery["state"] == "confirmed_accepted"
            response = client.get(
                "/tools/golf-simulator/shot/api-shot/local-trajectory"
            )
            assert response.status_code == 200, response.text
            record = response.json()
            assert record["research"] == prepare["research"]
            assert record["research"]["qualification"] == {
                "contact": "unverified",
                "numerical": "unverified",
                "scientific": "unverified",
            }
            assert "BallFlightSimulator" in record["provenance"]
            assert len(record["samples"]) > 2
            _api_retained(record, golf_simulator.get_current_session_service())
            _save(
                case["evidence"] / "api.json",
                {
                    "prepare": prepare,
                    "arm": arm,
                    "delivery": delivery,
                    "trajectory": record,
                },
            )
    finally:
        golf_simulator.reset_simulator_state()
    _phase(case, "api_local_delivery_and_recall")


def _api_retained(record: dict[str, Any], service: Any) -> None:
    retained = service.get_local_trajectory_record("api-shot")
    for channel, attribute in (
        ("position_m", "position"),
        ("velocity_mps", "velocity"),
    ):
        np.testing.assert_array_equal(
            [s[channel] for s in record["samples"]],
            [getattr(p, attribute) for p in retained.points],
        )


def _click(app: Any, dialog: Any, name: str) -> None:
    button = getattr(dialog, name)
    assert button.isEnabled(), dialog.status.text()
    button.click()
    _wait(app, lambda: dialog._worker is None)


def _table(dialog: Any, record: Any) -> list[list[str]]:
    expected = [
        [repr(float(v)) for v in (p.time, *p.position, *p.velocity)]
        for p in record.points
    ]
    actual = [
        [dialog.samples.item(r, c).text() for c in range(7)]
        for r in range(dialog.samples.rowCount())
    ]
    assert actual == expected
    return actual


def _qt_path(case: dict[str, Any], app: Any, admitted: Any) -> None:
    from src.tools.golf_simulator.research_dialog import ResearchGolfDialog

    dialog = ResearchGolfDialog(None, admitted)
    dialog.show()
    try:
        for button in (
            "connect_button",
            "prepare_button",
            "arm_button",
            "submit_button",
        ):
            _click(app, dialog, button)
            _phase(case, "qt_" + button)
        assert dialog._submitted, dialog.status.text()
        record = dialog.service.get_local_trajectory_record(admitted.shot.shot_id)
        assert "BallFlightSimulator" in record.provenance
        before = _table(dialog, record)
        assert dialog.grab().save(str(case["evidence"] / "research-submitted.png"))
        _click(app, dialog, "recall_button")
        recalled = dialog.service.get_local_trajectory_record(admitted.shot.shot_id)
        assert recalled.simulated_at_utc == record.simulated_at_utc
        assert recalled.provenance == record.provenance
        assert _table(dialog, recalled) == before
        assert dialog.grab().save(str(case["evidence"] / "research-recalled.png"))
        _save(
            case["evidence"] / "qt.json",
            {
                "context": admitted.to_record(),
                "table": before,
                "provenance": record.provenance,
                "status": dialog.status.text(),
                "synthetic_ui_acceptance": True,
                "gl_pixel_accuracy_claim": False,
            },
        )
        _phase(case, "qt_recalled_same_retained_samples")
    finally:
        dialog.cleanup()
        _wait(app, lambda: not dialog._shutdown_worker.is_running())
        assert dialog._shutdown_worker.error is None
        dialog.close()


@pytest.mark.timeout(120)
def test_native_impact_reaches_api_and_qt_local_research(
    replay_case: Any, tmp_path: Path
) -> None:
    """Actual native synthetic replay and Rust flight; no scientific acceptance."""
    from PyQt6.QtWidgets import QApplication
    from src.shared.python import workspace as owner
    from src.shared.python.physics.rust_kernel import is_rust_available

    pytest.importorskip("upstream_physics")
    assert is_rust_available(), (
        "This acceptance requires the real installed Rust kernel"
    )
    evidence = Path(
        os.environ.get(
            "NECROMATCHER_NATIVE_GOLF_EVIDENCE_DIRECTORY", str(tmp_path / "evidence")
        )
    )
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    _prepare_offscreen_font(app, evidence)
    case = {
        "library": replay_case[0],
        "evidence": evidence,
        "started": time.monotonic(),
        "timeline": {},
    }
    declaration, _ = _recorded_case(case["library"], tmp_path)
    _phase(case, "actual_recorded_replay")
    run = _impact_bundle(case, declaration)
    admitted = owner.load_research_impact_shot(
        case["library"], "native-replay", run, _metadata("qt-shot", "qt-local")
    )
    saved = json.loads((evidence / "impact" / "result.json").read_bytes())
    np.testing.assert_array_equal(
        admitted.shot.ball_velocity_m_s, saved["impact_state"]["ball_velocity"]
    )
    np.testing.assert_array_equal(
        admitted.shot.ball_angular_velocity_rad_s,
        saved["impact_state"]["ball_angular_velocity"],
    )
    _api_path(case, run)
    _qt_path(case, app, admitted)
    _phase(case, "all_paths_closed")
