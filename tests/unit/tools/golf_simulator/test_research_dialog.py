"""Actual local session controls run in background with unqualified retained data."""

import json
import os
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from src.shared.python.golf_simulator import LocalReferenceAdapter
from src.shared.python.workspace import ResearchImpactShot
from tests.unit.golf_simulator.test_research_session import shot
from src.tools.golf_simulator import research_dialog as owner

pytestmark = pytest.mark.unit


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def admitted():
    from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions
    from src.shared.python.physics.ball_properties import BallProperties
    from src.shared.python.physics.impact_model.types import ImpactParameters

    original = shot()
    context = {
        "source_to_target_rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "framepolicy": {
            "source_frame_id": "flight_xfwd_yleft_zup",
            "target_frame_id": "target_local_xyz",
            "operation": "proper_rotation_vectors",
        },
        "recorded_time_s": original.impact_time_s,
        "recorded_sample_index": 2,
        "replay_clock_policy": "authored_simulation_seconds",
        "scientific_qualified": False,
        "physical_source_time_qualified": False,
        "qualification": {
            "contact": "unverified",
            "numerical": "unverified",
            "scientific": "unverified",
        },
        "parents": {
            name + suffix: name if suffix == "_id" else "sha256:" + "b" * 64
            for name in ("replay", "profile", "fit", "model", "capture")
            for suffix in ("_id", "_hash")
        },
        "assumptions": {
            "geometry": "Authored point",
            "selection": "Authored sample",
            "environment": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in asdict(EnvironmentalConditions()).items()
            },
            "ball_assumptions": asdict(BallProperties()),
            "impact_params": asdict(ImpactParameters()),
            "impact_method": "RIGID_BODY",
            "flight_settings": {"dt_s": 0.01, "max_time_s": 10.0},
        },
    }
    return ResearchImpactShot(
        original,
        "replay",
        "a" * 32,
        *(["sha256:" + "b" * 64] * 3),
        json.dumps(context, sort_keys=True, separators=(",", ":")),
    )


def drain(dialog, app):
    deadline = time.monotonic() + 3
    while dialog._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        dialog._poll()
        time.sleep(0.005)
    assert dialog._worker is None


def install_local(monkeypatch, simulate=None):
    simulator = MagicMock()
    simulator.simulate_trajectory.return_value = [
        SimpleNamespace(time=0.0, position=[1.0, 2.0, 3.0], velocity=[4.0, 5.0, 6.0])
    ]
    if simulate:
        simulator.simulate_trajectory.side_effect = simulate
    adapter = LocalReferenceAdapter(simulator)
    monkeypatch.setattr(owner, "LocalReferenceAdapter", lambda: adapter)
    return adapter


def test_real_service_lifecycle_and_exact_retained_table(app, monkeypatch):
    adapter = install_local(monkeypatch)
    dialog = owner.ResearchGolfDialog(None, admitted())
    assert dialog.connect_button.isEnabled() and not dialog.prepare_button.isEnabled()
    dialog.connect_button.click()
    drain(dialog, app)
    dialog.prepare_button.click()
    drain(dialog, app)
    dialog.arm_button.click()
    drain(dialog, app)
    assert dialog.submit_button.isEnabled()
    dialog.submit_button.click()
    drain(dialog, app)
    assert "confirmed_accepted" in dialog.status.text()
    assert "Scientific" in dialog.boundary.text()
    assert (
        not dialog.submit_button.isEnabled() and not dialog.prepare_button.isEnabled()
    )
    assert dialog.samples.rowCount() == 1
    assert dialog.samples.item(0, 1).text() == "1.0"
    assert dialog.samples.item(0, 4).text() == "4.0"
    assert adapter.get_last_trajectory("research-shot") is not None
    dialog.recall_button.click()
    drain(dialog, app)
    assert dialog.samples.rowCount() == 1
    dialog.cleanup()


def test_public_disarm_cancel_and_no_demo_submit(app, monkeypatch):
    adapter = install_local(monkeypatch)
    dialog = owner.ResearchGolfDialog(None, admitted())
    dialog.connect_button.click()
    drain(dialog, app)
    dialog.prepare_button.click()
    drain(dialog, app)
    dialog.arm_button.click()
    drain(dialog, app)
    dialog.disarm_button.click()
    drain(dialog, app)
    assert dialog.arm_button.isEnabled() and not dialog.submit_button.isEnabled()
    dialog.cancel_button.click()
    drain(dialog, app)
    assert dialog.prepare_button.isEnabled()
    assert adapter.get_last_trajectory("research-shot") is None
    dialog.cleanup()


def test_slow_flight_keeps_qt_responsive_and_late_close_hidden(app, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def simulate(launch):
        entered.set()
        assert release.wait(2)
        return [
            SimpleNamespace(
                time=0.0, position=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0]
            )
        ]

    install_local(monkeypatch, simulate)
    dialog = owner.ResearchGolfDialog(None, admitted())
    for button in (dialog.connect_button, dialog.prepare_button, dialog.arm_button):
        button.click()
        drain(dialog, app)
    dialog.submit_button.click()
    assert entered.wait(1)
    heartbeats = []
    QTimer.singleShot(0, lambda: heartbeats.append(True))
    app.processEvents()
    assert heartbeats == [True]
    assert not dialog.cancel_button.isEnabled()
    dialog.cleanup()
    release.set()
    dialog._shutdown_worker.wait()
    dialog._poll()
    assert dialog.samples.rowCount() == 0


def test_reject_foreign_untyped_admission(app):
    with pytest.raises(TypeError):
        owner.ResearchGolfDialog(None, {"shot": shot()})


def test_failed_flight_never_offers_blind_resubmission(app, monkeypatch):
    def fail(launch):
        raise RuntimeError("Local flight unavailable")

    install_local(monkeypatch, fail)
    dialog = owner.ResearchGolfDialog(None, admitted())
    for button in (dialog.connect_button, dialog.prepare_button, dialog.arm_button):
        button.click()
        drain(dialog, app)
    dialog.submit_button.click()
    drain(dialog, app)
    assert "Local flight unavailable" in dialog.status.text()
    assert not dialog.prepare_button.isEnabled()
    assert not dialog.submit_button.isEnabled()
    assert not dialog.recall_button.isEnabled()
    dialog.cleanup()


def test_foreign_record_rejected_before_sample_publication(app):
    dialog = owner.ResearchGolfDialog(None, admitted())
    with pytest.raises(ValueError, match="another shot"):
        dialog._show_samples(SimpleNamespace(shot_id="foreign", points=[]))
    assert dialog.samples.rowCount() == 0
    dialog.cleanup()


def test_nonidentity_admission_never_labeled_identity(app):
    from src.shared.python.golf_simulator import AimContext

    original = admitted()
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    context = json.loads(original.context_json)
    context["source_to_target_rotation"] = [list(row) for row in rotation]
    transformed = replace(
        original,
        shot=replace(original.shot, aim_context=AimContext(rotation)),
        context_json=json.dumps(context, sort_keys=True, separators=(",", ":")),
    )
    dialog = owner.ResearchGolfDialog(None, transformed)
    assert "identity" not in dialog.boundary.text().lower()
    assert "declared source-to-target rotation" in dialog.boundary.text().lower()
    assert dialog.admitted.shot.aim_context.source_to_target_rotation == rotation
    dialog.cleanup()


def test_replay_button_requires_verified_complete_result(app, tmp_path):
    from src.tools.necromatcher.replay_impact_dialog import ReplayImpactDialog
    from tests.unit.tools.test_necromatcher_replay_impact_dialog import Session, view

    dialog = ReplayImpactDialog("replay", Session(), library_root=tmp_path)
    drain(dialog, app)
    assert not dialog.open_golf.isEnabled()
    dialog.run = view()
    dialog._controls()
    assert dialog.open_golf.isEnabled()
    dialog.run = view("running")
    dialog._controls()
    assert not dialog.open_golf.isEnabled()
    with pytest.raises(ValueError, match="another replay/run"):
        dialog._show_golf(SimpleNamespace(replay_id="foreign", run_id="a" * 32))
    dialog.cleanup()


def test_close_retires_armed_public_service_and_token(app, monkeypatch):
    import asyncio
    from src.shared.python.golf_simulator import SessionState

    install_local(monkeypatch)
    dialog = owner.ResearchGolfDialog(None, admitted())
    for button in (dialog.connect_button, dialog.prepare_button, dialog.arm_button):
        button.click()
        drain(dialog, app)
    prepared_id, token = dialog._prepared_id, dialog._token
    dialog.cleanup()
    assert dialog._token is None
    dialog._shutdown_worker.wait()
    assert dialog.service.current_state == SessionState.IDLE
    with pytest.raises(RuntimeError, match="not armed"):
        asyncio.run(dialog.service.submit_at_impact(prepared_id, token))


def test_close_during_prepare_retires_late_prepared_identity(app, monkeypatch):
    from src.shared.python.golf_simulator import GolfSessionService, SessionState

    entered, release = threading.Event(), threading.Event()
    original = GolfSessionService.prepare_research_shot

    def delayed(service, envelope):
        entered.set()
        assert release.wait(2)
        return original(service, envelope)

    monkeypatch.setattr(GolfSessionService, "prepare_research_shot", delayed)
    install_local(monkeypatch)
    dialog = owner.ResearchGolfDialog(None, admitted())
    dialog.connect_button.click()
    drain(dialog, app)
    dialog.prepare_button.click()
    assert entered.wait(1)
    dialog.cleanup()
    release.set()
    dialog._shutdown_worker.wait()
    assert dialog.service.current_state == SessionState.IDLE


def test_close_during_connect_disconnects_late_actual_adapter(app, monkeypatch):
    import asyncio

    entered, release = threading.Event(), threading.Event()
    original = LocalReferenceAdapter.connect

    async def delayed(adapter, config):
        entered.set()
        assert release.wait(2)
        return await original(adapter, config)

    monkeypatch.setattr(LocalReferenceAdapter, "connect", delayed)
    adapter = install_local(monkeypatch)
    dialog = owner.ResearchGolfDialog(None, admitted())
    dialog.connect_button.click()
    assert entered.wait(1)
    dialog.cleanup()
    release.set()
    dialog._shutdown_worker.wait()
    result = asyncio.run(adapter.submit(admitted().shot))
    assert result.detail == "Adapter not connected"
