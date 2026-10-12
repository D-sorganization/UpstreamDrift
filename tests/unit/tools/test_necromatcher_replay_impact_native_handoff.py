"""Opt-in desktop action → real replay/clean worker/Rust research handoff.

Only file-choice UI is substituted. Native extraction, dynamics, impact, flight,
execution stamps, job transport, storage and artifact verification remain real.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import numpy as np
import pytest

from tests.unit.workspace import conftest as workspace_fixtures
from tests.unit.workspace import test_necromatcher_replay as replay_fixtures

pytest.importorskip("PyQt6")
pytestmark = [pytest.mark.live_simulation, pytest.mark.integration]
fit_case = workspace_fixtures.fit_case
native_fit_case = workspace_fixtures.native_fit_case
replay_case = replay_fixtures.replay_case


def _wait(app: Any, condition: Callable[[], bool], deadline: float = 90.0) -> None:
    end = time.monotonic() + deadline
    while not condition():
        app.processEvents()
        if time.monotonic() >= end:
            pytest.fail(
                "Desktop research action did not close inside its test deadline"
            )
        time.sleep(0.005)
    app.processEvents()


def _recorded_case(library: Any, tmp_path: Path) -> tuple[Path, Any]:
    from src.shared.python import workspace as owner
    from src.shared.python.simulation_backends.trace_io import write_trace
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    trace = owner.replay_authored_profile(
        library,
        "replay-profile",
        owner.ReplayOptions(0, (20.0,) + (0.0,) * 43, 0.004, 0.001),
    )
    path = tmp_path / "recorded.h5"
    write_trace(trace, path)
    library.add_replay("native-replay", "practice", path)
    binding = load_native_fit_binding(library, "replay-fit")
    body = "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector"
    rotation, _ = binding.plant.frame_poses({"head": (body, (0, 0, 0))}, trace.q[0])[
        body
    ]
    geometry = owner.ReplayImpactGeometry(
        body,
        (0, 0, 0),
        rotation.T @ np.array([1, 0, 0]),
        rotation.T @ np.array([0, 0, 1]),
        0.2,
        0.005,
        "Synthetic body origin/face and effective mass/MOI; no historical anatomy",
    )
    selection = owner.ReplayImpactSelection(
        0, np.eye(3), (0, 0, 0), "Authored recorded sample; no detected contact"
    )
    expected = owner.extract_replay_impact_state(
        library, "native-replay", geometry, selection
    )
    declaration = tmp_path / "declaration.json"
    declaration.write_text(
        json.dumps(
            {"geometry": geometry.to_record(), "selection": selection.to_record()}
        ),
        encoding="utf-8",
    )
    return declaration, expected


def _instrument_session(
    session: Any, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, int]]
) -> None:
    """Observe actual public calls; delegate unchanged, with no service substitutes."""
    for name in ("submit", "view", "download"):
        original = getattr(session, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            calls.append((_name, threading.get_ident()))
            return _original(*args, **kwargs)

        monkeypatch.setattr(session, name, observed)


def _close_dialog(app: Any, dialog: Any) -> None:
    dialog.cleanup()
    _wait(app, lambda: not dialog._shutdown_worker.is_running())
    assert dialog._shutdown_worker.error is None


def _check_public_viewer(
    app: Any, dialog: Any, extracted: Path, evidence: Path
) -> None:
    """Check retained values/provenance, without claiming OpenGL pixel accuracy."""
    from src.shared.python.physics.flight_trajectory_export import VELOCITY_CHANNEL

    record = json.loads((extracted / "trajectory.json").read_bytes())
    assert dialog.open_tracer.isEnabled()
    dialog.open_tracer.click()
    _wait(app, lambda: getattr(dialog, "tracer_widget", None) is not None)
    curves = dialog.tracer_widget.imported_trajectories
    assert len(curves) == 1
    curve = next(iter(curves.values()))
    np.testing.assert_array_equal(
        curve.positions, [sample["position_m"] for sample in record["samples"]]
    )
    assert not curve.positions.flags.writeable
    assert curve.source_id == record["source_id"]
    assert curve.frame_id == record["frame_id"]
    assert curve.model_family == record["provenance"]["model_family"]
    assert curve.model_name == record["provenance"]["model_name"]
    assert np.all(np.diff([sample["time_s"] for sample in record["samples"]]) > 0)
    assert all(len(sample[VELOCITY_CHANNEL]) == 3 for sample in record["samples"])
    assert dialog.tracer_widget.grab().save(str(evidence / "shot-tracer.png"))
    dialog.tracer_widget.parentWidget().close()


def _check_saved_bundle(saved: Path, expected: Any, evidence: Path) -> Path:
    from src.shared.python import workspace as owner

    extracted = saved.parent / "retained"
    names = {"trajectory.json", "impact-receipt.json", "result.json", "request.json"}
    with ZipFile(saved) as archive:
        assert set(archive.namelist()) == names
        archive.extractall(extracted)
    for name in names:
        (evidence / name).write_bytes((extracted / name).read_bytes())
    recalled = owner.load_replay_impact_receipt(
        extracted / "impact-receipt.json", extracted / "trajectory.json"
    )
    for field in (
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "clubhead_orientation",
    ):
        np.testing.assert_array_equal(
            getattr(recalled, field), getattr(expected, field)
        )
    result = json.loads((extracted / "result.json").read_bytes())
    assert result["scientific_qualified"] is False
    assert result["physical_source_time_qualified"] is False
    assert result["impact_state"]["ball_velocity"][0] > 0
    return extracted


def _prepare_offscreen_font(app: Any, evidence: Path) -> None:
    """Bootstrap an existing OS font only when offscreen Qt has no font database."""
    from PyQt6.QtGui import QFont, QFontDatabase

    before = {"families": QFontDatabase.families(), "font": app.font().toString()}
    record = {"before": before, "bootstrap": None}
    if not before["families"]:
        font_path = Path(os.environ["WINDIR"]) / "Fonts" / "segoeui.ttf"
        if not font_path.is_file():
            pytest.skip(
                "Offscreen Qt has no fonts and existing Windows Segoe UI is absent"
            )
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        assert font_id >= 0
        families = QFontDatabase.applicationFontFamilies(font_id)
        assert families
        app.setFont(QFont(families[0], 10))
        record["bootstrap"] = {
            "path": str(font_path),
            "sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
            "font_id": font_id,
            "families": families,
            "installed_by_test": False,
        }
    record["after"] = {
        "families": QFontDatabase.families(),
        "font": app.font().toString(),
    }
    (evidence / "font-environment.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def _submit_stage(app: Any, case: dict, monkeypatch: pytest.MonkeyPatch) -> str:
    from PyQt6.QtWidgets import QFileDialog
    from src.shared.python import workspace as owner
    from src.tools.necromatcher.replay_impact_dialog import ReplayImpactDialog

    session = owner.NativeImpactSession(case["library"])
    _instrument_session(session, monkeypatch, case["calls"])
    dialog = ReplayImpactDialog(
        "native-replay", session, library_root=case["library"].root
    )
    dialog.show()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **kw: (str(case["saved"]), "")
    )
    try:
        _wait(app, lambda: dialog._worker is None)
        case["phase"]("first_dialog_source_loaded")
        dialog.load_declaration(case["declaration"])
        dialog.budget.setText("60")
        _wait(app, lambda: dialog.start.isEnabled())
        case["phase"]("declaration_loaded")
        dialog.start.click()
        _wait(
            app,
            lambda: (
                bool(dialog.run) and dialog.run["status"] not in {"pending", "running"}
            ),
        )
        case["phase"]("native_job_terminal")
        assert dialog.run["status"] == "succeeded", dialog.status.text()
        assert dialog.run["acceptance"] == "rejected"
        assert dialog.save.isEnabled()
        run_id = dialog.run["run_id"]
        dialog.save.click()
        _wait(app, lambda: case["saved"].is_file() and dialog._worker is None)
        case["phase"]("bundle_saved")
        assert dialog.grab().save(str(case["evidence"] / "completed-dialog.png"))
        assert {name for name, _ in case["calls"]} >= {"submit", "view", "download"}
        assert all(thread != case["main_thread"] for _, thread in case["calls"])
        return run_id
    finally:
        _close_dialog(app, dialog)
        case["phase"]("first_session_closed")


def _recall_stage(app: Any, case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.shared.python import workspace as owner
    from src.tools.necromatcher.replay_impact_dialog import ReplayImpactDialog

    reopened = owner.NativeImpactSession(case["library"])
    _instrument_session(reopened, monkeypatch, case["calls"])
    dialog = ReplayImpactDialog(
        "native-replay", reopened, library_root=case["library"].root
    )
    dialog.show()
    try:
        _wait(app, lambda: dialog._worker is None)
        case["phase"]("second_dialog_source_loaded")
        dialog.saved_run.setText(case["run_id"])
        dialog.recall.click()
        _wait(app, lambda: bool(dialog.run))
        case["phase"]("saved_run_recalled")
        assert dialog.run["run_id"] == case["run_id"]
        assert dialog.run["status"] == "succeeded"
        assert dialog.save.isEnabled()
        assert dialog.grab().save(str(case["evidence"] / "recalled-dialog.png"))
        _check_public_viewer(app, dialog, case["extracted"], case["evidence"])
        case["phase"]("public_viewer_verified")
        assert all(thread != case["main_thread"] for _, thread in case["calls"])
    finally:
        _close_dialog(app, dialog)
        case["phase"]("second_session_closed")


@pytest.mark.timeout(120)
def test_desktop_actions_reach_real_native_bundle_and_recall(
    replay_case: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("upstream_physics")
    from PyQt6.QtWidgets import QApplication

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library, _, _ = replay_case
    evidence = Path(
        os.environ.get(
            "NECROMATCHER_NATIVE_QT_EVIDENCE_DIRECTORY", str(tmp_path / "ui-evidence")
        )
    )
    evidence.mkdir(exist_ok=False)
    _prepare_offscreen_font(app, evidence)
    started = time.monotonic()
    timeline = []

    def phase(name: str) -> None:
        timeline.append({"phase": name, "elapsed_s": time.monotonic() - started})
        (evidence / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")

    phase("fixture_ready")
    declaration, expected = _recorded_case(library, tmp_path)
    phase("recorded_replay_and_expected_state")
    case = {
        "library": library,
        "declaration": declaration,
        "evidence": evidence,
        "phase": phase,
        "calls": [],
        "main_thread": threading.get_ident(),
        "saved": tmp_path / "checked-impact.zip",
    }
    case["run_id"] = _submit_stage(app, case, monkeypatch)
    case["extracted"] = _check_saved_bundle(case["saved"], expected, evidence)
    phase("bundle_verified")
    _recall_stage(app, case, monkeypatch)
