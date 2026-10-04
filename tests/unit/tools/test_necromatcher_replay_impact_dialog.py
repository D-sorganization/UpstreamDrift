"""The native replay-impact controls use background canonical services."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PyQt6")

from src.tools.necromatcher.replay_impact_dialog import (
    ReplayImpactDialog,
    checked_impact_view,
)

pytestmark = pytest.mark.unit


def declaration() -> dict[str, Any]:
    return {
        "geometry": {
            "body": "club",
            "local_head_point_m": [0, 0, 0.2],
            "local_face_normal": [1, 0, 0],
            "local_face_up": [0, 0, 1],
            "mass_kg": 0.2,
            "moi_kg_m2": 0.001,
            "assumption_description": "Authored geometry",
        },
        "selection": {
            "recorded_sample_index": 3,
            "world_to_flight_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "world_to_flight_translation_m": [0, 0, 0],
            "selection_description": "Operator-selected sample",
        },
    }


def view(status: str = "succeeded") -> dict[str, Any]:
    return {
        "run_id": "a" * 32,
        "replay_id": "replay",
        "status": status,
        "acceptance": "rejected",
        "message": "Research only",
        "blockers": [],
        "control_available": status == "running",
        "execution_verified": status == "succeeded",
        "download_available": status == "succeeded",
        "scientific_qualified": False,
        "physical_source_time_qualified": False,
        "artifactsummary": {
            "files": [
                "trajectory.json",
                "impact-receipt.json",
                "result.json",
                "request.json",
            ],
            **declaration(),
            "clockpolicy": "authored_simulation_seconds",
            "summary": {
                "carry_m": 150,
                "max_height_m": 20,
                "flight_time_s": 4,
                "landing_angle_deg": 30,
            },
        }
        if status == "succeeded"
        else None,
    }


class Session:
    def __init__(self) -> None:
        meta: dict[str, Any] = {
            "schema": "necromatcher/authored-replay/1",
            "scientific_qualified": False,
            "physical_source_time_qualified": False,
            "independent_replay_executed": True,
            "root_policy": "unactuated",
            "initial_state_policy": "exact_saved_pose_and_authored_rates",
        }
        for parent in ("fit", "model", "profile", "capture"):
            meta[parent + "_id"] = parent
            meta[parent + "_hash"] = "sha256:" + "b" * 64
        self.library = SimpleNamespace(
            load_replay=lambda _: SimpleNamespace(
                t=list(range(21)), dt=0.01, backend="mujoco", meta=meta
            )
        )
        self.calls: list[tuple[str, int]] = []
        self.result = view()
        self.closed = threading.Event()

    def submit(
        self, replay: str, geometry: Any, selection: Any, budget: float
    ) -> dict[str, Any]:
        self.calls.append(("submit", threading.get_ident()))
        assert replay == "replay" and geometry.to_record() == declaration()["geometry"]
        assert selection.to_record() == declaration()["selection"] and budget == 120
        return self.result

    def view(self, replay: str, run: str) -> dict[str, Any]:
        self.calls.append(("view", threading.get_ident()))
        assert replay == "replay" and run == "a" * 32
        return self.result

    def cancel(self, replay: str, run: str) -> dict[str, Any]:
        self.calls.append(("cancel", threading.get_ident()))
        return view("cancelled")

    def close(self) -> None:
        self.closed.set()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def wait(app: Any, predicate: Any) -> None:
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


def prepared(app: Any, tmp_path: Path) -> tuple[ReplayImpactDialog, Session]:
    session = Session()
    dialog = ReplayImpactDialog("replay", session, library_root=tmp_path / "library")
    wait(app, lambda: dialog.sample_count == 21)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()), encoding="utf-8")
    dialog.load_declaration(path)
    wait(app, lambda: dialog.declaration is not None)
    dialog.budget.setText("120")
    return dialog, session


def test_submit_and_saved_recall_use_background_and_canonical_assumptions(
    app: Any, tmp_path: Path
) -> None:
    dialog, session = prepared(app, tmp_path)
    try:
        dialog.start.click()
        wait(app, lambda: dialog.save.isEnabled())
        assert (
            "Sample: 3" in dialog.status.text() and "Carry: 150" in dialog.status.text()
        )
        assert (
            "authored" in dialog.boundary.text().lower()
            and "capture" in dialog.boundary.text()
        )
        dialog.saved_run.setText("a" * 32)
        dialog.recall.click()
        wait(app, lambda: any(name == "view" for name, _ in session.calls))
        assert all(identity != threading.get_ident() for _, identity in session.calls)
    finally:
        dialog.cleanup()
        wait(app, session.closed.is_set)


@pytest.mark.parametrize("fault", ["boolean", "sample", "rotation", "extra"])
def test_malformed_declaration_never_submits(
    app: Any, tmp_path: Path, fault: str
) -> None:
    dialog, session = prepared(app, tmp_path)
    record = declaration()
    if fault == "boolean":
        record["geometry"]["mass_kg"] = True
    if fault == "sample":
        record["selection"]["recorded_sample_index"] = 21
    if fault == "rotation":
        record["selection"]["world_to_flight_rotation"][0][0] = -1
    if fault == "extra":
        record["geometry"]["source_path"] = "wrong"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    try:
        dialog.load_declaration(path)
        wait(app, lambda: dialog._worker is None)
        assert not dialog.start.isEnabled() and not session.calls
    finally:
        dialog.cleanup()


@pytest.mark.parametrize("fault", ["foreign", "qualified", "unverified", "failed"])
def test_invalid_or_failed_results_never_save(
    app: Any, tmp_path: Path, fault: str
) -> None:
    dialog, session = prepared(app, tmp_path)
    if fault == "foreign":
        session.result["replay_id"] = "foreign"
    if fault == "qualified":
        session.result["scientific_qualified"] = True
    if fault == "unverified":
        session.result["execution_verified"] = False
    if fault == "failed":
        session.result = view("failed")
    try:
        dialog.start.click()
        wait(app, lambda: dialog._worker is None)
        assert not dialog.save.isEnabled()
    finally:
        dialog.cleanup()


def test_shutdown_is_responsive_and_discards_late_submit(
    app: Any, tmp_path: Path
) -> None:
    dialog, session = prepared(app, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = session.submit

    def slow(*args: Any) -> dict[str, Any]:
        entered.set()
        release.wait(5)
        return original(*args)

    session.submit = slow  # type: ignore[method-assign]
    dialog.start.click()
    wait(app, entered.is_set)
    began = time.monotonic()
    dialog.cleanup()
    assert time.monotonic() - began < 0.1
    app.processEvents()
    release.set()
    wait(app, session.closed.is_set)
    assert dialog.run is None and not dialog.save.isEnabled()


def test_verified_tracer_action_is_available_without_resimulation(
    app: Any, tmp_path: Path
) -> None:
    dialog, session = prepared(app, tmp_path)
    try:
        assert not dialog.open_tracer.isEnabled()
        dialog.start.click()
        wait(app, lambda: dialog.save.isEnabled())
        assert dialog.open_tracer.isEnabled()
    finally:
        dialog.cleanup()
        wait(app, session.closed.is_set)


def test_recall_fault_revokes_old_verified_actions(app: Any, tmp_path: Path) -> None:
    dialog, session = prepared(app, tmp_path)
    try:
        dialog.start.click()
        wait(app, lambda: dialog.save.isEnabled())

        def fault(*args: Any) -> dict[str, Any]:
            raise ValueError("Changed saved run")

        session.view = fault  # type: ignore[method-assign]
        dialog.saved_run.setText("a" * 32)
        dialog.recall.click()
        wait(app, lambda: dialog._worker is None)
        assert dialog.run is None and not dialog.run_id.text()
        assert not dialog.save.isEnabled() and not dialog.open_tracer.isEnabled()
    finally:
        dialog.cleanup()


def test_orphan_run_has_no_controls_but_allows_different_recall(
    app: Any, tmp_path: Path
) -> None:
    dialog, session = prepared(app, tmp_path)
    session.result = {**view("running"), "control_available": False}
    try:
        dialog.saved_run.setText("a" * 32)
        dialog.recall.click()
        wait(app, lambda: dialog.run is not None)
        assert not dialog.cancel.isEnabled() and not dialog.save.isEnabled()
        assert not dialog.open_tracer.isEnabled() and dialog.recall.isEnabled()
    finally:
        dialog.cleanup()


@pytest.mark.parametrize("fault", ["artifact", "summary", "members", "terminal"])
def test_malformed_view_containers_are_explicit_errors(fault: str) -> None:
    record = view()
    if fault == "artifact":
        record["artifactsummary"] = []
    if fault == "summary":
        record["artifactsummary"]["summary"] = []
    if fault == "members":
        record["artifactsummary"]["files"] = [False]
    if fault == "terminal":
        record["control_available"] = True
    with pytest.raises(ValueError):
        checked_impact_view(record, "replay", 21)


def test_view_snapshot_is_detached_from_caller_record() -> None:
    record = view()
    checked = checked_impact_view(record, "replay", 21)
    record["artifactsummary"]["summary"]["carry_m"] = -100
    assert checked["artifactsummary"]["summary"]["carry_m"] == 150


def test_cancel_uses_background_canonical_owner(app: Any, tmp_path: Path) -> None:
    dialog, session = prepared(app, tmp_path)
    session.result = view("running")
    try:
        dialog.start.click()
        wait(app, lambda: dialog.cancel.isEnabled())
        dialog.cancel.click()
        wait(
            app, lambda: dialog.run is not None and dialog.run["status"] == "cancelled"
        )
        assert any(name == "cancel" for name, _ in session.calls)
        assert all(identity != threading.get_ident() for _, identity in session.calls)
        assert not dialog.save.isEnabled() and not dialog.open_tracer.isEnabled()
    finally:
        dialog.cleanup()


def test_save_never_overwrites_and_keeps_valid_run_after_destination_fault(
    app: Any, tmp_path: Path
) -> None:
    dialog, session = prepared(app, tmp_path)
    archive = tmp_path / "verified.zip"
    archive.write_bytes(b"verified session output")

    def download(*args: Any) -> Path:
        session.calls.append(("download", threading.get_ident()))
        return archive

    session.download = download  # type: ignore[attr-defined]
    destination = tmp_path / "saved.zip"
    destination.write_bytes(b"existing")
    try:
        dialog.start.click()
        wait(app, lambda: dialog.save.isEnabled())
        dialog._save_to(destination)
        wait(app, lambda: dialog._worker is None)
        assert destination.read_bytes() == b"existing" and dialog.save.isEnabled()
        assert all(identity != threading.get_ident() for _, identity in session.calls)
    finally:
        dialog.cleanup()
