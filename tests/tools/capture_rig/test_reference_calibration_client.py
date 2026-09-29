"""Calibration requests use the established process lifecycle without blocking Qt."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

from src.tools.capture_rig.reference_calibration.client import ReferenceWorkerClient

pytestmark = [pytest.mark.ui, pytest.mark.integration]


def test_catalog_leaves_the_ui_responsive_and_returns_reference_sizes(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    client = ReferenceWorkerClient(host)
    ticks = []
    timer = QTimer(host)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(10)
    with qtbot.waitSignal(client.completed, timeout=20000) as response:
        client.request({"action": "catalog"})
        assert client.busy
    assert len(ticks) > 1
    assert not client.busy
    assert len(response.args[0]["targets"]) == 4


def test_missing_runtime_reports_recovery_and_releases_the_request(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    client = ReferenceWorkerClient(host, executable="missing-calibration-python.exe")
    with qtbot.waitSignal(client.failed, timeout=5000) as failure:
        client.request({"action": "catalog"})
    assert not client.busy
    assert "runtime" in failure.args[0].lower()


def test_cancelled_operation_never_applies_a_result(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    client = ReferenceWorkerClient(host)
    completed = []
    client.completed.connect(completed.append)
    with qtbot.waitSignal(client.failed, timeout=10000) as failure:
        client.request({"action": "catalog"})
        client.cancel()
    assert "stopped" in failure.args[0].lower()
    assert not client.busy
    assert completed == []
