"""Tests for the Save-fit JSON widget (issue #4707, slice 3/3)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# MUST set the platform BEFORE any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip(
    "PyQt6",
    reason="PyQt6 required for Save-fit widget tests",
    exc_type=ImportError,
)
pytest.importorskip(
    "PyQt6.QtWidgets",
    reason="PyQt6.QtWidgets not loadable in this environment",
    exc_type=ImportError,
)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.starting_pose_matcher.widgets.save_fit_button import (  # noqa: E402
    FIT_RESULT_SCHEMA_VERSION,
    SaveFitButton,
    compute_source_file_sha256,
    serialize_fit_result,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _fake_result(**overrides):
    """Plain object exposing CanonicalFitResult-shaped attributes."""

    class R:
        theta_optimal = [0.1, 0.2, 0.3]
        final_cost = 1.5
        final_rmse_m = 0.005
        solver_status = "success"
        iterations = 12
        n_evaluations = 47
        wall_clock_s = 2.5
        message = "converged"
        history = ()
        method = "trust-constr"
        git_commit = "deadbeef"
        engine_version = "1.2.3"
        target_hash = "abc123"
        timestamp_utc = "2026-05-09T00:00:00Z"
        meta = {}

    r = R()
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# serialize_fit_result — pure function tests
# ---------------------------------------------------------------------------


def test_serialize_fit_result_full_shape():
    doc = serialize_fit_result(_fake_result(), engine_name="pinocchio")
    assert doc["schema_version"] == FIT_RESULT_SCHEMA_VERSION
    assert doc["engine"] == {
        "name": "pinocchio",
        "version": "1.2.3",
        "method": "trust-constr",
    }
    assert doc["fit"]["theta_optimal"] == [0.1, 0.2, 0.3]
    assert doc["fit"]["solver_status"] == "success"
    assert doc["fit"]["iterations"] == 12
    assert doc["fit"]["n_evaluations"] == 47
    assert doc["fit"]["final_cost"] == pytest.approx(1.5)
    assert doc["fit"]["final_rmse_m"] == pytest.approx(0.005)
    assert doc["provenance"]["target_hash"] == "abc123"
    assert doc["provenance"]["timestamp_utc"] == "2026-05-09T00:00:00Z"
    assert doc["source"] == {"path": None, "sha256": None}


def test_serialize_fit_result_includes_source_sha256(tmp_path: Path):
    payload = b"hello-mocap"
    src = tmp_path / "swing.c3d"
    src.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    doc = serialize_fit_result(_fake_result(), engine_name="drake", source_file=src)
    assert doc["source"]["sha256"] == expected
    assert doc["source"]["path"] == str(src)


def test_serialize_fit_result_residuals_from_meta():
    res = _fake_result()
    res.meta = {"residuals": [0.01, 0.02, 0.03]}
    doc = serialize_fit_result(res, engine_name="opensim")
    assert doc["fit"]["residuals"] == [0.01, 0.02, 0.03]


def test_serialize_fit_result_explicit_residuals_override_meta():
    res = _fake_result()
    res.meta = {"residuals": [9.0, 9.0]}
    doc = serialize_fit_result(res, engine_name="opensim", residuals=[1.0, 2.0])
    assert doc["fit"]["residuals"] == [1.0, 2.0]


def test_serialize_fit_result_dict_input():
    doc = serialize_fit_result(
        {"theta_optimal": [0.0, 1.0], "engine_version": "0.0.1"},
        engine_name="fake",
    )
    assert doc["fit"]["theta_optimal"] == [0.0, 1.0]
    assert doc["engine"]["version"] == "0.0.1"


def test_serialize_fit_result_rejects_none():
    with pytest.raises(ValueError, match="result"):
        serialize_fit_result(None, engine_name="x")


def test_serialize_fit_result_rejects_empty_engine_name():
    with pytest.raises(ValueError, match="engine_name"):
        serialize_fit_result(_fake_result(), engine_name="")


def test_compute_source_file_sha256_missing():
    with pytest.raises(FileNotFoundError):
        compute_source_file_sha256("/nonexistent/path/to/file.c3d")


def test_compute_source_file_sha256_none():
    assert compute_source_file_sha256(None) is None
    assert compute_source_file_sha256("") is None


# ---------------------------------------------------------------------------
# SaveFitButton — Qt widget tests
# ---------------------------------------------------------------------------


def test_button_disabled_until_result_and_engine(_qapp):
    w = SaveFitButton()
    assert w.btn_save.isEnabled() is False
    w.set_result(_fake_result(), engine_name="")
    assert w.btn_save.isEnabled() is False
    w.set_engine_name("pinocchio")
    assert w.btn_save.isEnabled() is True


def test_save_to_path_writes_valid_json(_qapp, tmp_path: Path):
    w = SaveFitButton()
    w.set_engine_name("drake")
    w.set_result(_fake_result(), engine_name="drake")
    out = tmp_path / "out" / "fit.json"

    saved_paths: list[str] = []
    w.saved.connect(saved_paths.append)

    written = w.save_to_path(out)
    assert written == out
    assert out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["engine"]["name"] == "drake"
    assert doc["fit"]["theta_optimal"] == [0.1, 0.2, 0.3]
    assert saved_paths == [str(out)]
    assert "Saved fit" in w.lbl_status.text()


def test_save_to_path_requires_result(_qapp, tmp_path: Path):
    w = SaveFitButton()
    with pytest.raises(ValueError, match="no fit result"):
        w.save_to_path(tmp_path / "x.json")


def test_save_to_path_requires_engine(_qapp, tmp_path: Path):
    w = SaveFitButton()
    w.set_result(_fake_result(), engine_name="")
    with pytest.raises(ValueError, match="no engine"):
        w.save_to_path(tmp_path / "x.json")


def test_full_smoke_run_then_save(_qapp, tmp_path: Path):
    """Smoke test mirroring the GUI button -> fit -> save path."""
    from src.shared.python.motion_matching import provider_registry
    from src.tools.starting_pose_matcher.widgets.run_fit_button import (
        RunFitButton,
    )

    class _Provider:
        engine_name = "smoke-engine"

        def fit_swing(self, target):  # noqa: ARG002
            return _fake_result()

    provider_registry.clear_registry()
    try:
        provider_registry.register_provider(_Provider())
        run_w = RunFitButton()
        save_w = SaveFitButton()
        run_w.set_inputs(target=object(), engine_name="smoke-engine")

        # Simulate the GUI wiring: when run_fit finishes, push the result
        # plus engine name into the save widget.
        def _on_fit_finished(result):
            save_w.set_result(result, engine_name="smoke-engine")

        run_w.finished.connect(_on_fit_finished)
        run_w.start_fit()

        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(20)
        deadline = {"left": 3000}

        def _tick():
            if save_w.btn_save.isEnabled() or deadline["left"] <= 0:
                timer.stop()
                loop.quit()
            deadline["left"] -= timer.interval()

        timer.timeout.connect(_tick)
        timer.start()
        loop.exec()

        assert save_w.btn_save.isEnabled() is True
        out = save_w.save_to_path(tmp_path / "smoke_fit.json")
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["engine"]["name"] == "smoke-engine"
        assert doc["fit"]["theta_optimal"] == [0.1, 0.2, 0.3]
    finally:
        provider_registry.clear_registry()
