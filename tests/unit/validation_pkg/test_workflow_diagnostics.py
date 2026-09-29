"""Tests for src.shared.python.validation_pkg.workflow_diagnostics (Issues #1949, #1744)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.validation_pkg.workflow_diagnostics import (
    WorkflowDiagnosticContext,
)


class TestWorkflowDiagnosticContextBasic:
    def test_workflow_diagnostics_construction(self, tmp_path: Path) -> None:
        ctx = WorkflowDiagnosticContext(str(tmp_path), "test_workflow")
        assert ctx.workflow_name == "test_workflow"
        assert ctx.dump_dir == tmp_path

    def test_context_manager_enters_returns_self(self, tmp_path: Path) -> None:
        ctx = WorkflowDiagnosticContext(str(tmp_path), "test")
        with ctx as c:
            assert c is ctx

    def test_record_state_stores_value(self, tmp_path: Path) -> None:
        ctx = WorkflowDiagnosticContext(str(tmp_path), "test")
        ctx.record_state("step1", {"value": 42})
        assert ctx.states["step1"] == {"value": 42}

    def test_record_multiple_states(self, tmp_path: Path) -> None:
        ctx = WorkflowDiagnosticContext(str(tmp_path), "test")
        ctx.record_state("a", 1)
        ctx.record_state("b", "hello")
        ctx.record_state("c", [1, 2, 3])
        assert len(ctx.states) == 3

    def test_no_exception_no_dump(self, tmp_path: Path) -> None:
        with WorkflowDiagnosticContext(str(tmp_path), "clean") as ctx:
            ctx.record_state("step", "data")
        # No subdirectory should be created on clean exit
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 0

    def test_start_time_is_set(self, tmp_path: Path) -> None:
        ctx = WorkflowDiagnosticContext(str(tmp_path), "test")
        assert ctx.start_time is not None


class TestWorkflowDiagnosticContextOnFailure:
    def test_exception_is_reraised(self, tmp_path: Path) -> None:
        with (
            pytest.raises(ValueError, match="test error"),
            WorkflowDiagnosticContext(str(tmp_path), "failing"),
        ):
            raise ValueError("test error")

    def test_dump_directory_created_on_exception(self, tmp_path: Path) -> None:
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "failing") as ctx:
                ctx.record_state("step1", {"x": 1})
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) >= 1

    def test_diagnostics_json_written_on_exception(self, tmp_path: Path) -> None:
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "failing") as ctx:
                ctx.record_state("step1", {"x": 1})
                raise RuntimeError("test failure")
        except RuntimeError:
            pass
        # Find the diagnostics file
        json_files = list(tmp_path.glob("**/diagnostics.json"))
        assert len(json_files) == 1

    def test_diagnostics_json_valid_format(self, tmp_path: Path) -> None:
        state_data = {"input": [1, 2, 3], "output": 42}
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "my_workflow") as ctx:
                ctx.record_state("compute", state_data)
                raise KeyError("missing key")
        except KeyError:
            pass
        json_files = list(tmp_path.glob("**/diagnostics.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["workflow_name"] == "my_workflow"
        assert data["exception_type"] == "KeyError"
        assert "traceback" in data
        assert "states" in data

    def test_diagnostics_contains_recorded_states(self, tmp_path: Path) -> None:
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "test") as ctx:
                ctx.record_state("phase1", "phase1_data")
                ctx.record_state("phase2", 99)
                raise Exception("fail")
        except Exception as e:  # noqa: BLE001, F841
            pass
        json_files = list(tmp_path.glob("**/diagnostics.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "phase1" in data["states"]
        assert "phase2" in data["states"]

    def test_workflow_name_in_dump_directory(self, tmp_path: Path) -> None:
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "special_wf"):
                raise ValueError("boom")
        except ValueError:
            pass
        subdirs = list(tmp_path.iterdir())
        assert any("special_wf" in str(d) for d in subdirs)

    def test_duration_seconds_in_diagnostics(self, tmp_path: Path) -> None:
        try:
            with WorkflowDiagnosticContext(str(tmp_path), "test"):
                raise Exception("x")
        except Exception as e:  # noqa: BLE001, F841
            pass
        json_files = list(tmp_path.glob("**/diagnostics.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "duration_seconds" in data
        assert data["duration_seconds"] >= 0.0
