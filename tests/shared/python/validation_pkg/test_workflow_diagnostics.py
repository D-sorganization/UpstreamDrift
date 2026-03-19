import json
from pathlib import Path

from src.shared.python.validation_pkg.workflow_diagnostics import (
    WorkflowDiagnosticContext,
)


def test_workflow_diagnostic_context_success(tmp_path: Path) -> None:
    """Test that a successful workflow does not dump detailed diagnostics."""
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()

    with WorkflowDiagnosticContext(
        dump_dir=str(dump_dir), workflow_name="test_success"
    ) as ctx:
        ctx.record_state("step1", {"data": 42})
        ctx.record_state("step2", {"result": "ok"})
        # No exception raised

    # Check that no dump was created because it succeeded
    dumps = list(dump_dir.glob("*test_success*"))
    assert len(dumps) == 0


def test_workflow_diagnostic_context_failure(tmp_path: Path) -> None:
    """Test that a failing workflow dumps all recorded states to disk."""
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()

    raised = False
    try:
        with WorkflowDiagnosticContext(
            dump_dir=str(dump_dir), workflow_name="test_failure"
        ) as ctx:
            ctx.record_state("step_A", {"status": "started"})
            ctx.record_state("step_B", {"value": 100})
            raise RuntimeError("Intentional workflow failure")
    except RuntimeError:
        raised = True

    assert raised

    # Check that dump was created
    dumps = list(dump_dir.glob("*test_failure*"))
    assert len(dumps) == 1

    # Verify contents
    with open(dumps[0] / "diagnostics.json") as f:
        data = json.load(f)

    assert data["workflow_name"] == "test_failure"
    assert data["exception_type"] == "RuntimeError"
    assert "Intentional workflow failure" in data["exception_msg"]
    assert data["states"]["step_A"]["status"] == "started"
    assert data["states"]["step_B"]["value"] == 100
