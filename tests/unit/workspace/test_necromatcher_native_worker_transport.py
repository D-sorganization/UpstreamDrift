"""Only reviewed research operations can select a native subprocess."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_unknown_operation_rejected_before_process_creation(
    monkeypatch, tmp_path
) -> None:
    from src.shared.python.workspace import necromatcher_native_worker as transport

    started = []
    monkeypatch.setattr(
        transport, "secure_popen", lambda *args, **kwargs: started.append(args)
    )
    with pytest.raises(ValueError, match="operation"):
        transport.execute_native_research_worker(
            tmp_path / "request.json", 1, lambda: False, operation="other"
        )
    assert started == []


def test_legacy_refit_delegates_without_changing_three_argument_boundary(
    monkeypatch,
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    calls = []
    expected = {"fit": {"qualification": "monocular_research_hypothesis"}}

    def execute(path, budget, cancelled):
        calls.append((path, budget, cancelled()))
        return expected

    monkeypatch.setattr(jobs, "execute_native_research_worker", execute)
    assert jobs._execute_worker(Path("request.json"), 1, lambda: False) is expected
    assert calls == [(Path("request.json"), 1, False)]


def test_impact_selects_fixed_worker_and_closes_owned_streams(monkeypatch, tmp_path):
    from io import StringIO
    from src.shared.python.workspace import necromatcher_native_worker as transport

    calls = []

    class Process:
        returncode = 0
        stdout = StringIO()
        stderr = StringIO()

        def communicate(self, timeout):
            return '{"artifact_hashes":{}}', ""

        def poll(self):
            return 0

    process = Process()
    monkeypatch.setattr(
        transport,
        "secure_popen",
        lambda command, **kw: calls.append(command) or process,
    )
    assert transport.execute_native_research_worker(
        tmp_path / "request.json", 1, lambda: False, operation="impact"
    ) == {"artifact_hashes": {}}
    assert calls[0][-2:] == [
        "src.shared.python.workspace.necromatcher_impact_worker",
        str(tmp_path / "request.json"),
    ]
    assert process.stdout.closed and process.stderr.closed
