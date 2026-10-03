"""Only the two reviewed research operations can select a native subprocess."""

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
