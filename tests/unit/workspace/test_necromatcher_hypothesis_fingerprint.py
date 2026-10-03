"""New clean-worker implementation bytes belong to the one canonical fingerprint."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "filename",
    [
        "necromatcher_native_worker.py",
        "necromatcher_hypothesis_worker.py",
        "necromatcher_hypothesis.py",
        "necromatcher_hypothesis_contracts.py",
        "necromatcher_capture_identity.py",
        "necromatcher_fit_metrics.py",
        "__init__.py",
    ],
)
def test_hypothesis_transport_mutation_changes_canonical_stamp(
    tmp_path, monkeypatch, filename: str
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    root = Path(__file__).resolve().parents[3]
    relative = Path("src/shared/python/workspace") / filename
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes((root / relative).read_bytes())
    monkeypatch.setattr(jobs, "get_repo_root", lambda: tmp_path)
    before = jobs.fit_execution_stamp()
    assert relative.as_posix() in before["source_files"]
    destination.write_bytes(destination.read_bytes() + b"\n# mutation fixture\n")
    assert jobs.fit_execution_stamp()["source_sha256"] != before["source_sha256"]
