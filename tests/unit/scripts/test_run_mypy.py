from __future__ import annotations

import os
from pathlib import Path

from scripts.ci import run_mypy


def test_sanitized_mypy_env_removes_shared_python_roots(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    shared_python = repo_root / "src" / "shared" / "python"
    vendor_shared_python = (
        repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"
    )
    external = tmp_path / "external"
    for path in (shared_python, vendor_shared_python, external):
        path.mkdir(parents=True)

    monkeypatch.chdir(repo_root)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(shared_python), str(external), str(vendor_shared_python)]),
    )

    env = run_mypy._sanitized_mypy_env()

    assert env["PYTHONPATH"] == str(external)
