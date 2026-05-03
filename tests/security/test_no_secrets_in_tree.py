"""Secret-scanning regression test for the repository tree."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_trivy_secret_scan_has_no_findings() -> None:
    """Run Trivy fs secret scanning when the CLI is available."""
    trivy = shutil.which("trivy")
    if trivy is None:
        pytest.skip("trivy CLI is not installed in this environment")

    result = subprocess.run(
        [
            trivy,
            "fs",
            "--scanners",
            "secret",
            "--exit-code",
            "1",
            "--skip-dirs",
            ".git",
            "--skip-dirs",
            ".mypy_cache",
            "--skip-dirs",
            ".pytest_cache",
            ".",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
