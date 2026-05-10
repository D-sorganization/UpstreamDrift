"""Secret-scanning regression test for the repository tree.

This module runs Trivy filesystem secret-scan checks to detect
potential credential leaks in the repository.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.security
def test_no_secrets_in_tree() -> None:
    """Run Trivy filesystem secret scan on the repository tree.

    This test ensures no secrets (API keys, tokens, credentials) are
    accidentally committed to the repository.

    Skips if Trivy is not installed.
    """
    trivy = shutil.which("trivy")
    if trivy is None:
        pytest.skip("Trivy not installed for secret scanning")

    result = subprocess.run(
        [
            trivy,
            "fs",
            "--security-checks",
            "secret",
            "--exit-code",
            "1",
            "--quiet",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Trivy secret scan detected potential secrets:\n{result.stdout}\n{result.stderr}"
    )