"""
Stage 2 hygiene test: after the upstream_drift_tools -> sidekick rename,
no src/ or tests/ file (except this one) should import from upstream_drift_tools.

Issue: #5619
"""
import pathlib
import subprocess


REPO_ROOT = pathlib.Path(__file__).parents[3]


def test_no_upstream_drift_tools_imports_in_src():
    """After Stage 2, no src/ file should import from upstream_drift_tools."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "-l", "upstream_drift_tools", "src/"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    files = [f for f in result.stdout.strip().splitlines() if f]
    assert files == [], f"Found upstream_drift_tools imports in src/: {files}"


def test_no_upstream_drift_tools_imports_in_tests():
    """After Stage 2, no tests/ file (except this hygiene test) should import from upstream_drift_tools."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "-l", "upstream_drift_tools", "tests/"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    files = [
        f
        for f in result.stdout.strip().splitlines()
        if f and "test_no_deprecated_imports" not in f
    ]
    assert files == [], f"Found upstream_drift_tools imports in tests/: {files}"
