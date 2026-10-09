"""Run the actual Pinocchio marker test outside repository pytest conftests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    tools_root = repo_root / "vendor" / "ud-tools"
    test_source = repo_root / "tests" / "unit" / "engines" / "test_feedback_native_markers.py"
    output_root = (
        Path.home() / ".codex-feedback-pinocchio-11900" / "f09e-marker-output"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    isolated_test = output_root / "test_feedback_native_markers.py"
    shutil.copyfile(test_source, isolated_test)

    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "PYTHONPATH": os.pathsep.join(
            (str(repo_root), str(tools_root / "src"), str(tools_root / "src" / "shared" / "python"))
        ),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "UPSTREAMDRIFT_REPO_ROOT": str(repo_root),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(isolated_test),
            "-q",
            "-c",
            "/dev/null",
            "-k",
            "pinocchio_markers",
            f"--junitxml={output_root / 'results.xml'}",
        ],
        cwd=output_root,
        env=environment,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    summary = {
        "returncode": result.returncode,
        "python": sys.executable,
        "source_root": str(repo_root),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (output_root / "run.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
