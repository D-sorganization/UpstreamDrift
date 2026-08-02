"""Tests to ensure importing state_manager does not have filesystem side effects."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def test_import_has_no_filesystem_side_effects() -> None:
    """Importing the module should not create 'saved_states' folders."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Determine the absolute paths for Tools source root
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        tools_root = os.path.abspath(
            os.path.join(tests_dir, "..", "..", "..", "..", "..")
        )

        shared_python_path = os.path.join(tools_root, "src", "shared", "python")
        python_src_path = os.path.join(tools_root, "src", "python", "src")
        src_path = os.path.join(tools_root, "src")

        pythonpath = os.path.pathsep.join(
            [shared_python_path, python_src_path, src_path]
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = pythonpath

        # Run import command in subprocess to ensure clean context
        cmd = [
            sys.executable,
            "-c",
            (
                "import os; "
                "from sidekick.utils.state_manager import StateManager; "
                "print(os.listdir('.'))"
            ),
        ]

        res = subprocess.run(
            cmd,
            cwd=tmp_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        assert res.returncode == 0
        # The list of files in the directory should be empty (no saved_states)
        expected = "[]"
        actual = res.stdout.strip()
        assert actual == expected, f"Eager folders created: {actual}"
