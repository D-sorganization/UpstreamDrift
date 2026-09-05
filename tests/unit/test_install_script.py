from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "install.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def _run_install_script(
    tmp_path: Path, *, include_pipx: bool, include_checkout: bool
) -> tuple[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"

    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [[ "$1" == "-c" ]]; then
  echo "3.11"
  exit 0
fi
printf 'python3:%s\\n' "$*" >> "{calls}"
exit 0
""",
    )

    if include_pipx:
        _write_executable(
            bin_dir / "pipx",
            f"""#!/bin/bash
printf 'pipx:%s\\n' "$*" >> "{calls}"
exit 0
""",
        )

    cwd = tmp_path / "checkout" if include_checkout else tmp_path / "work"
    cwd.mkdir()
    if include_checkout:
        (cwd / "pyproject.toml").write_text(
            "[project]\nname = 'upstream-drift'\nversion = '0.0.0'\n",
            encoding="utf-8",
        )

    bash_bin = "bash"
    script_path = str(INSTALL_SCRIPT)
    env = os.environ.copy()
    if sys.platform == "win32":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            bash_bin = str(git_bash)
            script_path = INSTALL_SCRIPT.resolve().as_posix()
        env["PATH"] = f"{bin_dir};{env.get('PATH', '')}"
    else:
        env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = subprocess.run(
        [bash_bin, script_path],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    return calls.read_text(encoding="utf-8"), result.stdout


def test_install_script_defaults_to_remote_repo(tmp_path: Path) -> None:
    calls, stdout = _run_install_script(
        tmp_path, include_pipx=True, include_checkout=False
    )

    assert "Using remote install source" in stdout
    assert (
        "pipx:install git+https://github.com/D-sorganization/UpstreamDrift.git" in calls
    )


def test_install_script_uses_local_checkout_when_present(tmp_path: Path) -> None:
    calls, stdout = _run_install_script(
        tmp_path, include_pipx=False, include_checkout=True
    )

    assert "Using local checkout" in stdout
    assert "python3:-m pip install --user ." in calls
