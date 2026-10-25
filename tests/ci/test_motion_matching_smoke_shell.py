"""Exercise the motion-matching smoke launcher without a real GL or X server."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/run_motion_matching_smoke.sh"

FAKE_PYTHON = """\
#!/bin/bash
set -eu
if [[ "$1" == "-c" ]]; then
  case "${MUJOCO_GL:-}" in
    egl) [[ "${FAKE_EGL:-0}" == 1 ]] ;;
    osmesa) [[ "${FAKE_OSMESA:-0}" == 1 ]] ;;
    *) exit 1 ;;
  esac
  exit $?
fi
printf '%s\\0' "$@" > "$SMOKE_ARGV_FILE"
printf '%s\\n%s\\n' "${MUJOCO_GL:-}" "${DISPLAY:-}" > "$SMOKE_ENV_FILE"
exit "${FAKE_PYTEST_EXIT:-0}"
"""


def _bash_path(path: Path) -> str:
    """Return a path understood by Git Bash on Windows and Bash elsewhere."""
    if os.name == "nt":
        return f"/{path.drive[0].lower()}{path.as_posix()[2:]}"
    return str(path)


def _bash_executable() -> str | None:
    """Prefer Git Bash on Windows; system32 bash can launch a WSL distro."""
    if os.name == "nt":
        git_bash = (
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "Git/bin/bash.exe"
        )
        if git_bash.is_file():
            return str(git_bash)
        return None
    return shutil.which("bash")


FAKE_XVFB_RUN = """\
#!/bin/bash
set -eu
printf '%s\\0' "$@" > "$SMOKE_XVFB_ARGV_FILE"
if [[ "${1:-}" == "-a" ]]; then
  shift
fi
export DISPLAY=:987
exec "$@"
"""


@pytest.mark.parametrize(
    (
        "probe_env",
        "caller_env",
        "expected_backend",
        "xvfb_available",
        "uses_xvfb",
        "pytest_exit",
    ),
    [
        ({"FAKE_EGL": "1"}, {}, "egl", False, False, 0),
        (
            {"FAKE_EGL": "0", "FAKE_OSMESA": "1"},
            {},
            "osmesa",
            False,
            False,
            0,
        ),
        (
            {"FAKE_EGL": "0", "FAKE_OSMESA": "0"},
            {},
            "",
            True,
            True,
            0,
        ),
        ({"FAKE_EGL": "0", "FAKE_OSMESA": "0"}, {}, "", False, False, 0),
        ({}, {"DISPLAY": ":42"}, "", False, False, 0),
        ({}, {"MUJOCO_GL": "osmesa"}, "osmesa", False, False, 0),
        ({"FAKE_EGL": "1"}, {}, "egl", False, False, 7),
        ({"FAKE_EGL": "0", "FAKE_OSMESA": "0"}, {}, "", True, True, 9),
    ],
    ids=(
        "egl",
        "osmesa",
        "xvfb-fallback",
        "no-backend-no-xvfb",
        "inherited-display",
        "caller-backend",
        "pytest-failure-direct",
        "pytest-failure-xvfb",
    ),
)
def test_smoke_script_routes_display_and_disables_pytest_xvfb(
    tmp_path: Path,
    probe_env: dict[str, str],
    caller_env: dict[str, str],
    expected_backend: str,
    xvfb_available: bool,
    uses_xvfb: bool,
    pytest_exit: int,
) -> None:
    """Keep launcher-owned display routes while preventing plugin startup."""
    bash = _bash_executable()
    assert bash is not None, "Git Bash or Bash is required to exercise the launcher"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dirname = bin_dir / "dirname"
    fake_dirname.write_text(
        "#!/bin/bash\nset -eu\nprintf '%s\\n' \"${1%/*}\"\n",
        encoding="utf-8",
    )
    fake_dirname.chmod(0o755)

    fake_python = bin_dir / "python"
    fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
    fake_python.chmod(0o755)

    xvfb_argv_file = tmp_path / "xvfb-argv.bin"
    if xvfb_available:
        fake_xvfb = bin_dir / "xvfb-run"
        fake_xvfb.write_text(FAKE_XVFB_RUN, encoding="utf-8")
        fake_xvfb.chmod(0o755)

    argv_file = tmp_path / "pytest-argv.bin"
    env_file = tmp_path / "pytest-env.txt"
    env = os.environ.copy()
    env.update(
        {
            "DISPLAY": ":parent-display-must-not-leak",
            "MUJOCO_GL": "parent-backend-must-not-leak",
            "FAKE_EGL": "parent-probe-must-not-leak",
            "FAKE_OSMESA": "parent-probe-must-not-leak",
            "FAKE_PYTEST_EXIT": "88",
            "PATH": os.environ.get("PATH", ""),
        }
    )
    for name in (
        "DISPLAY",
        "MUJOCO_GL",
        "FAKE_EGL",
        "FAKE_OSMESA",
        "FAKE_PYTEST_EXIT",
    ):
        env.pop(name, None)
    env.update(probe_env)
    env.update(caller_env)
    env.update(
        {
            "PATH": _bash_path(bin_dir) if os.name == "nt" else str(bin_dir),
            "SMOKE_ARGV_FILE": _bash_path(argv_file),
            "SMOKE_ENV_FILE": _bash_path(env_file),
            "SMOKE_XVFB_ARGV_FILE": _bash_path(xvfb_argv_file),
            "FAKE_PYTEST_EXIT": str(pytest_exit),
        }
    )

    forwarded = [
        "--junitxml=motion-matching.xml",
        "--maxfail=2",
        "--custom-value=two words",
    ]
    completed = subprocess.run(
        [bash, _bash_path(SCRIPT), *forwarded],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == pytest_exit, completed.stderr

    pytest_argv = argv_file.read_bytes().decode().rstrip("\0").split("\0")
    pytest_env = env_file.read_text(encoding="utf-8").splitlines()
    plugin_options = [
        pytest_argv[index + 1]
        for index, argument in enumerate(pytest_argv[:-1])
        if argument == "-p"
    ]
    assert "no:xvfb" in plugin_options
    assert "pytest_timeout" in plugin_options
    assert pytest_argv[:3] == ["-m", "pytest", "tests/integration/motion_matching"]
    assert "slow and not live_simulation" in pytest_argv
    assert "--timeout=0" in pytest_argv
    assert pytest_argv[-len(forwarded) :] == forwarded
    assert pytest_env[0] == expected_backend

    if uses_xvfb:
        xvfb_argv = xvfb_argv_file.read_bytes().decode().rstrip("\0").split("\0")
        assert xvfb_argv[:2] == ["-a", "python"]
        assert pytest_env[1] == ":987"
    elif not xvfb_available and not caller_env and not expected_backend:
        assert "no headless GL; playback GIFs will be skipped" in completed.stderr
        assert pytest_env[1] == ""
    elif caller_env.get("DISPLAY"):
        assert pytest_env[1] == caller_env["DISPLAY"]
    else:
        assert pytest_env[1] == ""
