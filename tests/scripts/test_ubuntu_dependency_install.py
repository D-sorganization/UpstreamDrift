"""Exercise the CI installer with real Bash and a harmless APT command double."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
InstallerFixture = tuple[list[str], dict[str, str], Path]

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts/ci/install_ubuntu_dependencies.sh"
SOURCE = """Types: deb
URIs: http://archive.ubuntu.com/ubuntu/
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"""


def _shell_path(path: Path) -> str:
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if path.drive else value


@pytest.fixture
def installer(tmp_path: Path) -> InstallerFixture:
    """Fake only privileged commands; execute installer logic and files normally."""
    bash = (
        str(
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "Git/bin/bash.exe"
        )
        if os.name == "nt"
        else shutil.which("bash")
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is required for the Ubuntu installer contract")
    binary = tmp_path / "bin"
    binary.mkdir()
    (tmp_path / "ubuntu.sources").write_text(SOURCE, newline="\n")
    stub = binary / "sudo"
    stub.write_text(
        """#!/usr/bin/env bash
set -eu
if [ "$1" != apt-get ]; then exec "$@"; fi
shift
printf '%s\\n' "$*" >> "$APT_LOG"
for arg in "$@"; do
  case "$arg" in
    Dir::Etc::sourceparts=*) sources="${arg#*=}" ;;
    Dir::State::lists=*) lists="${arg#*=}" ;;
  esac
done
test -d "$lists"
test "$(find "$sources" -type f | wc -l)" -eq 1
cp "$sources/ubuntu.sources" "$APT_PROOF"
case " $* " in
  *' update '*)
    if [ "${FAIL_UPDATE:-0}" = 1 ]; then exit 100; fi
    if [ "${FAIL_ONCE:-0}" = 1 ] && [ ! -f "$APT_PROOF.failed" ]; then
      touch "$APT_PROOF.failed"
      exit 100
    fi ;;
  *' install '*) if [ "${FAIL_INSTALL:-0}" = 1 ]; then exit 100; fi ;;
esac
""",
        newline="\n",
    )
    stub.chmod(0o755)
    sleep = binary / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
    sleep.chmod(0o755)
    env = {
        **os.environ,
        "FIXTURE_BIN": _shell_path(binary),
        "UBUNTU_APT_SOURCE_FILE": _shell_path(tmp_path / "ubuntu.sources"),
        "TMPDIR": _shell_path(tmp_path),
        "APT_LOG": _shell_path(tmp_path / "apt.log"),
        "APT_PROOF": _shell_path(tmp_path / "source-copy"),
        "INSTALLER": _shell_path(INSTALLER),
    }
    command = [bash, "-c", 'export PATH="$FIXTURE_BIN:$PATH"; bash "$INSTALLER"']
    return command, env, tmp_path


def test_isolates_sources_and_indexes_preserving_signatures(
    installer: InstallerFixture,
) -> None:
    command, env, root = installer
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    calls = (root / "apt.log").read_text().splitlines()
    assert len(calls) == 2
    for call in calls:
        assert "Dir::Etc::sourcelist=/dev/null" in call
        assert "Dir::Etc::sourceparts=" in call
        assert "Dir::State::lists=" in call
        assert "DPkg::Lock::Timeout=300" in call
        assert "APT::Update::Error-Mode=any" in call
        assert "allow-unauthenticated" not in call
        assert "trusted=yes" not in call
    assert calls[0].endswith("update")
    assert "install -y libegl1 libgl1 xvfb" in calls[1]
    assert (root / "source-copy").read_text() == SOURCE
    assert (root / "ubuntu.sources").read_text() == SOURCE
    assert not list(root.glob("upstream-apt.*"))


@pytest.mark.parametrize("failure", ["FAIL_UPDATE", "FAIL_INSTALL"])
def test_persistent_apt_failure_is_not_ignored(
    installer: InstallerFixture, failure: str
) -> None:
    command, env, root = installer
    env[failure] = "1"
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 100
    calls = (root / "apt.log").read_text().splitlines()
    if failure == "FAIL_UPDATE":
        assert all(call.endswith("update") for call in calls)
    assert not list(root.glob("upstream-apt.*"))


def test_transient_failure_retries_then_installs(installer: InstallerFixture) -> None:
    command, env, root = installer
    env["FAIL_ONCE"] = "1"
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    calls = (root / "apt.log").read_text().splitlines()
    assert len(calls) == 3
    assert calls[0] == calls[1]
    assert "install -y" in calls[2]


@pytest.mark.parametrize("source", [None, SOURCE.split("Signed-By:")[0]])
def test_missing_or_unsigned_source_fails_before_apt(
    installer: InstallerFixture, source: str | None
) -> None:
    command, env, root = installer
    source_file = root / "ubuntu.sources"
    if source is None:
        source_file.unlink()
    else:
        source_file.write_text(source, newline="\n")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert "signed Ubuntu" in result.stderr
    assert not (root / "apt.log").exists()
