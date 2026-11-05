from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "install_trivy.sh"
ARCHIVE = "trivy_0.70.0_Linux-64bit.tar.gz"


def _release_fixture(tmp_path: Path, *, valid_checksum: bool = True) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    binary = tmp_path / "trivy"
    binary.write_text("#!/bin/sh\necho fixture\n", encoding="utf-8")
    with tarfile.open(release / ARCHIVE, "w:gz") as archive:
        archive.add(binary, arcname="trivy")

    digest = hashlib.sha256((release / ARCHIVE).read_bytes()).hexdigest()
    if not valid_checksum:
        digest = "0" * 64
    (release / "trivy_0.70.0_checksums.txt").write_text(
        f"{digest}  {ARCHIVE}\n", encoding="utf-8"
    )
    return release


def _run_installer(
    tmp_path: Path, *, valid_checksum: bool = True
) -> subprocess.CompletedProcess[str]:
    release = _release_fixture(tmp_path, valid_checksum=valid_checksum)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        "  if [ \"$1\" = '-o' ] || [ \"$1\" = '--output' ]; then output=$2; shift 2; else url=$1; shift; fi\n"
        "done\n"
        f'cp \'{release}/\'"$(basename "$url")" "$output"\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    github_path = tmp_path / "github-path"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_PATH": str(github_path),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
@pytest.mark.skipif(
    sys.platform != "linux", reason="installer targets Linux CI runners"
)
def test_installer_verifies_release_and_exports_trivy_on_path(tmp_path: Path) -> None:
    result = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    install_dir = tmp_path / "runner-temp" / "trivy-bin"
    assert (install_dir / "trivy").read_text(encoding="utf-8").startswith("#!/bin/sh")
    assert (tmp_path / "github-path").read_text(encoding="utf-8").strip() == str(
        install_dir
    )


@pytest.mark.unit
@pytest.mark.skipif(
    sys.platform != "linux", reason="installer targets Linux CI runners"
)
def test_installer_rejects_release_with_invalid_checksum(tmp_path: Path) -> None:
    result = _run_installer(tmp_path, valid_checksum=False)

    assert result.returncode != 0
    assert "checksum" in result.stderr.lower() or "failed" in result.stderr.lower()
    assert not (tmp_path / "github-path").exists()
