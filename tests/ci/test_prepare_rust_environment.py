"""Behavioral contracts for job-owned Rust installation state (#11977)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.ci.prepare_rust_environment import main, prepare_rust_environment

pytestmark = pytest.mark.unit


def test_partial_toolchain_is_preserved_and_never_selected(tmp_path: Path) -> None:
    partial = tmp_path / "rustup" / "toolchains" / "stable"
    partial.mkdir(parents=True)
    marker = partial / "interrupted-install"
    marker.write_bytes(b"retain diagnostic state")
    environment_file = tmp_path / "github-env"
    environment_file.write_text("EXISTING=value", encoding="utf-8")

    bindings = prepare_rust_environment(tmp_path, environment_file)

    rustup_home = Path(bindings["RUSTUP_HOME"])
    cargo_home = Path(bindings["CARGO_HOME"])
    assert rustup_home.is_dir() and cargo_home.is_dir()
    assert rustup_home != cargo_home
    assert not list(rustup_home.iterdir())
    assert not list(cargo_home.iterdir())
    assert rustup_home.is_relative_to(tmp_path.resolve())
    assert cargo_home.is_relative_to(tmp_path.resolve())
    assert marker.read_bytes() == b"retain diagnostic state"
    lines = environment_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "EXISTING=value"
    assert f"RUSTUP_HOME={rustup_home}" in lines
    assert f"CARGO_HOME={cargo_home}" in lines


def test_repeated_and_concurrent_jobs_receive_distinct_state(tmp_path: Path) -> None:
    def prepare(index: int) -> dict[str, str]:
        return prepare_rust_environment(tmp_path, tmp_path / f"env-{index}")

    first = prepare(0)
    retained = Path(first["RUSTUP_HOME"]) / "installed-toolchain"
    retained.write_bytes(b"owned by the first job")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(prepare, range(1, 4)))

    results.append(first)
    paths = [value for result in results for value in result.values()]
    assert len(set(paths)) == 8
    assert retained.read_bytes() == b"owned by the first job"
    assert all(Path(value).is_dir() for value in paths)


@pytest.mark.parametrize("target", ["temporary-root", "environment-file"])
def test_environment_line_injection_is_rejected_before_writes(
    tmp_path: Path, target: str
) -> None:
    temp_root = tmp_path
    environment_file = tmp_path / "github-env"
    if target == "temporary-root":
        temp_root = tmp_path / "unsafe\nOTHER=value"
    else:
        environment_file = tmp_path / "unsafe\nOTHER=value"

    with pytest.raises(ValueError, match="single-line"):
        prepare_rust_environment(temp_root, environment_file)

    assert not list(tmp_path.iterdir())


def test_missing_temp_root_does_not_publish_environment(tmp_path: Path) -> None:
    environment_file = tmp_path / "github-env"
    with pytest.raises(FileNotFoundError):
        prepare_rust_environment(tmp_path / "missing", environment_file)
    assert not environment_file.exists()


def test_file_temp_root_does_not_publish_environment(tmp_path: Path) -> None:
    temp_root = tmp_path / "file"
    temp_root.write_bytes(b"preserve")
    environment_file = tmp_path / "github-env"
    with pytest.raises(ValueError, match="existing directory"):
        prepare_rust_environment(temp_root, environment_file)
    assert not environment_file.exists()
    assert temp_root.read_bytes() == b"preserve"


def test_unwritable_environment_target_creates_no_job_state(tmp_path: Path) -> None:
    environment_file = tmp_path / "directory-not-file"
    environment_file.mkdir()
    with pytest.raises(OSError):
        prepare_rust_environment(tmp_path, environment_file)
    assert list(tmp_path.iterdir()) == [environment_file]


@pytest.mark.parametrize("name", ["RUNNER_TEMP", "GITHUB_ENV"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_cli_rejects_missing_or_empty_required_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str | None
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "github-env"))
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    assert main() == 1
    assert not list(tmp_path.iterdir())


def test_cli_ignores_inherited_toolchain_homes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_file = tmp_path / "github-env"
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_ENV", str(environment_file))
    monkeypatch.setenv("RUSTUP_HOME", str(tmp_path / "inherited-rustup"))
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "inherited-cargo"))

    assert main() == 0
    assert not (tmp_path / "inherited-rustup").exists()
    assert not (tmp_path / "inherited-cargo").exists()
    assert "RUSTUP_HOME=" in environment_file.read_text(encoding="utf-8")
