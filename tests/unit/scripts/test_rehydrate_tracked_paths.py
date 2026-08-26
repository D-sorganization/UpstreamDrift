from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.ci.rehydrate_tracked_paths import (
    RUNTIME_DOCKER_CONTEXT_PATHS,
    RehydrationError,
    rehydrate_tracked_paths,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Rehydration Contract Test")
    for relative_path, content in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_rehydrates_missing_and_modified_tracked_inputs(tmp_path: Path) -> None:
    repo, head = _init_repo(
        tmp_path,
        {
            "Dockerfile": "FROM scratch\n",
            ".dockerignore": ".git\n",
            "src/package.py": "VALUE = 1\n",
        },
    )
    (repo / "Dockerfile").unlink()
    (repo / "src" / "package.py").write_text("VALUE = 999\n", encoding="utf-8")

    restored = rehydrate_tracked_paths(
        repo,
        expected_head=head,
        paths=("Dockerfile", ".dockerignore", "src"),
    )

    assert restored == ("Dockerfile", ".dockerignore", "src")
    assert (repo / "Dockerfile").read_text(encoding="utf-8") == "FROM scratch\n"
    assert (repo / "src" / "package.py").read_text(encoding="utf-8") == ("VALUE = 1\n")
    assert _git(repo, "diff", "--name-only", "HEAD", "--", *restored) == ""


def test_preflight_is_atomic_when_requested_path_is_not_tracked(
    tmp_path: Path,
) -> None:
    repo, head = _init_repo(tmp_path, {"Dockerfile": "FROM scratch\n"})
    (repo / "Dockerfile").unlink()

    with pytest.raises(RehydrationError, match="not tracked at HEAD: missing.txt"):
        rehydrate_tracked_paths(
            repo,
            expected_head=head,
            paths=("Dockerfile", "missing.txt"),
        )

    assert not (repo / "Dockerfile").exists()


def test_rejects_head_mismatch_before_restoring(tmp_path: Path) -> None:
    repo, _head = _init_repo(tmp_path, {"Dockerfile": "FROM scratch\n"})
    (repo / "Dockerfile").unlink()

    with pytest.raises(RehydrationError, match="does not match expected"):
        rehydrate_tracked_paths(
            repo,
            expected_head="0" * 40,
            paths=("Dockerfile",),
        )

    assert not (repo / "Dockerfile").exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ("", ".", "../Dockerfile", "/Dockerfile", r"src\package.py", ":(glob)*"),
)
def test_rejects_unbounded_or_magic_pathspecs(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    repo, head = _init_repo(tmp_path, {"Dockerfile": "FROM scratch\n"})

    with pytest.raises((TypeError, ValueError), match="relative POSIX path"):
        rehydrate_tracked_paths(
            repo,
            expected_head=head,
            paths=(unsafe_path,),
        )


def test_runtime_profile_covers_every_production_docker_copy_source() -> None:
    assert RUNTIME_DOCKER_CONTEXT_PATHS == (
        "Dockerfile",
        ".dockerignore",
        "Cargo.toml",
        "rust_core",
        "requirements.lock",
        "scripts/config/pip_audit_waivers.json",
        "scripts/ci/check_pip_audit_waivers.py",
        "src",
        "pyproject.toml",
        "launch_golf_suite.py",
        "scripts/ci/start_api_server.py",
        ".env.example",
        "docker/entrypoint.sh",
    )


def test_workflow_rehydrates_immediately_before_runtime_build() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/docker-size-gates.yml").read_text(
        encoding="utf-8"
    )
    prepare = workflow.index("Prepare local BuildKit cache (runtime)")
    rehydrate = workflow.index("Rehydrate and verify runtime build context")
    build = workflow.index("Build Docker Image (runtime)")
    bounded = workflow[rehydrate:build]

    assert prepare < rehydrate < build
    assert "git rev-parse HEAD" in bounded
    assert "git cat-file -e HEAD:scripts/ci/rehydrate_tracked_paths.py" in bounded
    assert "git --literal-pathspecs checkout --force HEAD --" in bounded
    assert '--expected-head "${{ github.sha }}"' in bounded
    assert "--profile runtime" in bounded
    assert "target: runtime" in workflow[build:]
    assert "tags: upstream-drift:runtime" in workflow[build:]
