"""Every Swatinem/rust-cache step must set ``cache-bin: "false"``.

With ``cache-bin`` left at its default (true), rust-cache records each regular
file in ``$CARGO_HOME/bin`` during its main step and deletes those files in its
post step, so only binaries installed by the job get cached. On persistent
self-hosted runners a job without a per-job ``CARGO_HOME`` resolves it to the
shared ``~/.cargo``; the post step then deletes the host's ``rustup`` binary
(the cargo/clippy/rustc proxies are symlinks to it, so they survive dangling)
and every later Rust job on that host fails with "cargo: command not found".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIRS = (
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / ".forgejo" / "workflows",
)


def _workflow_files() -> list[Path]:
    files: list[Path] = []
    for directory in WORKFLOW_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.yml")))
            files.extend(sorted(directory.glob("*.yaml")))
    return [
        f
        for f in files
        if "Swatinem/rust-cache" in f.read_text(encoding="utf-8", errors="replace")
    ]


def test_workflows_using_rust_cache_are_discovered() -> None:
    assert _workflow_files(), "expected at least one workflow using Swatinem/rust-cache"


def _rust_cache_steps(workflow: Any) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(workflow, dict):
        return found
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict) and str(step.get("uses", "")).startswith(
                "Swatinem/rust-cache"
            ):
                found.append((str(job_name), step))
    return found


def _offenders(workflow: Any) -> list[str]:
    bad = []
    for job_name, step in _rust_cache_steps(workflow):
        value = str((step.get("with") or {}).get("cache-bin", "true")).lower()
        if value != "false":
            bad.append(f"{job_name}: {step.get('name') or step.get('uses')}")
    return bad


@pytest.mark.parametrize(
    "path", _workflow_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_rust_cache_never_cleans_shared_cargo_bin(path: Path) -> None:
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    offenders = _offenders(workflow)
    assert not offenders, (
        f'{path.relative_to(REPO_ROOT).as_posix()}: rust-cache without cache-bin: "false" deletes the host '
        f"rustup on self-hosted runners: {offenders}"
    )


def test_detector_flags_default_and_true_cache_bin() -> None:
    workflow = {
        "jobs": {
            "a": {"steps": [{"uses": "Swatinem/rust-cache@v2"}]},
            "b": {
                "steps": [
                    {"uses": "Swatinem/rust-cache@v2", "with": {"cache-bin": "true"}}
                ]
            },
            "c": {
                "steps": [
                    {"uses": "Swatinem/rust-cache@v2", "with": {"cache-bin": "false"}}
                ]
            },
        }
    }
    assert _offenders(workflow) == [
        "a: Swatinem/rust-cache@v2",
        "b: Swatinem/rust-cache@v2",
    ]
