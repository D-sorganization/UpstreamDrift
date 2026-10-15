"""Generated agent-context views are rendered on demand, never committed (#12544).

The Markdown map and offline browser embed the source fingerprint, so a
committed copy conflicted on every concurrent merge. They are git-ignored;
CI renders them before any step reads them, and ``agent_context check`` keeps
failing on stale boundary reviews. The review records and contracts stay
committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("docs/agent_context/README.md", "docs/agent_context/index.html")
COMMITTED_SOURCES = (
    "docs/agent_context/catalog.json",
    "docs/agent_context/reviews.json",
    "docs/agent_context/navigation.json",
    "docs/agent_context/USAGE.md",
)
RENDER = "agent_context --root . render"
READERS = ("agent_context --root . check", "agent_context --root . evaluate")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _workflow_jobs() -> list[tuple[str, dict[str, Any]]]:
    jobs = []
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (workflow.get("jobs") or {}).items():
            jobs.append((f"{path.name}:{name}", job))
    return jobs


def _reads_views(command: str) -> bool:
    return any(reader in command for reader in READERS) or (
        "tests/agent_context" in command
    )


@pytest.mark.parametrize("view", VIEWS)
def test_generated_view_is_git_ignored(view: str) -> None:
    assert _git("check-ignore", "-q", "--no-index", view).returncode == 0


def test_generated_views_are_not_tracked() -> None:
    assert _git("ls-files", "--", *VIEWS).stdout.strip() == ""


def test_reviews_catalog_and_contracts_stay_committed() -> None:
    tracked = _git("ls-files", "--", *COMMITTED_SOURCES).stdout.split()
    assert sorted(tracked) == sorted(COMMITTED_SOURCES)
    contracts = _git("ls-files", "--", "docs/agent_context/contracts").stdout
    assert contracts.strip(), "boundary contracts must stay committed"


def test_ci_renders_views_before_any_step_reads_them() -> None:
    readers_seen = 0
    for job_id, job in _workflow_jobs():
        commands = [str(step.get("run", "")) for step in job.get("steps") or []]
        readers = [i for i, run in enumerate(commands) if _reads_views(run)]
        if not readers:
            continue
        readers_seen += 1
        renders = [i for i, run in enumerate(commands) if RENDER in run]
        assert renders, f"{job_id} reads agent-context views without rendering"
        assert renders[0] < readers[0], f"{job_id} reads views before rendering"
        rendering = (job.get("steps") or [])[renders[0]]
        assert "if" not in rendering
        assert not rendering.get("continue-on-error", False)
    assert readers_seen, "the agent-context gate must still run in CI"


def test_no_hook_or_make_target_requires_a_committed_copy() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    entries = [
        str(hook.get("entry", ""))
        for repo in config.get("repos", [])
        for hook in repo.get("hooks", [])
    ]
    makefile = ROOT / "Makefile"
    if makefile.is_file():
        entries += makefile.read_text(encoding="utf-8").splitlines()
    for entry in entries:
        if any(reader in entry for reader in READERS):
            assert RENDER in entry, f"requires committed views: {entry!r}"


def test_check_still_fails_on_stale_boundary_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_module = pytest.importorskip("agent_context.service")
    stale = [{"id": "launcher-atlas", "verified": False}]
    monkeypatch.setattr(service_module.reviews, "statuses", lambda _catalog: stale)
    with pytest.raises(service_module.CatalogError, match="stale boundary review"):
        service_module.ContextService(ROOT).check()
