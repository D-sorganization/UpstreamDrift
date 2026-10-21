"""Execute the actual CI aggregate with fixture child-job conclusions."""

from __future__ import annotations

import os
import re
import subprocess
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FULL_TIER_JOBS = frozenset(
    {
        "security-scans",
        "tests",
        "rust-wheel-parity",
        "core-only-install",
        "alembic-postgres-round-trip",
    }
)


@cache
def workflow() -> dict:
    """Read GitHub YAML without YAML 1.1 converting ``on`` to a boolean."""
    return yaml.load(
        (ROOT / ".github/workflows/ci-standard.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def run_aggregate(
    tmp_path: Path,
    *,
    tier: str = "full",
    code: bool = True,
    fork: bool = False,
    docs: bool | None = None,
    publication: bool = False,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute production shell, never a reimplementation of its verdict."""
    jobs = workflow()["jobs"]
    gate = jobs["quality-gate"]
    prerequisite_jobs = {
        "agent-context",
        "pick-runner",
        "changed-paths",
        "always-on-unit-lane",
    }
    conclusions = {}
    for job in gate["needs"]:
        fleet_only = "head.repo.full_name == github.repository" in jobs[job].get(
            "if", ""
        )
        expected_run = job in prerequisite_jobs or (
            code
            and not (fork and fleet_only)
            and not (tier == "fast" and job in FULL_TIER_JOBS)
        )
        conclusions[job] = "success" if expected_run else "skipped"
    docs = not code if docs is None else docs
    conclusions["publication-quality"] = "success" if publication else "skipped"
    conclusions["docs-governance-gates"] = "success" if docs else "skipped"
    conclusions.update(overrides or {})
    env = os.environ.copy()
    for step in gate["steps"]:
        for name, expression in step.get("env", {}).items():
            match = re.fullmatch(r"\$\{\{ needs\.([\w-]+)\.result \}\}", expression)
            if match:
                env[name] = conclusions[match.group(1)]
    env.update(
        CI_TIER=tier,
        CODE_CHANGED=str(code).lower(),
        DOCS_CHANGED=str(docs).lower(),
        PUBLICATION_CHANGED=str(publication).lower(),
        IS_FORK_PR=str(fork).lower(),
        GITHUB_STEP_SUMMARY=str(tmp_path / "summary.md"),
    )
    script = "\n".join(step["run"] for step in gate["steps"] if "run" in step)
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
