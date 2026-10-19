"""The unit-test gate runs as a file-sharded matrix (#13314).

One leg collected and ran ~36k tests in ~31 min against a 35 min job limit, so
ordinary PRs timed out. The gate is now split by a stable hash of each test
file's path; these checks pin that every file lands in exactly one leg, that
the split cannot drift between xdist workers, and that the workflow wires each
matrix leg to its own shard while still feeding ``quality-gate``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci import pytest_unit_gate_shard as unit_gate_shard
from scripts.ci.pytest_unit_gate_shard import parse_shard, shard_of

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
RUN_STEP = "Run Green-Suite Unit Gate"


def _unit_gate_job() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job: dict[str, Any] = workflow["jobs"]["unit-test-gate"]
    return job


def _fake_config(spec: str | None, rootpath: Path) -> Any:
    stash = {unit_gate_shard.SHARD_KEY: parse_shard(spec)} if spec else {}
    return SimpleNamespace(stash=stash, rootpath=rootpath)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "expected"), [("1/1", (1, 1)), ("1/2", (1, 2)), ("2/2", (2, 2))]
)
def test_parse_shard_accepts_index_over_total(
    spec: str, expected: tuple[int, int]
) -> None:
    assert parse_shard(spec) == expected


@pytest.mark.unit
@pytest.mark.parametrize("spec", ["", "2", "0/2", "3/2", "1/0", "a/b", "1/2/3"])
def test_parse_shard_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="K/N"):
        parse_shard(spec)


@pytest.mark.unit
def test_every_test_file_lands_in_exactly_one_shard() -> None:
    files = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
    )
    assert files, "expected repository test files to partition"
    buckets: dict[int, list[str]] = {1: [], 2: []}
    for rel in files:
        buckets[shard_of(rel, 2)].append(rel)
    assert sorted(buckets[1] + buckets[2]) == files
    # A stable hash over thousands of files should split them near evenly;
    # a lopsided split would hand one leg the old timeout back.
    assert min(len(b) for b in buckets.values()) > 0.4 * len(files)


@pytest.mark.unit
def test_shard_of_is_stable_across_processes() -> None:
    # zlib.crc32 is seed-independent, so every xdist worker agrees.
    assert shard_of("tests/ci/test_unit_gate_shard.py", 2) == shard_of(
        "tests/ci/test_unit_gate_shard.py", 2
    )
    assert {shard_of(f"tests/test_{i}.py", 3) for i in range(60)} == {1, 2, 3}


@pytest.mark.unit
def test_hook_ignores_only_test_files_of_other_shards(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_example.py"
    helper = tmp_path / "tests" / "conftest.py"
    for path in (test_file, helper):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    own = shard_of("tests/test_example.py", 2)
    other = 3 - own

    hook = unit_gate_shard.pytest_ignore_collect
    assert hook(test_file, _fake_config(f"{own}/2", tmp_path)) is None
    assert hook(test_file, _fake_config(f"{other}/2", tmp_path)) is True
    assert hook(helper, _fake_config(f"{other}/2", tmp_path)) is None
    assert hook(test_file.parent, _fake_config(f"{other}/2", tmp_path)) is None
    assert hook(test_file, _fake_config(None, tmp_path)) is None


@pytest.mark.unit
def test_workflow_runs_each_matrix_leg_on_its_own_shard() -> None:
    job = _unit_gate_job()
    strategy = job["strategy"]
    shards = strategy["matrix"]["shard"]
    assert strategy["fail-fast"] is False
    assert shards == list(range(1, len(shards) + 1))
    assert len(shards) >= 2

    run = next(step for step in job["steps"] if step.get("name") == RUN_STEP)
    assert "-p scripts.ci.pytest_unit_gate_shard" in run["run"]
    assert f"--unit-gate-shard=${{{{ matrix.shard }}}}/{len(shards)}" in run["run"], (
        "each leg must name its own shard and the matrix size"
    )


_COLLECT_DIR = "tests/helpers"


def _collected_files(shard: str) -> set[str]:
    """Return the test files a real ``--collect-only`` run assigns to *shard*."""
    # Without UNIT_GATE_QUARANTINE the child does not arm the root conftest's
    # hang watchdog, and it inherits nothing from the parent's xdist worker.
    env = {k: v for k, v in os.environ.items() if k != "UNIT_GATE_QUARANTINE"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.ci.pytest_unit_gate_shard",
            f"--unit-gate-shard={shard}",
            _COLLECT_DIR,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    assert proc.returncode in (0, 5), proc.stdout + proc.stderr
    return {line.split("::", 1)[0] for line in proc.stdout.splitlines() if "::" in line}


@pytest.mark.unit
# Two child pytest sessions each import the repository conftest; that is
# legitimately slower than the 60 s per-test hang guard on a loaded runner.
@pytest.mark.timeout(320)
def test_real_collection_partitions_under_the_repository_conftest() -> None:
    # tests/conftest.py's own ignore hook returns False; the shard hook must
    # still win, or every leg would silently run the whole suite.
    every_file = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / _COLLECT_DIR).glob("test_*.py")
    }
    first = _collected_files("1/2")
    second = _collected_files("2/2")
    assert first and second, f"{_COLLECT_DIR} should span both shards"
    assert not first & second
    assert first | second == every_file
    assert {shard_of(rel, 2) for rel in first} == {1}
    assert {shard_of(rel, 2) for rel in second} == {2}
