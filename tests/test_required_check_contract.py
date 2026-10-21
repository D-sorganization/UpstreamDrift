"""Required check must reject failed, cancelled or unexpectedly skipped children."""

from pathlib import Path
import shutil

import pytest

from tests.ci.ci_tier_contract import run_aggregate, workflow

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not shutil.which("bash"), reason="Bash required"),
]
REQUIRED_CHILDREN = tuple(workflow()["jobs"]["quality-gate"]["needs"])


def test_required_context_is_unique_and_always_runs_on_pr_and_merge_group() -> None:
    data = workflow()
    assert {"pull_request", "merge_group"} <= set(data["on"])
    publishers = [
        job for job in data["jobs"].values() if job.get("name") == "quality-gate"
    ]
    assert len(publishers) == 1
    assert publishers[0]["if"] == "always()"


@pytest.mark.parametrize("job", REQUIRED_CHILDREN)
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_full_tier_rejects_any_missing_or_failed_required_child(
    tmp_path: Path, job: str, result: str
) -> None:
    verdict = run_aggregate(
        tmp_path, docs=True, publication=True, overrides={job: result}
    )
    assert verdict.returncode != 0, f"{job}={result} was accepted: {verdict.stdout}"


@pytest.mark.parametrize("tier", ["", "unknown"])
def test_missing_or_invalid_tier_cannot_make_required_check_green(
    tmp_path: Path, tier: str
) -> None:
    verdict = run_aggregate(tmp_path, tier=tier)
    assert verdict.returncode != 0, verdict.stdout


def test_failed_classifier_cannot_be_excused_as_a_docs_skip(tmp_path: Path) -> None:
    verdict = run_aggregate(
        tmp_path, tier="fast", code=False, overrides={"changed-paths": "failure"}
    )
    assert verdict.returncode != 0, verdict.stdout
