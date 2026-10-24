"""Prove selected even issues are already fixed on main (night-swarm closure receipts).

Each entry records the landing PR and the pytest nodes that exercise acceptance.
CI re-runs those nodes here so stale open issues cannot diverge from ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)


@dataclass(frozen=True)
class ShippedIssueReceipt:
    issue: int
    merge_pr: int
    evidence_nodes: tuple[str, ...]


SHIPPED_RECEIPTS: tuple[ShippedIssueReceipt, ...] = (
    ShippedIssueReceipt(
        issue=13579,
        merge_pr=13598,
        evidence_nodes=(
            "tests/unit/scripts/test_engine_cache_contract.py::"
            "test_docker_build_workflows_use_shared_buildkit_contract",
            "tests/unit/scripts/test_engine_cache_contract.py::"
            "test_no_workflow_uses_per_runner_type_local_or_gha_cache",
        ),
    ),
    ShippedIssueReceipt(
        issue=13072,
        merge_pr=13092,
        evidence_nodes=(
            "tests/unit/lifting/test_barbell_evidence.py::"
            "test_retracked_bundle_records_it_follows_its_source",
            "tests/unit/lifting/test_opensim_lift_ik_id.py::"
            "test_squat_receipt_range_violations_empty",
        ),
    ),
    ShippedIssueReceipt(
        issue=13104,
        merge_pr=13120,
        evidence_nodes=(
            "tests/unit/lifting/test_loaded_foot_tracking.py::"
            "test_loaded_feet_stay_within_the_lift5_ankle_bounds",
            "tests/unit/lifting/test_loaded_foot_tracking.py::"
            "test_tracking_converges_in_the_rk4_substep",
        ),
    ),
)


@pytest.mark.parametrize("receipt", SHIPPED_RECEIPTS, ids=lambda r: f"issue-{r.issue}")
def test_evidence_pytest_nodes_exist(receipt: ShippedIssueReceipt) -> None:
    for node in receipt.evidence_nodes:
        path = node.split("::", 1)[0]
        assert (REPO_ROOT / path).is_file(), node


_EVIDENCE_CASES: list[tuple[ShippedIssueReceipt, str]] = [
    (receipt, node)
    for receipt in SHIPPED_RECEIPTS
    for node in receipt.evidence_nodes
]


@pytest.mark.parametrize(
    ("receipt", "node"),
    _EVIDENCE_CASES,
    ids=[f"#{r.issue}-{n.split('::')[-1]}" for r, n in _EVIDENCE_CASES],
)
def test_evidence_node_passes(receipt: ShippedIssueReceipt, node: str) -> None:
    """Run the acceptance pytest node (same command CI uses for unit tests)."""
    result = subprocess.run(
        ["python3", "-m", "pytest", node, "-q", "--timeout=120"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"#{receipt.issue} evidence {node} failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )
