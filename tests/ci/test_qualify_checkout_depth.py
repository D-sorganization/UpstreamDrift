"""Tests for CI checkout depth qualification and sparse-state regression prevention.

Part of #14896, parent #14893.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci.qualify_checkout_depth import (
    CheckoutStepInfo,
    QualificationReport,
    check_sparse_contamination,
    inventory_workflow_checkouts,
    main,
    qualify_checkout_depth,
)

pytestmark = pytest.mark.unit

SAMPLE_WORKFLOW_YAML = """
name: CI Example
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          fetch-depth: 1
  spec-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          fetch-depth: 0
  unjustified-deep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          fetch-depth: 0
"""


def test_inventory_workflow_checkouts(tmp_path: Path) -> None:
    """Parse actions/checkout steps and extract fetch-depth and parameters."""
    wf_path = tmp_path / "ci.yml"
    wf_path.write_text(SAMPLE_WORKFLOW_YAML, encoding="utf-8")

    checkouts = inventory_workflow_checkouts(wf_path)
    assert len(checkouts) == 3
    c_map = {c.job_name: c for c in checkouts}

    assert c_map["lint"].fetch_depth == 1
    assert c_map["spec-check"].fetch_depth == 0
    assert c_map["unjustified-deep"].fetch_depth == 0


def test_qualify_checkout_depth(tmp_path: Path) -> None:
    """Verify qualification distinguishes justified full-history vs unjustified depth 0."""
    wf_path = tmp_path / "ci.yml"
    wf_path.write_text(SAMPLE_WORKFLOW_YAML, encoding="utf-8")

    # spec-check is justified; unjustified-deep is flagged
    justified_deep_jobs = {"spec-check"}
    reports = qualify_checkout_depth(wf_path, justified_deep_jobs=justified_deep_jobs)

    r_map = {r.job_name: r for r in reports}
    assert r_map["lint"].is_compliant
    assert r_map["spec-check"].is_compliant
    assert not r_map["unjustified-deep"].is_compliant
    assert "unjustified fetch-depth: 0" in (r_map["unjustified-deep"].reason or "")


def test_check_sparse_contamination_clean_and_dirty(tmp_path: Path) -> None:
    """Detect leaked sparse-checkout configuration in a workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)

    # Clean repo: no sparse contamination
    assert not check_sparse_contamination(repo)

    # Contaminated: core.sparseCheckout set to true without active config
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.sparseCheckout", "true"], check=True
    )
    assert check_sparse_contamination(repo)


def test_cli_execution_modes(tmp_path: Path) -> None:
    """CLI returns expected status codes on compliant vs non-compliant workflows."""
    wf = tmp_path / "compliant.yml"
    wf.write_text(
        "name: C\njobs:\n  t:\n    runs-on: u\n    steps:\n      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 1\n",
        encoding="utf-8",
    )
    assert main(["--workflow", str(wf)]) == 0

    bad_wf = tmp_path / "bad.yml"
    bad_wf.write_text(
        "name: B\njobs:\n  t:\n    runs-on: u\n    steps:\n      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n",
        encoding="utf-8",
    )
    assert main(["--workflow", str(bad_wf), "--strict"]) == 1
