"""Protected CI contracts for proximal-distal publication qualification."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci-standard.yml"
pytestmark = pytest.mark.unit


def _workflow() -> tuple[str, dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def test_publication_paths_drive_a_dedicated_protected_job() -> None:
    text, workflow = _workflow()
    jobs = workflow["jobs"]
    changed = jobs["changed-paths"]
    publication = jobs["publication-quality"]

    assert (
        changed["outputs"]["publication"] == "${{ steps.detect.outputs.publication }}"
    )
    detect = changed["steps"][2]["run"]
    for path in (
        ".github/workflows/ci-standard.yml",
        "docs/research/proximal_distal_energy_transfer/*",
        "scripts/research/proximal_distal_energy/*",
        "tests/research/test_publication_quality.py",
        "pyproject.toml",
    ):
        assert path in detect
    assert 'echo "publication=true" >> "$GITHUB_OUTPUT"' in detect
    assert publication["needs"] == ["pick-runner", "changed-paths"]
    assert publication["if"] == "needs.changed-paths.outputs.publication == 'true'"
    assert "pymupdf>=1.26.0" in text
    assert "pikepdf>=10.0.0" in text


def test_publication_job_runs_the_complete_revision_bound_validator() -> None:
    _, workflow = _workflow()
    steps = workflow["jobs"]["publication-quality"]["steps"]
    command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Validate release authority and every PDF page"
    )

    assert "qualify_open_release validate" in command
    assert '--source-revision "$(git rev-parse HEAD)"' in command
    assert "--publication-profile computational" in command


def test_required_quality_gate_fails_closed_on_publication_result() -> None:
    _, workflow = _workflow()
    gate = workflow["jobs"]["quality-gate"]
    script = gate["steps"][0]["run"]

    assert "publication-quality" in gate["needs"]
    assert gate["steps"][0]["env"]["PUBLICATION_QUALITY"] == (
        "${{ needs.publication-quality.result }}"
    )
    assert script.count("failed=0") == 1
    assert script.index("failed=0") < script.index("publication_expected")
    assert 'publication_expected="success"' in script
    assert 'publication_expected="skipped"' in script
    assert 'PUBLICATION_QUALITY" != "$publication_expected' in script
    assert "Docs-only publication change: publication-quality succeeded" in script
    assert script.index("publication-quality reported") < script.index(
        'if [ "$failed" -ne 0 ]'
    )
