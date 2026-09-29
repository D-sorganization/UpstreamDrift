from pathlib import Path

from scripts.check_github_actions_pinned import audit_workflows


def write_workflow(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_workflows_accepts_sha_pinned_actions(tmp_path: Path) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    write_workflow(
        workflow_dir / "ci.yml",
        """
name: CI
jobs:
  test:
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - uses: ./.github/workflows/reusable.yml
""",
    )

    assert audit_workflows(workflow_dir) == []


def test_audit_workflows_rejects_version_tags(tmp_path: Path) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    write_workflow(
        workflow_dir / "ci.yml",
        """
name: CI
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
""",
    )

    findings = audit_workflows(workflow_dir)

    assert len(findings) == 1
    assert findings[0].line_number == 6
    assert findings[0].reference == "actions/checkout@v4"


def test_audit_workflows_rejects_external_uses_without_ref(tmp_path: Path) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    write_workflow(
        workflow_dir / "ci.yml",
        """
name: CI
jobs:
  test:
    steps:
      - uses: actions/checkout
""",
    )

    findings = audit_workflows(workflow_dir)

    assert len(findings) == 1
    assert findings[0].reason == "missing commit ref"
