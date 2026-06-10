from __future__ import annotations

from pathlib import Path

from scripts.ci import check_workflow_pytest_paths as guard


def test_missing_workflow_pytest_path_is_reported(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        """
jobs:
  test:
    steps:
      - run: |
          pytest \\
            tests/missing_file.py
""",
        encoding="utf-8",
    )

    failures = guard.missing_workflow_test_paths([workflow])

    assert failures == [f"{workflow}: missing pytest path tests/missing_file.py"]


def test_existing_workflow_pytest_path_passes(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "present.py").write_text("", encoding="utf-8")
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("run: pytest tests/present.py\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert guard.missing_workflow_test_paths([workflow]) == []
