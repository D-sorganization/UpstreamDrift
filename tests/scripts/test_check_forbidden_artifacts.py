from pathlib import Path

from scripts.check_forbidden_artifacts import find_forbidden_artifacts


def test_find_forbidden_artifacts_reports_root_outputs(tmp_path: Path) -> None:
    assert find_forbidden_artifacts(tmp_path, ["coverage.json"]) == [
        Path("coverage.json")
    ]


def test_find_forbidden_artifacts_reports_jules_completist_data(
    tmp_path: Path,
) -> None:
    assert find_forbidden_artifacts(
        tmp_path, [".jules/completist_data/todo_markers.txt"]
    ) == [Path(".jules/completist_data/todo_markers.txt")]


def test_find_forbidden_artifacts_ignores_unrelated_agent_config(
    tmp_path: Path,
) -> None:
    assert find_forbidden_artifacts(tmp_path, [".agent/skills/lint/SKILL.md"]) == []
