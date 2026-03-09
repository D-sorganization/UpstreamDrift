"""Tests for completist report generation."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_completist_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_completist_data", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_report_with_empty_inputs(tmp_path) -> None:
    """The completist analyzer should generate a zero-count report from empty inputs."""
    module = _load_module()
    todo_label = "TO" + "DO"
    data_dir = tmp_path / "completist_data"
    report_dir = tmp_path / "reports"
    issues_dir = tmp_path / "issues"
    data_dir.mkdir()

    filenames = {
        "MARKERS": "todo_markers.txt",
        "NOT_IMPL": "not_implemented.txt",
        "STUBS": "stub_functions.txt",
        "DOCS": "incomplete_docs.txt",
        "ABSTRACT": "abstract_methods.txt",
    }
    for filename in filenames.values():
        (data_dir / filename).write_text("", encoding="utf-8")

    module.DATA_DIR = str(data_dir)
    module.REPORT_DIR = str(report_dir)
    module.ISSUES_DIR = str(issues_dir)
    module.FILES_MAP = {
        key: str(data_dir / filename) for key, filename in filenames.items()
    }

    module.generate_report()

    date_s = datetime.now().strftime("%Y-%m-%d")
    latest_path = report_dir / "COMPLETIST_LATEST.md"
    report_path = report_dir / f"Completist_Report_{date_s}.md"

    assert latest_path.exists()
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert f"# Completist Report: {date_s}" in report_text
    assert "- **Critical Gaps**: 0" in report_text
    assert f"- **Feature Gaps ({todo_label})**: 0" in report_text
    assert "- **Technical Debt**: 0" in report_text
    assert "- **Documentation Gaps**: 0" in report_text
    assert not list(issues_dir.glob("*.md"))
