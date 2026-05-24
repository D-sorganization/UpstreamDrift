"""Workflow contract tests for standalone Sidekick packaging and release."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_WORKFLOW = ROOT / ".github" / "workflows" / "package-standalone-sidekick.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"


def test_package_workflow_builds_distribution_with_ui_bundle_enabled() -> None:
    """Distribution builds must let the hatch hook create the UI bundle."""
    content = PACKAGE_WORKFLOW.read_text(encoding="utf-8")

    assert "python3 -m build --outdir dist/" in content
    assert "SKIP_UI_BUILD" not in content


def test_release_workflow_has_queue_protection_and_timeouts() -> None:
    """Release workflow jobs must follow the repo queue-saturation guardrails."""
    content = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "concurrency:" in content
    assert "cancel-in-progress: true" in content
    assert "attach-to-release:" in content
    attach_section = content.split("attach-to-release:", maxsplit=1)[1]
    assert "timeout-minutes:" in attach_section
