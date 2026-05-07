from __future__ import annotations

from pathlib import Path

import yaml


def test_optional_stack_workflow_forces_bash_for_self_hosted_steps() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/ci-optional-stack.yml").read_text(encoding="utf-8")
    )

    defaults = workflow.get("defaults", {}).get("run", {})

    assert defaults.get("shell") == "bash"
