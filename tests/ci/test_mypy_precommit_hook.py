"""Verify pre-push mypy hook configuration in .pre-commit-config.yaml."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _mypy_hook() -> dict[str, Any]:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    for repository in config.get("repos", []):
        if "mirrors-mypy" in repository.get("repo", ""):
            for hook in repository.get("hooks", []):
                if hook.get("id") == "mypy":
                    return dict(hook)
    raise AssertionError(
        "mypy hook not found under mirrors-mypy repository in .pre-commit-config.yaml"
    )


def _get_requirements_lock_numpy() -> str:
    lock_text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    match = re.search(r"^numpy==([0-9a-zA-Z\.\-]+)", lock_text, re.MULTILINE)
    assert match is not None, "numpy pin not found in requirements.lock"
    return match.group(0)


def test_mypy_hook_includes_numpy_dependency() -> None:
    """The mypy hook must include numpy matching requirements.lock (#11800)."""
    hook = _mypy_hook()
    deps = hook.get("additional_dependencies", [])
    expected_numpy_pin = _get_requirements_lock_numpy()
    assert expected_numpy_pin in deps, (
        f"mypy hook additional_dependencies must include '{expected_numpy_pin}' "
        f"matching requirements.lock to resolve numpy.typing aliases (issue #11800), got: {deps}"
    )
