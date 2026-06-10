"""Regression tests for configuration ownership boundaries (#7216)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def test_no_root_config_directories_remain() -> None:
    assert not (REPO_ROOT / "config").exists()
    assert not (REPO_ROOT / "configs").exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/development/configuration-systems.md",
        "scripts/config/architecture_debt_policy.json",
        "src/bunkershot3d/calibration/configs/canonical.yaml",
        "src/shared/python/ux/config/field_metadata.yaml",
        "src/shared/python/ux/config/error_messages.yaml",
    ],
)
def test_documented_config_homes_exist(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_file()
