"""Drift guard for chat modules synchronized with Tools.
The baseline hashes were captured from the matching files in the sibling
Tools repository and should only change when that upstream source changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5 if "tests" in __file__ else 4]
TOOLS_BASELINE_HASHES: dict[str, str] = {
    "src/shared/python/chat/__init__.py": "33ba253d1351e40b778e0c620d2c091f978285a4b8487040e5c2f93bb28e6268",
    "src/shared/python/chat/chat_dock_widget.py": "c1947bcd98d7868c3681116b103547519a89a4010c46d89410c3f471133551fc",
    "src/shared/python/chat/models.py": "efba02eab03e4b74cc7511fa45595b790379a2ddbc0f95fa54f9f67f91ba94e8",
    "src/shared/python/chat/tests/__init__.py": "5a0bba6299ce217de8cbfc2e20a354ccf479e8d45152f69ad2543d9183d07812",
    "src/shared/python/chat/tests/test_chat.py": "e7ed8d44073b8fe2015aa006218d6c1b717b52e057e51f5985a78e6177254c30",
}


def _runtime_equivalent_source(relative_path: str) -> bytes:
    """Return the source bytes that should match the Tools runtime baseline."""
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return source.encode("utf-8")


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    sorted(TOOLS_BASELINE_HASHES.items()),
)
def test_chat_modules_match_tools_baseline(
    relative_path: str,
    expected_sha256: str,
) -> None:
    """Verify the selected leaf modules still match the Tools baseline."""
    path = REPO_ROOT / relative_path
    if not path.exists():
        pytest.fail(f"Missing file: {relative_path}")
    actual_sha256 = hashlib.sha256(
        _runtime_equivalent_source(relative_path)
    ).hexdigest()
    assert actual_sha256 == expected_sha256, relative_path
