"""Drift guard for chat modules synchronized with Tools.
The baseline hashes were captured from the matching files in the sibling
Tools repository and should only change when that upstream source changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5 if "tests" in __file__ else 4]
# Note: ``src/shared/python/chat/__init__.py`` intentionally diverges from
# Tools to add UpstreamDrift-specific exports (service_base, router_factory),
# so it is not included in this hash baseline. Likewise ``chat_dock_widget.py``
# in UpstreamDrift is a lazy import shim; the canonical Tools content lives in
# ``_chat_dock_widget_qt.py``.
TOOLS_BASELINE_HASHES: dict[str, str] = {
    "src/shared/python/chat/_chat_dock_widget_qt.py": "a5b5689b1e2cd75c20bee9f459de815a85b482fa0410200468afc09cb8f3dfdc",
    "src/shared/python/chat/models.py": "4be2df1e9aee66c849efa6de9930d058eba4c0a0efe31528768850da33ce4838",
    "src/shared/python/chat/tests/__init__.py": "5a0bba6299ce217de8cbfc2e20a354ccf479e8d45152f69ad2543d9183d07812",
    "src/shared/python/chat/tests/test_chat.py": "cd1c1c21570262bbafa19e6217d4cff07fe073051423c031aa17f3ea051881c7",
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
