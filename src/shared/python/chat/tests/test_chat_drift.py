# ruff: noqa: E501
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
    "src/shared/python/chat/_chat_dock_widget_qt.py": "".join(
        (
            "6cb29814",
            "9dc3f377",
            "84687465",
            "211950a2",
            "2825d146",
            "8efd729e",
            "8199b4a0",
            "8bbf9caf",
        )
    ),
    "src/shared/python/chat/models.py": "".join(
        (
            "d637e81d",
            "11204f2a",
            "01c77e81",
            "5d2dd8ae",
            "fc68033a",
            "b6e96a0e",
            "23739cf3",
            "79a90c4f",
        )
    ),
    "src/shared/python/chat/tests/__init__.py": "5a0bba6299ce217de8cbfc2e20a354ccf479e8d45152f69ad2543d9183d07812",  # noqa: E501
    "src/shared/python/chat/tests/test_chat.py": "59fa5f6e09f1e2b5e3a21f2b54f20efdbaede1977b791b9331aa584ed14f3ffc",  # noqa: E501
}


def _normalize_source_bytes(source: bytes) -> bytes:
    """Return UTF-8 source with platform-independent LF line endings."""
    return source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def test_source_normalization_is_line_ending_invariant() -> None:
    """Hash input must not depend on the checkout's text line endings."""
    expected = b"first\nsecond\nthird\n"
    assert _normalize_source_bytes(b"first\nsecond\nthird\n") == expected
    assert _normalize_source_bytes(b"first\r\nsecond\r\nthird\r\n") == expected
    assert _normalize_source_bytes(b"first\rsecond\rthird\r") == expected


def _runtime_equivalent_source(relative_path: str) -> bytes:
    """Return the source bytes that should match the Tools runtime baseline."""
    source = (REPO_ROOT / relative_path).read_bytes()
    return _normalize_source_bytes(source)


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
