"""Fail-fast diagnosis when a foreign ``shared`` package shadows the Tools tree (#14748)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.native_viewer_export._bootstrap import shared_shadow_message

VENDOR_SRC = Path("/repo/vendor/ud-tools/src")


def test_no_message_when_shared_comes_from_vendor_tree() -> None:
    inside = str(VENDOR_SRC / "shared" / "__init__.py")
    assert shared_shadow_message(VENDOR_SRC, inside) is None


def test_no_message_when_shared_is_a_namespace_package() -> None:
    assert shared_shadow_message(VENDOR_SRC, None) is None


def test_message_names_foreign_origin_and_remedy() -> None:
    foreign = "/other/Tools_Private/src/shared/__init__.py"
    message = shared_shadow_message(VENDOR_SRC, foreign)
    assert message is not None
    assert foreign in message
    assert "editable" in message
    assert "venv" in message


def test_vendor_src_must_be_a_path() -> None:
    with pytest.raises(TypeError, match="vendor_src"):
        shared_shadow_message("not-a-path", None)  # type: ignore[arg-type]
