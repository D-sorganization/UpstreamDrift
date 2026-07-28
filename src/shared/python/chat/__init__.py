"""Shared chat subpackage.

This package extends __path__ to merge local extensions with vendor/ud-tools shared chat modules.
"""

from pathlib import Path

_vendor_chat = (
    Path(__file__).resolve().parents[4]
    / "vendor"
    / "ud-tools"
    / "src"
    / "shared"
    / "python"
    / "chat"
)
if _vendor_chat.exists() and str(_vendor_chat) not in __path__:
    __path__.append(str(_vendor_chat))
