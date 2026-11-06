"""``python -m src.tools.native_viewer_export``."""

from __future__ import annotations

from pathlib import Path
import sys

_VENDOR_SRC = Path(__file__).resolve().parents[3] / "vendor" / "ud-tools" / "src"


def _run() -> int:
    try:
        from src.tools.native_viewer_export.cli import main
    except ModuleNotFoundError as exc:
        from src.tools.native_viewer_export._bootstrap import shared_shadow_message

        shared = sys.modules.get("shared")
        message = shared_shadow_message(_VENDOR_SRC, getattr(shared, "__file__", None))
        if message is None:
            raise
        raise SystemExit(f"{exc}\n{message}") from exc
    return main()


if __name__ == "__main__":
    sys.exit(_run())
