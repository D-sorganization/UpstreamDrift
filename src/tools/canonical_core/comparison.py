"""Standalone entry point for the canonical-core comparison shell."""

from __future__ import annotations

from src.tools.canonical_core.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main(["--tool-id", "canonical_core_comparison"]))
