"""The pinned Tools adapters must keep the hardening #8776 found dropped.

UpstreamDrift's ``src/shared/python/ai/adapters`` is a Tools child copy ruled
``tools-canonical`` (docs/shared_tools/seam_rulings.v1.json); the fixes for the
empty-trailing-turn bug, the Ollama error ladder and the BitNet prompt ceiling
live in Tools. This guard reads the vendored sources so a vendor bump that
loses them fails here (see docs/audits/consolidation_hardening_audit_2026-09.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.helpers.seam_guards import require_vendor_path

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ADAPTERS = (
    _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python" / "ai" / "adapters"
)

_STRIP_GUARD = re.compile(r"(current_message|effective_message|message)\.strip\(\)")


def _adapter_source(name: str) -> str:
    path = _ADAPTERS / f"{name}_adapter.py"
    require_vendor_path(path)
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama", "gemini"])
def test_provider_formatters_guard_against_blank_trailing_turn(provider: str) -> None:
    source = _adapter_source(provider)
    assert _STRIP_GUARD.search(source), (
        f"{provider}_adapter.py lost the blank-message guard restored after "
        "3e09be404 (#8776)"
    )


def test_ollama_keeps_typed_error_ladder() -> None:
    source = _adapter_source("ollama")
    assert "isinstance(e, httpx.ConnectError)" in source
    assert "Is Ollama running" in source


def test_bitnet_keeps_prompt_ceiling_and_validation() -> None:
    source = _adapter_source("bitnet")
    assert "_MAX_PROMPT_BYTES" in source, "BitNet prompt size ceiling dropped (#8322)"
    assert "_build_validated_prompt" in source, (
        "BitNet prompt validation dropped (#8322)"
    )
