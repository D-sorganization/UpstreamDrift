"""Unit tests asserting knowledge-pack and Sidekick Wizards availability (K0 + K3a).

Validates that UpstreamDrift's shared AI surface includes the knowledge-pack
engine (Tools#5348) and Sidekick Wizards glue (Tools#5350) introduced in
Tools commit 95ed6b47 (issue #10944).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_knowledge_package_exported_symbols() -> None:
    """Verify knowledge-pack engine public API is available from shared.python.ai."""
    from src.shared.python.ai import knowledge

    assert hasattr(knowledge, "load_manifest")
    assert hasattr(knowledge, "build_pack")
    assert hasattr(knowledge, "chunk_document")
    assert hasattr(knowledge, "KnowledgePack")
    assert hasattr(knowledge, "PackManifest")


@pytest.mark.unit
def test_sidekick_wizards_exported_symbols() -> None:
    """Verify Sidekick Wizards glue is available from shared.python.ai."""
    from src.shared.python.ai import wizards

    assert hasattr(wizards, "wizard_for")
    assert hasattr(wizards, "knowledge_for_context")
    assert hasattr(wizards, "reset_wizards")


@pytest.mark.unit
def test_knowledge_chunk_document_basic() -> None:
    """Verify document chunking works as expected from src.shared.python.ai."""
    from src.shared.python.ai.knowledge import chunk_document

    markdown = "# Heading 1\n\nParagraph text here.\n\n# Heading 2\n\nSecond paragraph."
    chunks = chunk_document(markdown, suffix=".md", max_chars=1000)
    assert len(chunks) == 2
    assert chunks[0].title == "Heading 1"
    assert "Paragraph text here." in chunks[0].text
