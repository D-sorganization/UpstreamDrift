"""Unit tests for K3b Drift Wizard knowledge pack and Sidekick integration (#10943).

Validates that:
- knowledge/wizard.yml loads and conforms to the WizardConfig contract.
- knowledge/pack.yml loads and every include glob matches >= 1 file.
- Sidekick retrieval locates expected product passages.
- Stale pack banner rendering behaves correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.ai.knowledge.manifest import PackManifest, load_manifest
from src.shared.python.ai.knowledge.pack import KnowledgePack, build_pack
from src.shared.python.ai.knowledge.sources import _glob_re, list_repo_files
from src.shared.python.ai.knowledge.wizard import (
    STALE_BANNER,
    KnowledgeContext,
    WizardConfig,
    WizardKnowledge,
    load_wizard_config,
)
from src.shared.python.ai.types import ConversationContext, Message
from src.shared.python.ai.wizards import (
    knowledge_for_context,
    reset_wizards,
    wizard_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_wizard_config_loads() -> None:
    """Verify that knowledge/wizard.yml exists and loads as a valid WizardConfig."""
    wizard_path = REPO_ROOT / "knowledge/wizard.yml"
    assert wizard_path.is_file(), "knowledge/wizard.yml must exist"
    config = load_wizard_config(REPO_ROOT)
    assert isinstance(config, WizardConfig)
    assert config.key == "upstream_drift"
    assert config.name == "Drift Wizard"
    assert config.manifest.resolve() == (REPO_ROOT / "knowledge/pack.yml").resolve()
    assert config.pack.resolve() == (REPO_ROOT / ".knowledge/pack.sqlite").resolve()
    assert bool(config.capabilities)


@pytest.mark.unit
def test_manifest_loads() -> None:
    """Verify knowledge/pack.yml loads and includes required sources."""
    manifest_path = REPO_ROOT / "knowledge/pack.yml"
    assert manifest_path.is_file(), "knowledge/pack.yml must exist"
    manifest = load_manifest(manifest_path)
    assert isinstance(manifest, PackManifest)
    assert manifest.id == "upstream_drift"

    authorities = {s.authority for s in manifest.sources}
    assert "product" in authorities, "Expected 'product' authority source"
    assert "reference" in authorities, "Expected 'reference' authority source"


@pytest.mark.unit
def test_every_include_glob_matches_at_least_one_file() -> None:
    """Every glob pattern in knowledge/pack.yml must match at least one file."""
    manifest_path = REPO_ROOT / "knowledge/pack.yml"
    assert manifest_path.is_file(), "knowledge/pack.yml must exist"
    manifest = load_manifest(manifest_path)

    repo_files = list_repo_files(REPO_ROOT)
    for source in manifest.sources:
        for glob_pattern in source.include:
            pattern = _glob_re(glob_pattern)
            matched = [f for f in repo_files if pattern.fullmatch(f)]
            assert bool(matched), (
                f"Glob {glob_pattern!r} in source ({source.authority}) matched no files!"
            )


@pytest.mark.unit
def test_fixture_question_retrieves_passage_from_expected_doc(tmp_path: Path) -> None:
    """Build a temporary pack and ensure a user question retrieves relevant doc passages."""
    manifest_path = REPO_ROOT / "knowledge/pack.yml"
    assert manifest_path.is_file(), "knowledge/pack.yml must exist"
    manifest = load_manifest(manifest_path)

    pack_path = tmp_path / "test_pack.sqlite"
    build_pack(
        manifest=manifest,
        roots={"UpstreamDrift": REPO_ROOT},
        out=pack_path,
    )

    pack = KnowledgePack.open(pack_path)
    passages = pack.search("How do I get started with simulations?", k=5)
    assert bool(passages), "Expected at least one passage retrieved from pack"
    assert any(
        "user_guide" in p.source.lower()
        or "getting_started" in p.source.lower()
        or "help" in p.source.lower()
        or "readme" in p.source.lower()
        for p in passages
    ), f"Expected user guide or help passage, got: {[p.source for p in passages]}"


@pytest.mark.unit
def test_wizard_provides_context_for_sidekick(tmp_path: Path) -> None:
    """WizardKnowledge produces KnowledgeContext with citations and app name."""
    manifest_path = REPO_ROOT / "knowledge/pack.yml"
    assert manifest_path.is_file(), "knowledge/pack.yml must exist"
    manifest = load_manifest(manifest_path)

    pack_path = tmp_path / "test_pack.sqlite"
    build_pack(
        manifest=manifest,
        roots={"UpstreamDrift": REPO_ROOT},
        out=pack_path,
    )

    config = WizardConfig(
        key="upstream_drift",
        name="Drift Wizard",
        description="Product expert for UpstreamDrift",
        capabilities=("Guide users on biomechanical simulation engines",),
        manifest=manifest_path,
        pack=pack_path,
        k=5,
        roots={"UpstreamDrift": REPO_ROOT},
    )
    wizard = WizardKnowledge(config)
    ctx = wizard.context_for("How do I configure biomechanical simulation?")
    assert isinstance(ctx, KnowledgeContext)
    assert ctx.stale is False
    assert bool(ctx.passages), "Expected passages in wizard context"
    rendered = ctx.render()
    assert "Drift Wizard" in rendered
    assert (
        "Knowledge from Drift Wizard (cite as [n]; answer from these first):"
        in rendered
    )


@pytest.mark.unit
def test_wizard_stale_pack_shows_banner(tmp_path: Path) -> None:
    """When a pack is stale, the rendered context must include the stale banner."""
    manifest_path = REPO_ROOT / "knowledge/pack.yml"
    assert manifest_path.is_file(), "knowledge/pack.yml must exist"
    pack_path = tmp_path / "test_pack.sqlite"

    build_pack(
        manifest=load_manifest(manifest_path),
        roots={"UpstreamDrift": REPO_ROOT},
        out=pack_path,
    )

    config = WizardConfig(
        key="upstream_drift",
        name="Drift Wizard",
        description="Product expert for UpstreamDrift",
        capabilities=("Guide users on biomechanical simulation engines",),
        manifest=manifest_path,
        pack=pack_path,
        k=5,
        roots={"UpstreamDrift": REPO_ROOT},
    )
    wizard = WizardKnowledge(config)
    with patch.object(wizard, "is_stale", return_value=True):
        ctx = wizard.context_for("How do I run simulations?")
        assert isinstance(ctx, KnowledgeContext)
        assert ctx.stale is True
        assert STALE_BANNER in ctx.render()


@pytest.mark.unit
def test_sidekick_wizards_glue_integration() -> None:
    """Verify sidekick.wizards glue resolves the Drift Wizard from project_root."""
    reset_wizards()
    wizard = wizard_for(REPO_ROOT)
    assert wizard is not None
    assert wizard.config.key == "upstream_drift"
    assert wizard.config.name == "Drift Wizard"

    pack_path = wizard.config.pack
    pack_existed = pack_path.is_file()
    if not pack_existed:
        build_pack(
            manifest=load_manifest(wizard.config.manifest),
            roots={"UpstreamDrift": REPO_ROOT},
            out=pack_path,
        )
    try:
        ctx = ConversationContext(
            messages=[Message(role="user", content="How do I get started?")],
            metadata={"project_root": str(REPO_ROOT)},
        )
        kctx = knowledge_for_context(ctx)
        assert isinstance(kctx, KnowledgeContext)
    finally:
        if not pack_existed:
            pack_path.unlink(missing_ok=True)
