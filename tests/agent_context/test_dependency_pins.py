"""Every installed provider must match the context source revision."""

import tomllib
from pathlib import Path

import pytest
from agent_context.service import ContextService

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_python_and_rust_install_paths_match_verified_gitlink() -> None:
    """Reject an older release wheel even when source and Rust pins agree."""
    state = ContextService(ROOT).status()
    revision = state["provenance"]["dependencies"]["vendor/ud-tools"]["pinned"]
    requirements = (ROOT / "requirements-tools.txt").read_text(encoding="utf-8")
    active = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    repository = "https://github.com/D-sorganization/Tools.git"
    assert active == ["ud-tools @ git+" + repository + "@" + revision]
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    assert cargo["workspace"]["dependencies"]["tools-core"]["rev"] == revision
