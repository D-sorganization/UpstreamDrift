"""The public docs use the approved fixed-state ZTCF/ZVCF decomposition."""

import os
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    ROOT / "SPEC.md",
    ROOT / "AGENTS.md",
    ROOT / "docs/engines/engine_capabilities.md",
    ROOT / "docs/help/drake_golf.md",
    ROOT / "docs/help/opensim_golf.md",
    ROOT / "docs/help/pinocchio_golf.md",
    ROOT / "docs/simulation_backends/README.md",
    ROOT / "src/shared/python/simulation_backends/README.md",
)
pytestmark = pytest.mark.unit


def test_public_docs_use_the_approved_fixed_state_component_mapping() -> None:
    for path in DOCUMENTS:
        text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).casefold()
        assert "ztcf is the drift component" in text, path
        assert "zvcf is the control component" in text, path
        assert "not additive over integrated trajectories" in text, path


def test_spec_records_that_the_contract_reverses_issue_11553() -> None:
    text = re.sub(
        r"\s+", " ", (ROOT / "SPEC.md").read_text(encoding="utf-8")
    ).casefold()
    assert "reverses #11553" in text
    assert "exact at a fixed state" in text


def test_each_consumer_links_to_the_normative_spec_section() -> None:
    spec = ROOT / "SPEC.md"
    for path in DOCUMENTS[1:]:
        link = Path(os.path.relpath(spec, path.parent)).as_posix()
        text = path.read_text(encoding="utf-8")
        assert f"]({link})" in text, path
        assert (path.parent / link).resolve() == spec.resolve(), path
        assert spec.is_file(), path
