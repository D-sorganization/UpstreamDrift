"""Primary-source bibliographic contract for Koike (2016) in the open release."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ROOT / "docs/research/proximal_distal_energy_transfer/references.bib"

PRIMARY_TITLE = "Force and Moment Exerted by Each Hand on an Instrumented Golf Club"
PRIMARY_BOOKTITLE = "34th International Conference on Biomechanics in Sports"
PRIMARY_URL = "https://ojs.ub.uni-konstanz.de/cpa/article/view/6828"
LEGACY_WRONG_TITLE = "Measurement of Individual Hand Forces Exerted on a Golf Club Grip"


def _koike2016_block() -> str:
    text = REFERENCES.read_text(encoding="utf-8")
    match = re.search(r"@inproceedings\{koike2016,(.*?)\n\}", text, re.DOTALL)
    assert match is not None, "koike2016 entry is missing from references.bib"
    return match.group(1)


def test_koike2016_matches_primary_proceedings_metadata() -> None:
    """The governed release must cite the proceedings title, not a legacy mislabel."""
    block = _koike2016_block()
    assert LEGACY_WRONG_TITLE not in block
    assert f"title     = {{{PRIMARY_TITLE}}}" in block
    assert f"booktitle = {{{PRIMARY_BOOKTITLE}}}" in block
    assert f"url       = {{{PRIMARY_URL}}}" in block
