"""Characterization tests pinning the URDF campaign issue definitions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

# sha256 of the JSON-serialised ISSUES list measured before the split by domain.
PINNED_ISSUES_SHA256 = (
    "3db696d486254a554dd74aea7a873b45d1f68e8f7b8b64d7ff968ea59d95705a"
)


@pytest.fixture(scope="module")
def campaign():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import _urdf_campaign_issues

        yield _urdf_campaign_issues
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


@pytest.mark.unit
def test_issue_definitions_are_unchanged(campaign) -> None:
    assert len(campaign.ISSUES) == 33
    digest = hashlib.sha256(
        json.dumps(campaign.ISSUES, ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == PINNED_ISSUES_SHA256


@pytest.mark.unit
def test_issue_tuples_are_well_formed(campaign) -> None:
    for title, labels, body in campaign.ISSUES:
        assert title
        assert isinstance(labels, list)
        assert body.strip()


@pytest.mark.unit
def test_main_creates_only_non_skipped_issues(campaign, monkeypatch, tmp_path) -> None:
    created: list[str] = []

    def fake_create(title: str, body: str, labels: list[str]) -> int:
        created.append(title)
        return len(created)

    monkeypatch.setattr(campaign, "gh_issue_create", fake_create)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    campaign.main()
    expected = [
        t
        for t, _, _ in campaign.ISSUES
        if not t.startswith(campaign.SKIP_TITLES_PREFIX)
    ]
    assert created == expected
    written = json.loads(
        (tmp_path / "scripts" / "_urdf_campaign_issues.json").read_text()
    )
    assert [row["title"] for row in written] == expected
