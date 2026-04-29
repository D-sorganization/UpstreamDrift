"""Enforce the pip-audit waiver schema (issue #3076).

Every entry in ``.github/pip_audit_waivers.yaml`` MUST carry an expiry
date in YYYY-MM-DD form. Missing / malformed expiries fail this test.

Also verifies that every waiver has a non-empty justification, and that
no two waivers share the same CVE.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
WAIVERS_PATH = REPO_ROOT / ".github" / "pip_audit_waivers.yaml"

REQUIRED_FIELDS = {"cve", "package", "reason", "expires"}


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def waivers() -> list[dict]:
    """Load waivers from the canonical YAML file."""
    assert WAIVERS_PATH.exists(), (
        f"{WAIVERS_PATH} is missing; pip-audit waivers are required to live "
        "in this file so CI can verify expiry dates (issue #3076)."
    )
    data = yaml.safe_load(WAIVERS_PATH.read_text()) or {}
    entries = data.get("waivers")
    assert isinstance(
        entries, list
    ), "Top-level 'waivers:' key must be a list of mappings."
    return entries


def test_every_waiver_has_required_fields(waivers: list[dict]) -> None:
    """Every waiver must carry cve, package, reason, and expires."""
    for idx, entry in enumerate(waivers):
        assert isinstance(entry, dict), f"waivers[{idx}] is not a mapping"
        missing = REQUIRED_FIELDS - entry.keys()
        cve = entry.get("cve", "<unknown>")
        assert (
            not missing
        ), f"waivers[{idx}] ({cve}) is missing required fields: {sorted(missing)}"


def test_every_waiver_expires_is_parseable_date(waivers: list[dict]) -> None:
    """expires must be YYYY-MM-DD (string) or a YAML date scalar."""
    for entry in waivers:
        expires = entry["expires"]
        cve = entry["cve"]
        if isinstance(expires, _dt.date) and not isinstance(expires, _dt.datetime):
            continue  # YAML date scalar is accepted
        assert isinstance(
            expires, str
        ), f"{cve}: expires must be a YYYY-MM-DD string, got {type(expires).__name__}"
        try:
            _dt.date.fromisoformat(expires)
        except ValueError as exc:  # pragma: no cover - assertion failure
            pytest.fail(f"{cve}: expires is not YYYY-MM-DD: {exc}")


def test_every_waiver_reason_is_nonempty(waivers: list[dict]) -> None:
    """Each waiver must carry a human-readable justification."""
    for entry in waivers:
        cve = entry["cve"]
        reason = entry.get("reason") or ""
        assert reason.strip(), f"{cve}: reason must not be empty"


def test_no_duplicate_cves(waivers: list[dict]) -> None:
    """Two waivers for the same CVE are a configuration error."""
    seen: set[str] = set()
    dupes: list[str] = []
    for entry in waivers:
        cve = entry["cve"]
        if cve in seen:
            dupes.append(cve)
        seen.add(cve)
    assert not dupes, f"Duplicate waiver entries for: {dupes}"
