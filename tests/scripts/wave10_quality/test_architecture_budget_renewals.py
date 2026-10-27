"""Regression tests for time-boxed architecture-budget exception renewals."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import check_architecture_budget as mod

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "scripts" / "config" / "architecture_budget.json"

_RENEWAL_EXPIRES_ON = "2027-01-31"
_TRACKING_ISSUE = 13853
_RENEWAL_REASON = (
    "Issue #13853: script source is hash-pinned by native requalification evidence; "
    "refactor lands with the next native requalification run "
    "(renewed once, 2026-10-26, owner-approved)."
)

_HASH_FROZEN_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "scripts/research/proximal_distal_energy/"
        "make_articulated_native_constraint_discrepancy_figure.py",
        "make_figure",
        "function-lines",
    ),
    (
        "scripts/research/proximal_distal_energy/articulated_drift_contact_attribution.py",
        "_evaluate_controls",
        "parameters",
    ),
)

# Snapshot from main @ b0c6bcc2f7: exception inventory keys only.
_MAIN_EXCEPTION_KEY_COUNT = 108
_MAIN_EXCEPTION_KEYS_SHA256 = (
    "21a2934ee76c29ae6d0935926623a1f373172c2c7e2b8b812e87a6f96cc1eaa5"
)


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(f"architecture budget config missing: {_CONFIG_PATH}")
    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "exceptions" not in config:
        raise ValueError(f"malformed architecture budget config: {_CONFIG_PATH}")
    return config


def _exception_keys(config: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(mod._exception_key(entry) for entry in config["exceptions"]))


def _keys_digest(keys: tuple[tuple[str, str, str], ...]) -> str:
    return hashlib.sha256(repr(keys).encode()).hexdigest()


def _find_exception_row(
    config: dict[str, Any],
    path: str,
    symbol: str,
    rule: str,
) -> dict[str, Any]:
    matches = [
        entry
        for entry in config["exceptions"]
        if mod._exception_key(entry) == (path, symbol, rule)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one exception row for {(path, symbol, rule)}, "
            f"found {len(matches)}"
        )
    return matches[0]


@pytest.mark.unit
def test_hash_frozen_research_exceptions_renewed_to_tracking_issue() -> None:
    """Hash-frozen research rows are renewed under the open native-run tracker."""
    config = _load_config()
    keys = _exception_keys(config)
    assert len(config["exceptions"]) == _MAIN_EXCEPTION_KEY_COUNT
    assert _keys_digest(keys) == _MAIN_EXCEPTION_KEYS_SHA256

    for path, symbol, rule in _HASH_FROZEN_ROWS:
        row = _find_exception_row(config, path, symbol, rule)
        assert row["expires_on"] == _RENEWAL_EXPIRES_ON
        assert row["issue"] == _TRACKING_ISSUE
        assert "#13853" in str(row["reason"])
        assert str(row["reason"]) == _RENEWAL_REASON
        assert date.fromisoformat(str(row["expires_on"])) <= date.fromisoformat(
            _RENEWAL_EXPIRES_ON
        )


@pytest.mark.unit
def test_no_architecture_budget_exception_expires_before_2026_11_01_for_renewed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renewed rows stay active when evaluated on the 2026-10-31 -> 2026-11-01 boundary."""
    config = _load_config()
    evaluation_day = date(2026, 11, 1)

    def _active_at_boundary(exc: dict[str, Any]) -> bool:
        expires_on = exc.get("expires_on")
        if not expires_on:
            return True
        return evaluation_day <= date.fromisoformat(str(expires_on))

    monkeypatch.setattr(mod, "_exception_is_active", _active_at_boundary)

    active, invalid = mod._collect_active_exceptions(config)

    for path, symbol, rule in _HASH_FROZEN_ROWS:
        key = (path, symbol, rule)
        assert key in active, f"{key} must be active on {evaluation_day}"
        expired = [msg for msg in invalid if path in msg and symbol in msg]
        assert expired == [], f"unexpected invalid messages for {key}: {expired}"
