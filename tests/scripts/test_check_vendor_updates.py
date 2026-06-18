from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_vendor_updates

pytestmark = pytest.mark.unit

_WORKFLOW = Path(".github/workflows/vendor-freshness.yml")


def _status(*, is_current: bool) -> check_vendor_updates.SubmoduleStatus:
    return check_vendor_updates.SubmoduleStatus(
        path="vendor/ud-tools",
        pinned_sha="abc123",
        upstream_sha="def456",
        is_current=is_current,
        message="vendor/ud-tools status line",
    )


def test_json_mode_emits_parseable_status_array(monkeypatch, capsys) -> None:
    def fake_check(path: str, api_url: str | None, use_network: bool):
        assert path == "vendor/ud-tools"
        assert api_url == "https://example.test/commit"
        assert use_network is False
        return _status(is_current=False)

    monkeypatch.setattr(
        check_vendor_updates,
        "SUBMODULE_UPSTREAM",
        {"vendor/ud-tools": "https://example.test/commit"},
    )
    monkeypatch.setattr(check_vendor_updates, "check_submodule", fake_check)

    exit_code = check_vendor_updates.main(["--no-network", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == [
        {
            "path": "vendor/ud-tools",
            "pinned_sha": "abc123",
            "upstream_sha": "def456",
            "is_current": False,
            "message": "vendor/ud-tools status line",
        }
    ]


def test_text_mode_emits_status_and_stale_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_vendor_updates,
        "SUBMODULE_UPSTREAM",
        {"vendor/ud-tools": "https://example.test/commit"},
    )
    monkeypatch.setattr(
        check_vendor_updates,
        "check_submodule",
        lambda *_args: _status(is_current=False),
    )

    assert check_vendor_updates.main(["--no-network"]) == 0

    output = capsys.readouterr().out
    assert "vendor/ud-tools status line" in output
    assert "1 submodule(s) are behind upstream." in output


def test_fail_on_stale_preserves_machine_readable_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_vendor_updates,
        "SUBMODULE_UPSTREAM",
        {"vendor/ud-tools": "https://example.test/commit"},
    )
    monkeypatch.setattr(
        check_vendor_updates,
        "check_submodule",
        lambda *_args: _status(is_current=False),
    )

    assert check_vendor_updates.main(["--no-network", "--json", "--fail-on-stale"]) == 1

    data = json.loads(capsys.readouterr().out)
    assert data[0]["is_current"] is False


def test_vendor_freshness_workflow_keeps_json_artifact_parseable() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    run_step = workflow[
        workflow.index("python scripts/check_vendor_updates.py") : workflow.index(
            'STALE=$(python -c "'
        )
    ]

    assert "--json > vendor_status.json" in run_step
    assert "2>&1" not in run_step
    assert "|| true" not in run_step
