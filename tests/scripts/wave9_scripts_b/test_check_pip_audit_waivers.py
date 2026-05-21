"""Tests for scripts/ci/check_pip_audit_waivers.py."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.ci import check_pip_audit_waivers as mod


def _write_manifest(path: Path, *, waivers: list[dict] | None = None) -> Path:
    body = {"schema_version": 1, "waivers": waivers or []}
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _waiver_dict(**over: object) -> dict:
    base = {
        "vuln": "GHSA-xxx",
        "package": "requests",
        "reason": "no fix yet",
        "tracked_in": "#1234",
        "expires_on": (date.today() + timedelta(days=30)).isoformat(),
    }
    base.update(over)
    return base


def test_load_waivers_success(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "w.json", waivers=[_waiver_dict()])
    waivers = mod.load_waivers(manifest)
    assert len(waivers) == 1
    assert waivers[0].vuln == "GHSA-xxx"
    assert waivers[0].package == "requests"


def test_load_waivers_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_waivers(tmp_path / "nope.json")


def test_load_waivers_not_object(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod.load_waivers(p)


def test_load_waivers_bad_schema(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        mod.load_waivers(p)


def test_load_waivers_missing_field(tmp_path: Path) -> None:
    bad = _waiver_dict()
    del bad["reason"]
    p = _write_manifest(tmp_path / "w.json", waivers=[bad])
    with pytest.raises(ValueError, match="missing waiver field"):
        mod.load_waivers(p)


def test_load_waivers_bad_date(tmp_path: Path) -> None:
    p = _write_manifest(
        tmp_path / "w.json", waivers=[_waiver_dict(expires_on="not-a-date")]
    )
    with pytest.raises(ValueError, match="ISO date"):
        mod.load_waivers(p)


def test_load_waivers_bad_tracked_in(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path / "w.json", waivers=[_waiver_dict(tracked_in="1234")])
    with pytest.raises(ValueError, match="tracked_in"):
        mod.load_waivers(p)


def test_load_waivers_entry_not_dict(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"schema_version": 1, "waivers": ["x"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        mod.load_waivers(p)


def test_load_waivers_list_required(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"schema_version": 1, "waivers": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        mod.load_waivers(p)


def test_find_expired_waivers() -> None:
    fresh = mod.Waiver("v1", "p", "r", "#1", date.today() + timedelta(days=1))
    old = mod.Waiver("v2", "p", "r", "#1", date.today() - timedelta(days=1))
    expired = mod.find_expired_waivers([fresh, old])
    assert expired == [old]


def test_load_reported_vulns(tmp_path: Path) -> None:
    rpt = tmp_path / "rpt.json"
    rpt.write_text(
        json.dumps(
            {
                "dependencies": [
                    {"name": "requests", "vulns": [{"id": "GHSA-xxx"}]},
                    {"name": "Pkg", "vulns": [{"id": "GHSA-yyy"}, "bad"]},
                    "not-a-dict",
                    {"name": "", "vulns": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = mod.load_reported_vulns(rpt)
    assert ("requests", "GHSA-xxx") in out
    assert ("pkg", "GHSA-yyy") in out


def test_load_reported_vulns_bad_shape(tmp_path: Path) -> None:
    rpt = tmp_path / "rpt.json"
    rpt.write_text(json.dumps({"dependencies": "no"}), encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_reported_vulns(rpt)


def test_find_stale_waivers() -> None:
    w = mod.Waiver("v1", "Pkg", "r", "#1", date.today())
    reported: set[tuple[str, str]] = {("pkg", "v1")}
    assert mod.find_stale_waivers([w], reported) == []
    assert mod.find_stale_waivers([w], set()) == [w]


def test_build_ignore_flags() -> None:
    w1 = mod.Waiver("v1", "a", "r", "#1", date.today())
    w2 = mod.Waiver("v2", "b", "r", "#2", date.today())
    assert mod.build_ignore_flags([w1, w2]) == [
        "--ignore-vuln",
        "v1",
        "--ignore-vuln",
        "v2",
    ]


def test_main_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _write_manifest(tmp_path / "w.json", waivers=[_waiver_dict()])
    monkeypatch.setattr("sys.argv", ["x", "--waiver-file", str(p)])
    assert mod.main() == 0
    assert "--ignore-vuln" in capsys.readouterr().out


def test_main_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _waiver_dict(expires_on=(date.today() - timedelta(days=1)).isoformat())
    p = _write_manifest(tmp_path / "w.json", waivers=[bad])
    monkeypatch.setattr("sys.argv", ["x", "--waiver-file", str(p)])
    assert mod.main() == 1
    assert "Expired" in capsys.readouterr().err


def test_main_stale_with_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _write_manifest(tmp_path / "w.json", waivers=[_waiver_dict()])
    rpt = tmp_path / "rpt.json"
    rpt.write_text(json.dumps({"dependencies": []}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--waiver-file", str(p), "--audit-report", str(rpt)],
    )
    assert mod.main() == 1
    assert "Stale" in capsys.readouterr().err
