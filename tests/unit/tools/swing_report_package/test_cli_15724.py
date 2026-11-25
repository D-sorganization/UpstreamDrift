"""Swing report package command line (#15724, SRP-5); fakes, nothing renders."""

from __future__ import annotations

import json

import pytest

from src.tools.swing_report_package import bundle_phases, cli, runner
from src.tools.swing_report_package.catalog import load_catalog
from src.tools.swing_report_package.phases import KeyPhase

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "unavailable_jobs", lambda jobs: {})
    path = tmp_path / "swing.npz"
    path.write_bytes(b"")
    return path


def _dry_run(capsys, bundle, *extra):
    argv = ["--bundle", str(bundle), "--out", str(bundle.parent / "o")]
    assert cli.main([*argv, "--dry-run", *extra]) == 0
    return json.loads(capsys.readouterr().out)


CATALOG = load_catalog()
DEFAULT = set(CATALOG.default_package)
MIDHANDS = set(CATALOG.packages["midhands"])


def test_list_prints_the_items_and_packages(capsys) -> None:
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in (*CATALOG.items, "midhands: "))
    assert "default: " + ", ".join(CATALOG.default_package) in out


@pytest.mark.parametrize(
    ("extra", "items"),
    [
        ([], DEFAULT),
        (["--package", "midhands"], MIDHANDS),
        (
            ["--add", "midhands_grid", "--drop", "tip_to_tail"],
            DEFAULT - {"tip_to_tail"} | MIDHANDS - {"midhands_phase_sheet"},
        ),
    ],
)
def test_a_dry_run_prints_the_plan_and_writes_nothing(capsys, bundle, extra, items):
    manifest = _dry_run(capsys, bundle, *extra)
    assert {j["item"] for j in manifest["jobs"]} == items
    assert {j["status"] for j in manifest["jobs"]} == {"planned"}
    assert not (bundle.parent / "o").exists()


def test_a_custom_config(capsys, bundle, tmp_path) -> None:
    config = tmp_path / "custom.json"
    doc = {"default": False, "include": ["case_total"], "views": ["face_on"]}
    config.write_text(json.dumps(doc), encoding="utf-8")
    custom = _dry_run(capsys, bundle, "--config", str(config))
    assert [j["key"] for j in custom["jobs"]] == ["case_total@face_on@0.25x"]


@pytest.mark.parametrize(
    "extra",
    [["--add", "nope"], ["--package", "nope"], ["--config", "nope.json"], []],
)
def test_bad_input_is_an_argument_error(bundle, extra) -> None:
    file = bundle if extra else bundle.parent / "no.npz"  # [] = no bundle file
    with pytest.raises(SystemExit) as stop:
        cli.main(["--bundle", str(file), "--out", "o", "--dry-run", *extra])
    assert stop.value.code == 2
    with pytest.raises(SystemExit):  # no --bundle at all
        cli.main(["--out", "o"])


@pytest.mark.parametrize(("status", "code"), [("done", 0), ("failed", 1)])
def test_a_run_wires_the_renderer_and_phases(bundle, monkeypatch, status, code):
    seen = {}

    def fake_run(jobs, root, **kwargs):
        seen.update(kwargs, root=root)
        return {"jobs": [{"key": j.key, "status": status} for j in jobs]}

    monkeypatch.setattr(cli, "run_package", fake_run)
    out = bundle.parent / "o"
    assert cli.main(["--bundle", str(bundle), "--out", str(out)]) == code
    assert seen["root"] == out and seen["swing"] == "swing"
    assert callable(seen["render"]) and callable(seen["phases_of"])


def test_bundle_phases_detects_from_the_configured_bodies(monkeypatch) -> None:
    calls = {}

    def tracks(bundle, swing, club, bodies):
        calls["bodies"] = dict(bodies)
        return [0.0, 0.1, 0.2], {"clubhead": "c", **{k: k for k in bodies}}, 0.15

    def detect(t, points, impact, top):
        calls.update(points=points, impact=impact, top=top)
        return (KeyPhase("impact", "Impact", 1, impact),)

    monkeypatch.setattr(bundle_phases, "_body_tracks", tracks)
    monkeypatch.setattr(bundle_phases, "_top_time_s", lambda t, clubhead: 0.05)
    monkeypatch.setattr(bundle_phases, "detect_phases", detect)
    settings = {"phase_bodies": {"lead_shoulder": "ls", "lead_wrist": "lw"}}
    phases_of = bundle_phases.bundle_phases(
        "b.npz", "s", "Driver", impact_time_s=0.12, settings=settings
    )
    assert not calls  # lazy: nothing loads until the runner asks
    (phase,) = phases_of("s")
    assert calls["bodies"] == settings["phase_bodies"]
    assert calls["impact"] == pytest.approx(0.12) and calls["top"] == 0.05
    assert phase.time_s == pytest.approx(0.12)
    with pytest.raises(ValueError, match="phase_bodies"):
        bundle_phases.bundle_phases("b.npz", "s", "Driver", settings={})
