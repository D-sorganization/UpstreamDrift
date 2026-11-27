"""Swing report package window model (#15724): the pure argument building and
plan preview behind the launcher window; no Qt."""

from __future__ import annotations

import sys

import pytest

from src.tools.swing_report_package import cli, gui_model
from src.tools.swing_report_package.catalog import load_catalog

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

CATALOG = load_catalog()


def _choice(**kw):
    files = {"bundle": "b.npz", "out": "pkg", "swing": "s", "club": "D"}
    base = {**files, "package": "default", "checked": CATALOG.default_package}
    return gui_model.ReportChoice(**{**base, **kw})


def test_settings_load_and_a_missing_key_is_refused(tmp_path) -> None:
    settings = gui_model.load_gui_settings()
    assert settings["module"] == "src.tools.swing_report_package"
    (tmp_path / "s.json").write_text('{"window": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        gui_model.load_gui_settings(tmp_path / "s.json")


def test_rows_packages_and_starting_items_come_from_the_catalog() -> None:
    assert gui_model.package_names(CATALOG) == ("default", *CATALOG.packages)
    assert [r[0] for r in gui_model.item_rows(CATALOG)] == list(CATALOG.items)
    for name, items in {**CATALOG.packages, "default": CATALOG.default_package}.items():
        assert set(gui_model.starting_items(CATALOG, name)) == set(items)
    with pytest.raises(ValueError, match="unknown package"):
        gui_model.starting_items(CATALOG, "nope")


def test_ticks_become_add_and_drop_flags() -> None:
    checked = [n for n in CATALOG.default_package if n != "tip_to_tail"]
    args = gui_model.selection_args(CATALOG, "default", [*checked, "midhands_grid"])
    expected = ["--package", "default", "--add", "midhands_grid"]
    assert args == [*expected, "--drop", "tip_to_tail"]


def test_cli_args_round_trip_through_the_cli_parser() -> None:
    args = gui_model.cli_args(CATALOG, _choice(checked=("case_total",)))
    ns = cli.build_parser().parse_args(args)
    got = (str(ns.bundle), str(ns.out), ns.swing, ns.club)
    assert got == ("b.npz", "pkg", "s", "D")
    assert ns.package == "default" and "case_ztcf" in ns.drop
    assert "case_total" not in ns.drop


def test_cli_args_need_a_bundle_and_an_out_folder() -> None:
    with pytest.raises(ValueError, match="bundle"):
        gui_model.cli_args(CATALOG, _choice(bundle=" "))
    with pytest.raises(ValueError, match="output"):
        gui_model.cli_args(CATALOG, _choice(out=""))


def test_preview_is_the_planners_manifest_and_needs_no_files() -> None:
    midhands = CATALOG.packages["midhands"]
    choice = _choice(bundle="", out="", package="midhands", checked=midhands)
    manifest = gui_model.plan_preview(CATALOG, choice)
    assert {job["item"] for job in manifest["jobs"]} == set(midhands)
    lines = gui_model.preview_lines(manifest)
    assert len(lines) == len(manifest["jobs"]) + 1
    assert all(job["key"] in "\n".join(lines) for job in manifest["jobs"])


def test_command_runs_the_package_module() -> None:
    settings = gui_model.load_gui_settings()
    program, argv = gui_model.module_command(CATALOG, _choice(), settings)
    assert program == sys.executable
    assert argv == ["-m", settings["module"], *gui_model.cli_args(CATALOG, _choice())]
