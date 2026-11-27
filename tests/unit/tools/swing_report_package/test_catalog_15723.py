"""Swing report package catalog and planner (#15723, SRP-4).

A package is a default selection of catalog items (videos, grids, phase
sheets, charts) that the user can extend, trim and override item by item,
view by view and speed by speed. The planner turns a request into ordered
jobs, pulling in what a composite needs, and never renders anything itself.
"""

from __future__ import annotations

import json

import pytest

from src.tools.swing_report_package.catalog import (
    CATALOG_PATH,
    PackageRequest,
    load_catalog,
    plan_package,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _ids(jobs) -> list[str]:
    return [j.item for j in jobs]


def test_the_default_package_plans_its_items_with_composites_last() -> None:
    catalog = load_catalog()
    jobs = plan_package(catalog, PackageRequest())
    items = set(_ids(jobs))
    assert set(catalog.default_package) <= items
    order = _ids(jobs)
    for job in jobs:
        for need in job.needs:
            assert any(
                order.index(j.item) < order.index(job.item)
                for j in jobs
                if j.key == need
            )
    # one video job per item, view and speed
    videos = [j for j in jobs if j.kind == "video"]
    assert len({j.key for j in videos}) == len(videos)


def test_a_composite_pulls_in_the_renders_it_needs() -> None:
    catalog = load_catalog()
    jobs = plan_package(catalog, PackageRequest(include=("case_grid",), default=False))
    assert {"case_ztcf", "case_zvcf", "case_total", "case_grid"} == set(_ids(jobs))


def test_views_speeds_and_overrides_are_the_users() -> None:
    catalog = load_catalog()
    request = PackageRequest(
        include=("midhands_total",),
        default=False,
        views=("down_the_line",),
        speeds=(0.5,),
        render_overrides={"size": "3840x2160"},
        item_overrides={"midhands_total": {"ztcf_arrow_gain": 6}},
    )
    (job,) = plan_package(catalog, request)
    assert job.view == "down_the_line" and job.speed == 0.5
    assert job.args["size"] == "3840x2160"
    assert job.args["ztcf_arrow_gain"] == 6
    assert job.args["ztcf_zvcf_hands"] == "midhands"
    argv = job.exporter_argv()
    assert argv[argv.index("--views") + 1] == "down_the_line"
    assert argv[argv.index("--speeds") + 1] == "0.5"
    assert "--ztcf-zvcf-hand-force" in argv  # a true flag is passed bare


def test_exclude_removes_and_unknown_items_are_refused() -> None:
    catalog = load_catalog()
    jobs = plan_package(catalog, PackageRequest(exclude=("tip_to_tail",)))
    assert "tip_to_tail" not in _ids(jobs)
    with pytest.raises(ValueError, match="unknown"):
        plan_package(catalog, PackageRequest(include=("teleport",)))
    with pytest.raises(ValueError, match="override"):
        plan_package(catalog, PackageRequest(item_overrides={"nope": {}}))


def test_a_request_file_round_trips(tmp_path) -> None:
    path = tmp_path / "my_package.json"
    path.write_text(
        json.dumps({"include": ["case_grid"], "default": False, "views": ["face_on"]}),
        encoding="utf-8",
    )
    request = PackageRequest.from_file(path)
    assert request.include == ("case_grid",) and not request.default
    with pytest.raises(ValueError, match="key"):
        path.write_text(json.dumps({"inclde": []}), encoding="utf-8")
        PackageRequest.from_file(path)


def test_the_catalog_is_checked(tmp_path) -> None:
    doc = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    doc["items"]["case_grid"]["needs"] = ["ghost"]
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="ghost"):
        load_catalog(bad)
