"""Marker selection of a pipeline run: document leg seeds and dropped markers (#15296)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.shared.python.motion_matching.execution import spec_builder
from src.shared.python.motion_matching.execution.subject_preset import (
    load_subject_preset,
)
from src.shared.python.motion_matching.pipeline import cli
from src.shared.python.motion_matching.pipeline.constants import (
    LEG_SEEDS,
    SPEC,
    square_forefoot_seeds,
)
from src.shared.python.motion_matching.pipeline.lane import Lane
from src.shared.python.motion_matching.pipeline.marker_selection import (
    LEG_SEED_SOURCES,
    base_leg_seeds,
    select_markers,
)
from src.shared.python.motion_matching.tour_capture_contract import TOUR_CAPTURE

pytestmark = [pytest.mark.unit]
CAPTURE_LABELS = TOUR_CAPTURE.labels
DROPPED = ["Marker_0:0:0", "Uname*36", "Uname*37", "Uname*38"]


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def preset() -> dict[str, Any]:
    kwargs = load_subject_preset("tour_average_driver").builder_kwargs()
    return spec_builder.compose_anthropometric_document(**kwargs).document


def _legacy(document: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The attachments and labels ``calibrate_run`` used before #15296."""
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in document["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    return upper, tuple({**upper, **LEG_SEEDS})


def test_a_document_without_leg_seeds_selects_what_it_always_did(committed) -> None:
    upper, labels = _legacy(committed)
    for source in LEG_SEED_SOURCES:
        chosen = select_markers(committed, CAPTURE_LABELS, source)
        assert chosen.attachments == upper
        assert list(chosen.attachments) == list(upper)
        assert chosen.labels == labels
        assert chosen.leg_seeds is None and chosen.dropped == ()
        assert chosen.report() == {}


def test_preset_leg_seeds_win_over_the_built_in_ones(preset) -> None:
    chosen = select_markers(preset, CAPTURE_LABELS)
    assert chosen.leg_seeds is not None and set(chosen.leg_seeds) == set(LEG_SEEDS)
    for label, (body, offset) in chosen.leg_seeds.items():
        carried = preset["marker_attachments"][label]
        assert (body, list(offset)) == (carried["body"], carried["offset_m"])
        assert body == LEG_SEEDS[label][0]
    assert chosen.leg_seeds["RKneeOut"][1] != LEG_SEEDS["RKneeOut"][1]
    assert chosen.report() == {"leg_seeds": "document", "dropped_markers": DROPPED}
    assert select_markers(preset, CAPTURE_LABELS, "builtin").leg_seeds is None


def test_dropped_markers_never_reach_the_fit(preset, committed) -> None:
    chosen = select_markers(preset, CAPTURE_LABELS)
    assert chosen.dropped == tuple(DROPPED)
    assert not set(DROPPED) & (set(chosen.labels) | set(chosen.attachments))
    assert len(chosen.labels) == 34
    carried = copy.deepcopy(committed)  # a label the document still attaches
    carried["subject"] = {"dropped_markers": ["HeadSide"]}
    chosen = select_markers(carried, CAPTURE_LABELS)
    assert "HeadSide" not in chosen.labels and "HeadSide" not in chosen.attachments
    assert len(chosen.labels) == len(_legacy(committed)[1]) - 1


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d["subject"].update(dropped_markers=["Nose"]), "capture.*Nose"),
        (lambda d: d["subject"].update(dropped_markers="HeadTop"), "list of labels"),
        (lambda d: d["subject"].update(dropped_markers=["RToeIn"]), "needs.*RToeIn"),
        (lambda d: d["marker_attachments"]["LToeIn"].update(offset_m=None), "LToeIn"),
    ],
)
def test_a_selection_the_fit_cannot_honour_is_refused(preset, edit, match) -> None:
    document = copy.deepcopy(preset)
    edit(document)
    with pytest.raises(ValueError, match=match):
        select_markers(document, CAPTURE_LABELS)


@pytest.mark.parametrize(
    "raw",
    [{"HeadSide": False}, {}, 0, False, "[]", "", ("HeadSide",)],
    ids=["dict", "empty-dict", "int", "bool", "stringified-list", "empty-str", "tuple"],
)
def test_dropped_markers_must_be_a_list_not_a_falsy_or_iterable_stand_in(
    committed, raw
) -> None:
    """#15342: only a list (or an omitted / ``None`` value) is a label list."""
    document = copy.deepcopy(committed)
    document["subject"] = {"dropped_markers": raw}
    with pytest.raises(ValueError, match="list of labels"):
        select_markers(document, CAPTURE_LABELS)


@pytest.mark.parametrize(
    "subject", [{}, {"dropped_markers": None}, {"dropped_markers": []}]
)
def test_an_omitted_none_or_empty_dropped_markers_selects_nothing(
    committed, subject
) -> None:
    document = copy.deepcopy(committed)
    document["subject"] = subject
    assert select_markers(document, CAPTURE_LABELS).dropped == ()


def test_the_leg_seed_source_is_a_checked_setting(preset) -> None:
    with pytest.raises(ValueError, match="leg seed source"):
        select_markers(preset, CAPTURE_LABELS, "spec")
    args = cli.build_parser().parse_args([])
    assert args.leg_seeds == "document" and LEG_SEED_SOURCES == ("document", "builtin")


def test_the_lane_starts_its_leg_calibration_from_the_selected_seeds(preset) -> None:
    lane = Lane.__new__(Lane)  # leg_seeds reads only these two attributes
    lane.feet, lane.document_leg_seeds = None, None
    assert lane.leg_seeds() == dict(LEG_SEEDS) == base_leg_seeds(lane)
    assert base_leg_seeds(SimpleNamespace()) == dict(LEG_SEEDS)
    seeds = select_markers(preset, CAPTURE_LABELS).leg_seeds
    lane.document_leg_seeds = seeds
    assert lane.leg_seeds() == seeds == base_leg_seeds(lane)
    lane.feet = object()  # foot progression on: the forefoot pair is squared
    assert lane.leg_seeds() == square_forefoot_seeds(seeds)


@pytest.mark.parametrize("name", ["committed", "preset"])
def test_calibrate_run_hands_the_selection_to_the_lane(
    name, request, tmp_path, monkeypatch
) -> None:
    document = request.getfixturevalue(name)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(document), encoding="utf-8")
    seen: dict[str, Any] = {}

    def calibrate(ctx, lane, base_spec, upper_base, upper, labels):
        seen.update(lane=lane, upper=upper, labels=labels)
        return SimpleNamespace(address_report={})

    monkeypatch.setattr(cli, "_calibrate_and_scale", calibrate)
    monkeypatch.setattr(cli, "get_plant", lambda engine, spec: object())
    argv = ["--spec", str(spec), "--out", str(tmp_path), "--foot-progression", "off"]
    run = cli.calibrate_run(cli.build_parser().parse_args(argv))
    chosen = select_markers(document, CAPTURE_LABELS)
    assert seen["labels"] == chosen.labels == seen["lane"].labels
    assert seen["upper"] == chosen.attachments
    assert seen["lane"].leg_seeds() == dict(chosen.leg_seeds or LEG_SEEDS)
    assert run.cal_res.address_report == (
        {"marker_selection": chosen.report()} if chosen.report() else {}
    )
    if name == "committed":
        assert (seen["upper"], seen["labels"]) == _legacy(document)
