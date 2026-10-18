"""The same colour picks give identical material RGBA in every writer (HRG-B2, #13167).

Writers compared here: the CLI-resolved worker job, ``resolve_appearance``, the
desktop ``CharacterBuilderModel`` and the web API service. The TypeScript
serializer is held to the same JSON fixture by ``appearanceDocument.test.ts``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.services.character_appearance_service import (
    appearance_library,
    build_appearance,
)
from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    resolve_appearance,
)
from src.shared.python.model_appearance.schema import document_from_dict
from src.tools.character_builder.core import CharacterBuilderModel
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.backends._worker_job import WorkerJob

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/appearance/colour_roundtrip.json"
    ).read_text(encoding="utf-8")
)
CASES = FIXTURE["cases"]
_FLAGS = {
    "skin_color": "--skin-tone",
    "hair_color": "--hair",
    "top_color": "--top-color",
    "bottom_color": "--bottom-color",
}


def _materials(doc: dict) -> dict[str, list[float]]:
    return {k: list(v["base_color"]) for k, v in sorted(doc["materials"].items())}


def _summary(doc: dict) -> dict:
    return {
        "skin_tone": doc["skin_tone"],
        "clothing": doc["clothing"],
        "headwear_material": doc["head"].get("headwear_material"),
        "materials": _materials(doc),
    }


def _cli_document(picks: dict, tmp_path: Path) -> dict:
    argv = ["--bundle", "b", "--out", "o", "--swing", "s"]
    argv += ["--clothing", picks["clothing"]]
    if picks.get("skin_tone") and "skin_color" not in picks:
        argv += ["--skin-tone", picks["skin_tone"]]
    for key, flag in _FLAGS.items():
        if key in picks:
            argv += [flag, picks[key]]
    settings = cli.build_settings(cli.build_parser().parse_args(argv))
    job = WorkerJob(
        "b",
        "q",
        [0],
        ["v"],
        1,
        1,
        [0.0, 0.0, 0.0],
        None,
        "o",
        appearance=settings.appearance,
    )
    job.dump(
        tmp_path / "job.json"
    )  # through the worker job file, as the worker reads it
    return WorkerJob.load(tmp_path / "job.json").appearance


def _desktop_document(picks: dict) -> dict:
    model = CharacterBuilderModel()
    for key, value in picks.items():
        model.set_appearance(key, value)
    return model.appearance_document()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_writer_matches_the_shared_fixture(case: dict, tmp_path: Path) -> None:
    picks, expected = case["picks"], case["expected"]
    documents = {
        "web api service": build_appearance(picks, None),
        "desktop model": _desktop_document(picks),
        "cli worker job": _cli_document(picks, tmp_path),
    }
    for writer, doc in documents.items():
        assert _summary(doc) == expected, writer
        document_from_dict(doc)  # validates


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_resolved_rgba_is_identical_across_writers(case: dict, tmp_path: Path) -> None:
    picks = case["picks"]
    rgba = []
    for doc in (
        build_appearance(picks, None),
        _desktop_document(picks),
        _cli_document(picks, tmp_path),
    ):
        parsed = document_from_dict(doc)
        mats = library.library_materials(parsed)
        garments = library.CLOTHING[parsed.clothing]
        rgba.append(
            [
                mats[parsed.skin_tone].rgba,
                mats[library.headwear_material_name(parsed)].rgba,
                mats[garments["torso"]].rgba,
                mats[garments[library.PANTS_PART]].rgba,
            ]
        )
    assert rgba[0] == rgba[1] == rgba[2]


def test_fixture_library_subset_is_the_real_library() -> None:
    real = appearance_library()
    sub = FIXTURE["library"]
    assert sub["clothing"] == {k: real["clothing"][k] for k in sub["clothing"]}
    assert sub["headwear_default_material"] == real["headwear_default_material"]
    assert sub["materials"] == {k: real["materials"][k] for k in sub["materials"]}


def test_resolve_appearance_equals_the_writers() -> None:
    case = CASES[-1]
    picks = case["picks"]
    doc = resolve_appearance(
        AppearanceOverrides(
            skin_tone=picks["skin_color"],
            hair=picks["hair_color"],
            clothing=picks["clothing"],
            top_color=picks["top_color"],
            bottom_color=picks["bottom_color"],
        ),
        None,
    )
    assert _summary(doc) == case["expected"]
