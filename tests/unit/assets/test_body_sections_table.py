"""The committed body section table and the settings it was built from (#15299)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BODY = Path(__file__).resolve().parents[3] / "assets" / "body_cc0"
TABLE = json.loads((BODY / "body_sections.json").read_text(encoding="utf-8"))
SETTINGS = BODY / "section_settings.json"
HEAD = BODY.parent / "head_cc0" / "PROVENANCE.json"
SEGMENTS = {"torso", "neck", "upper_arm", "forearm", "thigh", "shank"}
MIN_TORSO_STATIONS = 12

pytestmark = pytest.mark.unit


def test_settings_name_every_segment_the_lofts_need() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert set(settings["segments"]) == SEGMENTS
    assert settings["segments"]["torso"]["stations"] >= MIN_TORSO_STATIONS
    assert settings["mixes"]["neutral"] is None
    assert set(settings["mixes"]["male_athletic"]) >= {"male", "muscle", "weight"}
    assert not any(Path(rel).is_absolute() for rel in settings["source_files"])


def test_table_records_its_settings_and_sources() -> None:
    assert TABLE["licence"] == "CC0-1.0"
    assert TABLE["settings_sha256"] == hashlib.sha256(SETTINGS.read_bytes()).hexdigest()
    head = json.loads(HEAD.read_text(encoding="utf-8"))
    assert TABLE["source_commit"] == head["source_commit"]
    by_path = {source["path"]: source["sha256"] for source in TABLE["sources"]}
    assert by_path["makehuman/data/3dobjs/base.obj"] == head["source_sha256"]
    assert len(by_path) == 3 and all(len(sha) == 64 for sha in by_path.values())
    assert TABLE["mixes"]["neutral"]["targets"] == []
    targets = TABLE["mixes"]["male_athletic"]["targets"]
    assert targets and all(len(target["sha256"]) == 64 for target in targets)


@pytest.mark.parametrize("mix", ["neutral", "male_athletic"])
def test_every_segment_is_normalised_the_way_a_loft_reads_it(mix: str) -> None:
    measured = TABLE["mixes"][mix]
    assert set(measured["segments"]) == SEGMENTS
    assert len(measured["segments"]["torso"]["rows"]) >= MIN_TORSO_STATIONS
    width, depth = (TABLE["loft_columns"].index(key) for key in ("width", "depth"))
    columns = len(TABLE["loft_columns"]) + len(TABLE["size_columns"])
    for segment in measured["segments"].values():
        rows = segment["rows"]
        assert all(len(row) == columns for row in rows)
        assert max(row[width] for row in rows) == max(row[depth] for row in rows) == 1
        assert [row[0] for row in rows] == sorted(row[0] for row in rows)
        assert all(value > 0.0 for row in rows for value in row[-4:])
    assert measured["shoulder"]["trapezius_slope_deg"] > 0.0
    assert measured["head"]["menton_to_vertex_mm"] > measured["head"]["breadth_mm"]


def test_table_rebuilds_from_the_pinned_makehuman_tree() -> None:
    """Needs the pinned MakeHuman tree: set ``MAKEHUMAN_ROOT`` to its checkout."""
    root = os.environ.get("MAKEHUMAN_ROOT", "")
    if not root or not (Path(root) / TABLE["sources"][0]["path"]).is_file():
        pytest.skip("MAKEHUMAN_ROOT does not point at a MakeHuman checkout")
    script = str(BODY / "build_body_sections.py")
    done = subprocess.run(
        [sys.executable, script, root, "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
