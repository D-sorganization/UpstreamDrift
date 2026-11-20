"""The continuous CC0 body skin asset: provenance, shape and fit bounds (#15299)."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FOLDER = Path(__file__).resolve().parents[3] / "assets" / "body_skin_cc0"
SCRIPT = FOLDER / "build_body_skin.py"
SETTINGS: dict[str, Any] = json.loads((FOLDER / "skin_settings.json").read_text())
PROVENANCE: dict[str, Any] = json.loads((FOLDER / "PROVENANCE.json").read_text())
ASSET = FOLDER / PROVENANCE["asset"]


def _arrays() -> dict[str, np.ndarray]:
    with np.load(ASSET) as data:
        return {key: data[key] for key in data.files}


def _landmarks_m() -> dict[str, np.ndarray]:
    points = _arrays()["landmarks"] * SETTINGS["quantum_m"]
    return dict(zip(SETTINGS["landmarks"], points, strict=True))


def test_provenance_matches_the_committed_asset() -> None:
    digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    assert PROVENANCE["asset_sha256"] == digest
    assert PROVENANCE["licence"] == "CC0-1.0"
    assert PROVENANCE["source_commit"] == SETTINGS["source_commit"]
    assert sorted(PROVENANCE["source_sha256"]) == sorted(SETTINGS["source_files"])
    for name in (PROVENANCE["licence_file"], PROVENANCE["licence_statement_file"]):
        assert (FOLDER / name).resolve().is_file()
    assert (FOLDER / "NOTICE").is_file()


def test_no_absolute_or_private_paths_are_recorded() -> None:
    for name in ("PROVENANCE.json", "skin_settings.json", "NOTICE", SCRIPT.name):
        text = (FOLDER / name).read_text(encoding="utf-8")
        for marker in ("/home/", "/tmp/", "/Users/", ":\\", "~/"):
            assert marker not in text, f"{name} holds {marker!r}"
    for entry in PROVENANCE["morph_targets"]:
        assert not Path(entry["path"]).is_absolute()


def test_mesh_is_one_connected_surface_within_budget() -> None:
    arrays = _arrays()
    faces = arrays["faces"].astype(np.int64)
    count = len(arrays["vertices"])
    assert len(faces) <= SETTINGS["budget"]["max_triangles"]
    assert ASSET.stat().st_size <= SETTINGS["budget"]["max_asset_bytes"]
    assert np.array_equal(np.unique(faces), np.arange(count))
    parent = np.arange(count)
    for _ in range(64):  # label propagation; the mesh diameter is far below 64 hops
        low = parent[faces].min(axis=1)
        for corner in range(3):
            np.minimum.at(parent, faces[:, corner], low)
        parent = parent[parent]
    assert len(np.unique(parent)) == 1, "trunk and limbs must be one surface"


def test_weights_are_normalised_over_the_named_groups() -> None:
    arrays = _arrays()
    groups = PROVENANCE["groups"]
    assert arrays["weight_index"].shape[1] == SETTINGS["max_influences"]
    assert int(arrays["weight_index"].max()) < len(groups)
    assert set(np.unique(arrays["weight_index"])) == set(range(len(groups)))
    total = arrays["weight_value"].astype(int).sum(axis=1)
    assert np.all(total == SETTINGS["weight_levels"])


def test_landmarks_sit_inside_the_skin_and_are_symmetric() -> None:
    marks = _landmarks_m()
    vertices = _arrays()["vertices"] * SETTINGS["quantum_m"]
    assert np.allclose(0.5 * (marks["hip_l"] + marks["hip_r"]), 0.0, atol=1e-4)
    slack = SETTINGS["reference_fit"]["symmetry_mm"] / 1000.0
    for name, point in marks.items():
        if name.endswith("_l"):
            mirror = marks[name[:-2] + "_r"] * np.array([1.0, -1.0, 1.0])
            assert np.allclose(point, mirror, atol=slack), name
    assert vertices[:, 1].max() > marks["shoulder_l"][1] > marks["hip_l"][1] > 0.0
    assert np.ptp(vertices[:, 2]) < SETTINGS["stature_m"]


def test_fit_scales_to_the_reference_joint_centres_stay_in_bounds() -> None:
    marks = _landmarks_m()
    low, high = SETTINGS["reference_fit"]["scale_bounds"]
    for name, (a, b, model_mm) in SETTINGS["reference_fit"]["segments_mm"].items():
        stored_mm = 1000.0 * float(np.linalg.norm(marks[a] - marks[b]))
        assert low <= model_mm / stored_mm <= high, (name, stored_mm, model_mm)


@pytest.mark.skipif(
    not os.environ.get("MAKEHUMAN_ROOT"), reason="MAKEHUMAN_ROOT not set (opt-in)"
)
def test_rebuild_from_makehuman_matches_the_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stop.value.code == 0
