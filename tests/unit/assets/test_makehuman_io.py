"""Shared MakeHuman reader: rig, base mesh groups and morph targets (#15299)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(__file__).resolve().parents[3] / "assets"
sys.path.insert(0, str(ASSETS / "makehuman_cc0"))

import mh_rig_io  # noqa: E402
import mh_targets  # noqa: E402

pytestmark = pytest.mark.unit

CC0_HEADER = "# This asset was explicitly released as CC0 in september 2020.\n"
OBJ = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 2 0
v 1 2 0
g body
f 1 2 3 4
g joint-top
f 4/1 3/2 6/3 5/4
"""


def _tree(tmp_path: Path, licence: str = "CC0") -> Path:
    data = tmp_path / "makehuman" / "data"
    (data / "3dobjs").mkdir(parents=True)
    (data / "rigs").mkdir()
    (data / "3dobjs" / "base.obj").write_text(OBJ, encoding="utf-8")
    skeleton = {
        "license": licence,
        "bones": {"root": {"head": "base", "tail": "top", "parent": None}},
        "joints": {"base": [0, 1], "top": [4, 5]},
    }
    weights = {"license": "CC0", "weights": {"root": [[0, 1.0], [2, 0.25]]}}
    (data / "rigs" / "default.mhskel").write_text(json.dumps(skeleton), "utf-8")
    (data / "rigs" / "default_weights.mhw").write_text(json.dumps(weights), "utf-8")
    return tmp_path


def _target(tree: Path, name: str, body: str, header: str = CC0_HEADER) -> Path:
    path = tree / "makehuman" / "data" / "targets" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    return path


def test_rig_reads_mesh_groups_joints_and_weights(tmp_path: Path) -> None:
    rig = mh_rig_io.load_makehuman_rig(_tree(tmp_path))
    assert rig.vertices.shape == (6, 3)
    assert rig.body_quads.tolist() == [[0, 1, 2, 3]]
    np.testing.assert_allclose(rig.bone_head("root"), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(rig.bone_tail("root"), [0.5, 2.0, 0.0])
    assert rig.bone_parent("root") is None
    dense = mh_rig_io.dense_weights(rig, ("root", "absent"))
    assert dense[:, 0].tolist() == [1.0, 0.0, 0.25, 0.0, 0.0, 0.0]
    assert not dense[:, 1].any()
    with pytest.raises(ValueError, match="no bone"):
        rig.bone_head("femur")


def test_groups_reader_keeps_every_group_and_face_size(tmp_path: Path) -> None:
    obj = _tree(tmp_path) / "makehuman" / "data" / "3dobjs" / "base.obj"
    vertices, groups = mh_rig_io.read_obj_groups(obj)
    assert vertices.shape == (6, 3)
    assert groups == {"body": [[0, 1, 2, 3]], "joint-top": [[3, 2, 5, 4]]}


def test_rig_fails_closed_on_licence_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not marked CC0"):
        mh_rig_io.load_makehuman_rig(_tree(tmp_path / "a", licence="AGPL"))
    with pytest.raises(ValueError, match="missing MakeHuman input"):
        mh_rig_io.load_makehuman_rig(tmp_path / "nowhere")


def test_target_reader_and_weighted_application(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    grow = mh_targets.read_target(_target(tree, "grow.target", "1 0.5 0 0\n3 0 1 0\n"))
    assert grow.indices.tolist() == [1, 3]
    rig = mh_rig_io.load_makehuman_rig(tree)
    moved = mh_targets.apply_targets(rig.vertices, [(grow, 0.5), (grow, 0.25)])
    np.testing.assert_allclose(moved[1], [1.375, 0.0, 0.0])
    np.testing.assert_allclose(moved[3], [0.0, 1.75, 0.0])
    np.testing.assert_array_equal(moved[[0, 2, 4, 5]], rig.vertices[[0, 2, 4, 5]])
    assert moved is not rig.vertices and rig.vertices[1, 0] == 1.0
    still = mh_targets.read_target(_target(tree, "average.target", ""))
    assert still.deltas.shape == (0, 3)
    np.testing.assert_array_equal(
        mh_targets.apply_targets(rig.vertices, [(still, 1.0)]), rig.vertices
    )


@pytest.mark.parametrize(
    ("body", "header", "message"),
    [
        ("1 0.5 0 0\n", "# no licence here\n", "not marked CC0"),
        ("1 0.5 0\n", CC0_HEADER, "malformed"),
        ("1 0 0 0\n1 0 0 0\n", CC0_HEADER, "repeats"),
    ],
)
def test_target_reader_contract(
    tmp_path: Path, body: str, header: str, message: str
) -> None:
    path = _target(_tree(tmp_path), "bad.target", body, header)
    with pytest.raises(ValueError, match=message):
        mh_targets.read_target(path)


def test_apply_targets_contract(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    far = mh_targets.read_target(_target(tree, "far.target", "9 1 0 0\n"))
    with pytest.raises(ValueError, match="outside the mesh"):
        mh_targets.apply_targets(np.zeros((6, 3)), [(far, 1.0)])
    with pytest.raises(ValueError, match="finite"):
        mh_targets.apply_targets(np.zeros((12, 3)), [(far, float("nan"))])


def test_macro_mix_is_the_product_of_the_macro_factors() -> None:
    mix = mh_targets.macro_mix(
        {
            "male": 1.0,
            "muscle": 0.75,
            "weight": 0.5,
            "height": 0.75,
            "ethnic": {"african": 0.25, "caucasian": 0.75},
        }
    )
    root = "macrodetails"
    assert mix == {
        f"{root}/african-male-young.target": 0.25,
        f"{root}/caucasian-male-young.target": 0.75,
        f"{root}/universal-male-young-averagemuscle-averageweight.target": 0.5,
        f"{root}/universal-male-young-maxmuscle-averageweight.target": 0.5,
        f"{root}/height/male-young-averagemuscle-averageweight-maxheight.target": 0.25,
        f"{root}/height/male-young-maxmuscle-averageweight-maxheight.target": 0.25,
    }
    both = mh_targets.macro_mix(
        {"male": 0.5, "muscle": 0.5, "weight": 0.25, "height": 0.5, "ethnic": {}}
    )
    assert both[f"{root}/universal-female-young-averagemuscle-minweight.target"] == 0.25
    assert sum(both.values()) == pytest.approx(1.0)
    for bad in ({"male": 1.5}, {"muscle": -0.1}, {"ethnic": {"african": 0.7}}):
        settings = {"male": 1.0, "muscle": 0.5, "weight": 0.5, "height": 0.5}
        with pytest.raises(ValueError):
            mh_targets.macro_mix({**settings, "ethnic": {}, **bad})


def test_morphed_rig_moves_skin_and_joints_and_records_hashes(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    _target(tree, "macrodetails/lift.target", "4 0 1 0\n5 0 1 0\n")
    rig = mh_rig_io.load_makehuman_rig(tree)
    morphed, used = mh_targets.morphed_rig(rig, tree, {"macrodetails/lift.target": 0.5})
    np.testing.assert_allclose(morphed.bone_tail("root"), [0.5, 2.5, 0.0])
    np.testing.assert_allclose(rig.bone_tail("root"), [0.5, 2.0, 0.0])
    path = "makehuman/data/targets/macrodetails/lift.target"
    assert [(u["path"], u["weight"]) for u in used] == [(path, 0.5)]
    assert used[0]["sha256"] == mh_rig_io.sha256_file(tree / path)
    with pytest.raises(ValueError, match="missing MakeHuman target"):
        mh_targets.morphed_rig(rig, tree, {"macrodetails/none.target": 1.0})


def test_hand_and_head_assets_rebuild_from_the_shared_reader() -> None:
    """Needs the pinned MakeHuman tree: set ``MAKEHUMAN_ROOT`` to its checkout."""
    root = os.environ.get("MAKEHUMAN_ROOT", "")
    base = Path(root) / "makehuman" / "data" / "3dobjs" / "base.obj"
    if not root or not base.is_file():
        pytest.skip("MAKEHUMAN_ROOT does not point at a MakeHuman checkout")
    for script, source in (
        (ASSETS / "hand_cc0" / "build_hand_asset.py", Path(root)),
        (ASSETS / "head_cc0" / "build_head_asset.py", base),
    ):
        done = subprocess.run(
            [sys.executable, str(script), str(source), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
