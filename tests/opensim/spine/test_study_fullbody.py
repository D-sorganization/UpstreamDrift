"""Full-body option of the spine study: request, receipt and stage (SPN-9c, #12758).

Every number is INVENTED.  The stage test uses the in-repo ``golf_humanoid``
model as BOTH the source and the target of an identity map, so the whole
coordinate-table, club-wrench and ground-reaction chain runs without any
licensed asset; the physical content of the Bruno map is covered by the
asset-gated tests in ``test_fullbody_licensed.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fullbody_loads import (
    POLICIES,
    LoadSet,
    loads_record,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    FullbodyMap,
    parse_fullbody_maps,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_study import (
    FullbodySpec,
    run_fullbody,
)
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyReceipt,
    SpineStudyRequest,
    request_fingerprint,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    load_receipt,
    receipt_to_json,
    write_receipt,
)
from src.engines.physics_engines.opensim.python.spine.study_stages import stage_names
from tests.helpers.opensim_support import require_opensim

REPO = Path(__file__).resolve().parents[3]
GOLF_MODEL = REPO / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
BRUNO = "bruno2015_fullbody_os4_v2"
SHA = "0123456789abcdef" * 4


def _ok(tmp_path: Path, **kw) -> SpineStudyRequest:
    base = {
        "study_id": "s",
        "motion_path": tmp_path / "m.sto",
        "golf_model_path": tmp_path / "g.osim",
        "model_path": tmp_path / "x.osim",
        "model_id": BRUNO,
        "output_dir": tmp_path,
    }
    base.update(kw)
    return SpineStudyRequest(**base)


# ------------------------------------------------------------ request contract


@pytest.mark.unit
def test_request_fields_validate(tmp_path: Path) -> None:
    ok = _ok(tmp_path, fullbody_map=BRUNO, external_loads_policy="club_and_grf")
    assert ok.fullbody_map == BRUNO
    assert POLICIES == ("none", "supplied", "grf_only", "club_and_grf")
    with pytest.raises(ValueError, match="unknown fullbody_map"):
        _ok(tmp_path, fullbody_map="nope", external_loads_policy="none")
    with pytest.raises(ValueError, match="external_loads_policy"):
        _ok(tmp_path, fullbody_map=BRUNO, external_loads_policy="everything")
    with pytest.raises(ValueError, match="requires a fullbody_map"):
        _ok(tmp_path, external_loads_policy="club_and_grf")
    with pytest.raises(ValueError, match="requires external_loads"):
        _ok(tmp_path, fullbody_map=BRUNO, external_loads_policy="supplied")
    with pytest.raises(ValueError, match="generates its own"):
        _ok(
            tmp_path,
            fullbody_map=BRUNO,
            external_loads_policy="grf_only",
            external_loads=tmp_path / "l.xml",
        )
    with pytest.raises(ValueError, match="movement variant"):
        _ok(
            tmp_path,
            fullbody_map=BRUNO,
            external_loads_policy="none",
            variant_id="x",
        )
    with pytest.raises(TypeError, match="fullbody_map"):
        _ok(tmp_path, fullbody_map=3)


@pytest.mark.unit
def test_policy_defaults_to_supplied_or_none(tmp_path: Path) -> None:
    plain = _ok(tmp_path, fullbody_map=BRUNO)
    assert plain.resolved_loads_policy == "none"
    supplied = _ok(tmp_path, fullbody_map=BRUNO, external_loads=tmp_path / "l.xml")
    assert supplied.resolved_loads_policy == "supplied"
    assert _ok(tmp_path).resolved_loads_policy == "none"


@pytest.mark.unit
def test_canonical_dict_is_unchanged_without_the_option(tmp_path: Path) -> None:
    plain = _ok(tmp_path).canonical_dict()
    assert "fullbody_map" not in plain and "external_loads_policy" not in plain
    full = _ok(
        tmp_path, fullbody_map=BRUNO, external_loads_policy="grf_only"
    ).canonical_dict()
    assert full["fullbody_map"] == BRUNO
    assert full["external_loads_policy"] == "grf_only"
    assert {k: v for k, v in full.items() if k in plain} == plain


@pytest.mark.unit
def test_stage_names_insert_the_fullbody_stage_after_the_drive(tmp_path: Path) -> None:
    base = stage_names(_ok(tmp_path))
    assert base[:3] == ("golf_input", "drive", "inverse_dynamics")
    full = stage_names(_ok(tmp_path, fullbody_map=BRUNO))
    assert full[:4] == ("golf_input", "drive", "fullbody", "inverse_dynamics")


# ------------------------------------------------------------- receipt record


@pytest.mark.unit
def test_loads_record_states_what_is_excluded() -> None:
    none = loads_record("none", None, map_id=BRUNO)
    assert none["policy"] == "none" and none["included"] == []
    items = {e["item"] for e in none["excluded"]}
    assert items == {"club_hand_wrench", "ground_reaction"}
    assert "gravity only" in none["statement"]

    foot = LoadSet(
        (_cp("heel_a_r", "calcn_r"),),
        np.zeros((3, 1, 3)),
        np.zeros((3, 1, 3)),
        ("ground_reaction",),
    )
    grf = loads_record("grf_only", foot, map_id=BRUNO)
    assert [e["item"] for e in grf["excluded"]] == ["club_hand_wrench"]
    assert "policy grf_only" in grf["excluded"][0]["reason"]
    assert "club excluded" in grf["statement"]
    assert grf["included"][0] == {
        "name": "heel_a_r",
        "body": "calcn_r",
        "kind": "ground_reaction",
        "frame": "ground",
    }

    hand = LoadSet(
        (_cp("hand_l", "hand_L"),),
        np.zeros((3, 1, 3)),
        np.zeros((3, 1, 3)),
        ("club_hand_wrench",),
    )
    full = loads_record("club_and_grf", _join(hand, foot), map_id=BRUNO)
    assert full["excluded"] == []
    assert {e["kind"] for e in full["included"]} == {
        "club_hand_wrench",
        "ground_reaction",
    }
    assert "club excluded" not in full["statement"]

    supplied = loads_record("supplied", None, map_id=BRUNO, supplied_name="l.xml")
    assert supplied["supplied_file"] == "l.xml"
    with pytest.raises(ValueError, match="policy"):
        loads_record("bogus", None, map_id=BRUNO)


def _cp(name: str, body: str):
    from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
        ContactPoint,
    )

    return ContactPoint(name, body, (0.0, 0.0, 0.0), "")


def _join(a: LoadSet, b: LoadSet) -> LoadSet:
    from src.engines.physics_engines.opensim.python.spine.fullbody_loads import (
        combine_load_sets,
    )

    return combine_load_sets(a, b)


def _receipt(fullbody: dict | None) -> SpineStudyReceipt:
    return SpineStudyReceipt(
        study_id="s",
        status="unqualified",
        status_reasons=("x",),
        request={"study_id": "s"},
        request_sha256=SHA,
        inputs=(FileRecord("motion", "m.sto", SHA),),
        code_version="0" * 40,
        opensim_version="v",
        scipy_version="1",
        numpy_version="1",
        law="n/a",
        distribution_source="n/a",
        asset=None,
        stages=(),
        artifacts=(),
        residual={},
        reserve={},
        context_comparisons=(),
        fullbody=fullbody,
    )


@pytest.mark.unit
def test_receipt_carries_the_fullbody_record_and_omits_it_when_absent(
    tmp_path: Path,
) -> None:
    record = {"policy": "grf_only", "statement": "club excluded", "included": []}
    path = write_receipt(_receipt(record), tmp_path / "r.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fullbody"] == record
    assert load_receipt(path).fullbody == record
    assert '"fullbody"' not in receipt_to_json(_receipt(None))
    assert (
        load_receipt(write_receipt(_receipt(None), tmp_path / "n.json")).fullbody
        is None
    )


@pytest.mark.unit
def test_fingerprint_depends_on_the_option(tmp_path: Path) -> None:
    for name in ("m.sto", "g.osim", "x.osim"):
        (tmp_path / name).write_text("x")
    plain = request_fingerprint(_ok(tmp_path))
    other = request_fingerprint(_ok(tmp_path, fullbody_map=BRUNO))
    again = request_fingerprint(_ok(tmp_path, fullbody_map=BRUNO))
    assert plain != other and other == again


# ------------------------------------------------- identity-map stage on golf


def _identity_map() -> FullbodyMap:
    """golf_humanoid -> golf_humanoid: invented identity map (no licensed data)."""
    osim = require_opensim()
    model = osim.Model(str(GOLF_MODEL))
    names = [
        model.getCoordinateSet().get(i).getName()
        for i in range(model.getNumCoordinates())
    ]
    golf, targets = (
        [],
        {
            k: []
            for k in (
                "driven_by_pelvis_pose",
                "driven_by_spine_drive",
                "driven_direct",
                "driven_orientation_match",
                "held_neutral",
                "locked_in_asset",
            )
        },
    )
    for name in names:
        if name.startswith("pelvis_"):
            golf.append(
                {
                    "golf": name,
                    "treatment": "pelvis_pose",
                    "target": name,
                    "sign": 1,
                    "min_axis_dot": 0.9,
                }
            )
            targets["driven_by_pelvis_pose"].append(name)
        elif name.startswith("lumbar_"):
            golf.append({"golf": name, "treatment": "spine_drive", "reason": "trunk"})
            targets["driven_by_spine_drive"].append(name)
        elif name.startswith("arm_"):
            golf.append(
                {
                    "golf": name,
                    "treatment": "orientation_match",
                    "group": f"shoulder_{name[-1]}",
                }
            )
            targets["driven_orientation_match"].append(name)
        elif (
            name.startswith(
                ("hip_", "knee_angle_r", "knee_angle_l", "ankle_", "elbow_")
            )
            and "beta" not in name
        ):
            golf.append(
                {
                    "golf": name,
                    "treatment": "direct",
                    "target": name,
                    "sign": 1,
                    "min_axis_dot": 0.9,
                }
            )
            targets["driven_direct"].append(name)
        else:
            golf.append(
                {"golf": name, "treatment": "excluded", "reason": "identity test map"}
            )
            targets["held_neutral"].append(name)
    raw = {
        "schema_version": 1,
        "maps": {
            "golf_identity": {
                "source_model": "golf_humanoid",
                "verified": True,
                "names_status": "synthetic identity",
                "orientation_groups": {
                    f"shoulder_{s}": {
                        "golf_body": f"humerus_{s}",
                        "golf_coordinates": [
                            f"arm_flex_{s}",
                            f"arm_add_{s}",
                            f"arm_rot_{s}",
                        ],
                        "target_body": f"humerus_{s}",
                        "target_coordinates": [
                            f"arm_flex_{s}",
                            f"arm_add_{s}",
                            f"arm_rot_{s}",
                        ],
                        "reference": "world",
                    }
                    for s in "rl"
                },
                "hand_bodies": {"hand_r": "hand_r", "hand_l": "hand_l"},
                "foot_bodies": ["calcn_r", "toes_r", "calcn_l", "toes_l"],
                "golf_coordinates": golf,
                "target_coordinates": targets,
            }
        },
    }
    return parse_fullbody_maps(raw)["golf_identity"]


def _golf_motion(path: Path, n: int = 31) -> tuple[Path, dict[str, np.ndarray]]:
    osim = require_opensim()
    model = osim.Model(str(GOLF_MODEL))
    coords = model.getCoordinateSet()
    t = np.linspace(0.0, 0.3, n)
    rng = np.random.default_rng(7)
    cols: dict[str, np.ndarray] = {}
    for i in range(coords.getSize()):
        c = coords.get(i)
        name = c.getName()
        if "beta" in name:
            continue
        amp = 0.02 if name.startswith("pelvis_t") else 0.15
        phase = rng.uniform(0, 3)
        cols[name] = amp * np.sin(4 * t + phase)
    lines = [
        "synthetic",
        "version=1",
        f"nRows={n}",
        f"nColumns={len(cols) + 1}",
        "inDegrees=no",
        "endheader",
        "time\t" + "\t".join(cols),
    ]
    for i, ti in enumerate(t):
        lines.append("\t".join([f"{ti:.12f}"] + [f"{cols[k][i]:.15g}" for k in cols]))
    path.write_text("\n".join(lines) + "\n")
    return path, cols


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("policy", ["none", "grf_only", "club_and_grf"])
def test_identity_stage_builds_table_loads_and_record(
    tmp_path: Path, policy: str
) -> None:
    fmap = _identity_map()
    motion, cols = _golf_motion(tmp_path / "swing.sto")
    t = np.linspace(0.0, 0.3, 31)
    trunk = {k: cols[k] for k in cols if k.startswith(("pelvis_", "lumbar_"))}
    spec = FullbodySpec(
        fmap=fmap,
        golf_model_path=GOLF_MODEL,
        motion_path=motion,
        target_model_path=GOLF_MODEL,
        policy=policy,
        output_dir=tmp_path / "out",
        stem="idstudy",
    )
    result = run_fullbody(spec, t, trunk)
    for name in fmap.targets_with("driven_direct"):
        np.testing.assert_allclose(result.coordinates[name], cols[name], atol=1e-12)
    for group in fmap.groups:
        for name in group.target_coordinates:
            np.testing.assert_allclose(result.coordinates[name], cols[name], atol=1e-5)
    assert result.metrics["orientation_max_residual_rad"] < 1e-5
    record = result.record
    assert record["policy"] == policy and record["map_id"] == "golf_identity"
    kinds = {item["kind"] for item in record["included"]}
    if policy == "none":
        assert result.load_xml is None and kinds == set()
        assert {e["item"] for e in record["excluded"]} == {
            "club_hand_wrench",
            "ground_reaction",
        }
    elif policy == "grf_only":
        assert kinds == {"ground_reaction"}
        assert [e["item"] for e in record["excluded"]] == ["club_hand_wrench"]
    else:
        assert kinds == {"ground_reaction", "club_hand_wrench"}
        bodies = {
            i["body"] for i in record["included"] if i["kind"] == "club_hand_wrench"
        }
        assert bodies == {"hand_r", "hand_l"}
        assert record["excluded"] == []
    if result.load_xml is not None:
        assert result.load_xml.is_file()
        roles = {a.role for a in result.artifacts}
        assert {"external_loads_xml", "external_loads_sto"} <= roles
        assert all(len(a.sha256) == 64 for a in result.artifacts)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_identity_stage_is_deterministic_and_rejects_short_motions(
    tmp_path: Path,
) -> None:
    fmap = _identity_map()
    motion, cols = _golf_motion(tmp_path / "swing.sto")
    t = np.linspace(0.0, 0.3, 31)
    trunk = {k: cols[k] for k in cols if k.startswith(("pelvis_", "lumbar_"))}

    def go(sub: str):
        spec = FullbodySpec(
            fmap, GOLF_MODEL, motion, GOLF_MODEL, "club_and_grf", tmp_path / sub, "d"
        )
        return run_fullbody(spec, t, trunk)

    a, b = go("a"), go("b")
    assert [x.sha256 for x in a.artifacts] == [x.sha256 for x in b.artifacts]
    spec = FullbodySpec(
        fmap, GOLF_MODEL, motion, GOLF_MODEL, "grf_only", tmp_path / "c", "d"
    )
    with pytest.raises(ValueError, match="13 frames"):
        run_fullbody(spec, t[:10], {k: v[:10] for k, v in trunk.items()})
    with pytest.raises(ValueError, match="time grid"):
        run_fullbody(spec, t + 1.0, trunk)
