"""Full-body coordinate map: schema, coverage and direct transfer (SPN-9c, #12758).

Pure tests (no OpenSim): they validate the committed names-only JSON and the
loader's design-by-contract checks.  Every coordinate value below is INVENTED.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    DEFAULT_FULLBODY_MAP_PATH,
    apply_direct_map,
    get_fullbody_map,
    load_fullbody_maps,
    neutral_columns,
    parse_fullbody_maps,
)

pytestmark = pytest.mark.unit

BRUNO = "bruno2015_fullbody_os4_v2"
RAW = json.loads(Path(DEFAULT_FULLBODY_MAP_PATH).read_text(encoding="utf-8"))


def _raw() -> dict:
    return copy.deepcopy(RAW)


def _entry(raw: dict) -> dict:
    return raw["maps"][BRUNO]


def test_committed_map_loads_and_accounts_for_every_coordinate() -> None:
    fmap = get_fullbody_map(BRUNO)
    assert fmap.verified
    assert len(fmap.golf_treatment) == 39
    assert len(fmap.target_treatment) == 165
    assert set(fmap.golf_treatment.values()) == {
        "direct",
        "pelvis_pose",
        "orientation_match",
        "spine_drive",
        "excluded",
    }
    for name, why in fmap.golf_reason.items():
        assert fmap.golf_treatment[name] in {"excluded", "spine_drive"}
        assert why.strip()


def test_every_direct_and_group_target_is_classified_consistently() -> None:
    fmap = get_fullbody_map(BRUNO)
    for entry in fmap.direct:
        assert fmap.target_treatment[entry.target] == "driven_direct"
        assert entry.sign in (-1, 1)
    for entry in fmap.pelvis:
        assert fmap.target_treatment[entry.target] == "driven_by_pelvis_pose"
    for group in fmap.groups:
        for name in group.target_coordinates:
            assert fmap.target_treatment[name] == "driven_orientation_match"
        for name in group.golf_coordinates:
            assert fmap.golf_treatment[name] == "orientation_match"
    assert len(fmap.targets_with("driven_direct")) == len(fmap.direct)


def test_the_knee_sign_is_recorded_not_assumed() -> None:
    fmap = get_fullbody_map(BRUNO)
    signs = {e.golf: e.sign for e in fmap.direct}
    assert signs["knee_angle_r"] == -1 and signs["knee_angle_l"] == -1
    assert signs["hip_flexion_r"] == 1


def test_apply_direct_map_applies_the_sign_and_leaves_the_input_untouched() -> None:
    fmap = get_fullbody_map(BRUNO)
    values = {"knee_angle_r": np.array([0.1, 0.2]), "hip_flexion_r": np.array([1.0, 2])}
    out = apply_direct_map(fmap, values)
    np.testing.assert_allclose(out["knee_angle_r"], [-0.1, -0.2])
    np.testing.assert_allclose(out["hip_flexion_r"], [1.0, 2.0])
    assert "elbow_flexion_r" not in out
    np.testing.assert_allclose(values["knee_angle_r"], [0.1, 0.2])


def test_neutral_columns_cover_only_the_held_coordinates() -> None:
    fmap = get_fullbody_map(BRUNO)
    cols = neutral_columns(fmap, 4)
    assert set(cols) == set(fmap.targets_with("held_neutral"))
    assert all(v.shape == (4,) and not v.any() for v in cols.values())
    with pytest.raises(ValueError, match="n_frames"):
        neutral_columns(fmap, 0)


def test_check_coverage_accepts_exact_sets_and_names_the_offenders() -> None:
    fmap = get_fullbody_map(BRUNO)
    golf, target = set(fmap.golf_treatment), set(fmap.target_treatment)
    fmap.check_coverage(golf, target)
    with pytest.raises(ValueError, match="unclassified.*extra_coord"):
        fmap.check_coverage(golf, target | {"extra_coord"})
    with pytest.raises(ValueError, match="absent from the target.*hip_flexion_r"):
        fmap.check_coverage(golf, target - {"hip_flexion_r"})
    with pytest.raises(ValueError, match="unclassified.*new_golf"):
        fmap.check_coverage(golf | {"new_golf"}, target)


def test_summary_states_what_is_not_driven() -> None:
    summary = get_fullbody_map(BRUNO).summary()
    assert summary["map_id"] == BRUNO
    assert summary["golf_coordinates"]["excluded"] == sorted(
        summary["golf_coordinates"]["excluded"]
    )
    assert "subtalar_angle_r" in summary["golf_coordinates"]["excluded"]
    assert summary["target_counts"]["driven_orientation_match"] == 6
    assert summary["target_counts"]["locked_in_asset"] > 0
    json.dumps(summary)


def test_loader_rejects_bad_documents() -> None:
    raw = _raw()
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        parse_fullbody_maps(raw)
    with pytest.raises(TypeError, match="mapping"):
        parse_fullbody_maps([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown map id"):
        get_fullbody_map("nope")
    with pytest.raises(TypeError, match="map_id"):
        get_fullbody_map(3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda e: e["golf_coordinates"][6].update(treatment="bogus"), "treatment"),
        (lambda e: e["golf_coordinates"][0].update(sign=0), "sign"),
        (lambda e: e["golf_coordinates"][0].update(min_axis_dot=1.5), "min_axis_dot"),
        (
            lambda e: e["golf_coordinates"].append(dict(e["golf_coordinates"][0])),
            "twice",
        ),
        (
            lambda e: e["golf_coordinates"][0].update(target="hip_flexion_l"),
            "duplicate target",
        ),
        (
            lambda e: e["golf_coordinates"][0].update(target="not_a_target"),
            "not classified",
        ),
        (
            lambda e: [
                c.pop("reason", None)
                for c in e["golf_coordinates"]
                if c["treatment"] == "excluded"
            ],
            "reason",
        ),
        (
            lambda e: e["target_coordinates"]["held_neutral"].append("Abs_FE"),
            "classified twice",
        ),
        (lambda e: e["target_coordinates"].pop("locked_in_asset"), "treatment"),
        (lambda e: e["hand_bodies"].update({"hand_r": ""}), "hand_bodies"),
        (
            lambda e: e["orientation_groups"]["shoulder_r"].update(
                golf_coordinates=["arm_flex_r", "arm_add_r"]
            ),
            "three",
        ),
    ],
)
def test_loader_rejects_each_inconsistency(mutate, match) -> None:
    raw = _raw()
    mutate(_entry(raw))
    with pytest.raises(ValueError, match=match):
        parse_fullbody_maps(raw)


def test_load_from_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    assert BRUNO in load_fullbody_maps(path)
    with pytest.raises(ValueError, match="not found"):
        load_fullbody_maps(tmp_path / "absent.json")
