"""full-body-v2 regional trunk geometry, mass and observability (HRG-D2, #13173).

Joint centres come from landmarks (the HRG-C3 L5/S1 design offset from the
hip-joint-centre midpoint, and the de Leva cervicale as C7/T1), T12/L1 from
the Gray's Anatomy regional lengths. The trunk keeps every de Leva solid
(re-parented, never re-split) and drops only the 0.3 kg dummy body.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching import full_body_v2_geometry as geo
from src.shared.python.motion_matching.pelvis_frame import relocate_pelvis

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
THORAX = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod"
DUMMY = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/UpperTorsoBase"
NECK = "GolfSwing3D_Kinetic/Neck Joint"
SEED = 13173


def _v1() -> dict:
    return json.loads(V1_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1() -> dict:
    return _v1()


@pytest.fixture(scope="module")
def doc(v1) -> dict:
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-D2 unit fixture")


def _neck_centre(spec: dict) -> np.ndarray:
    poses = spec_body_poses(spec)
    neck = next(j for j in spec["joints"] if j["name"] == NECK)
    pose = np.linalg.inv(poses[PELVIS]) @ poses[neck["parent"]]
    return (pose @ np.asarray(neck["parent_to_base"]))[:3, 3]


# ----------------------------------------------------------------- locations


def test_landmark_locations_follow_c3_l5s1_and_gray_regional_lengths(v1) -> None:
    loc = geo.landmark_trunk_locations(v1)
    twist = next(
        j for j in v1["joints"] if j["primitives"][0]["coordinate"] == "TorsoInput"
    )
    pivot = np.asarray(twist["parent_to_base"])[:3, 3]
    np.testing.assert_allclose(loc.l5s1_in_pelvis_m, pivot, atol=1e-15)
    c7 = _neck_centre(v1)
    fraction = 18.0 / (18.0 + 28.0)
    np.testing.assert_allclose(
        loc.t12l1_in_pelvis_m, pivot + fraction * (c7 - pivot), atol=1e-15
    )
    assert pytest.approx(fraction, abs=1e-15) == geo.LUMBAR_LENGTH_FRACTION
    assert "Gray" in loc.provenance and "HRG-C3" in loc.provenance
    evidence = loc.evidence
    assert evidence is not None
    np.testing.assert_allclose(evidence["c7t1_in_pelvis_m"], c7, atol=1e-15)


def test_landmark_l5s1_fails_closed_on_a_calibrated_unrelocated_pelvis(v1) -> None:
    moved = copy.deepcopy(v1)
    for joint in moved["joints"]:
        if joint["name"] in ("hip_r", "hip_l"):
            joint["parent_to_base"][2][3] -= 0.05
    with pytest.raises(ValueError, match="relocate_pelvis"):
        geo.landmark_trunk_locations(moved)


def test_landmark_l5s1_accepts_a_c3_revision(v1) -> None:
    rev = relocate_pelvis(v1, (0.0, -0.09, -0.02), (0.0, 0.09, -0.01)).spec
    loc = geo.landmark_trunk_locations(rev)
    design = rev["spec_revision"]["design_offsets_from_hjc_midpoint_m"]["l5s1"]
    mid = geo.hjc_midpoint(rev)
    np.testing.assert_allclose(loc.l5s1_in_pelvis_m, mid + design, atol=1e-12)
    np.testing.assert_allclose(mid, 0.0, atol=1e-12)


def test_neutral_curvature_is_checked_against_yukawa_2018(v1) -> None:
    report = geo.landmark_trunk_locations(v1).evidence["neutral_curvature"]
    # Uniform-curvature chords: thoracic chord relative to lumbar chord is
    # ISB extension (LL - TK) / 2 = (49.7 - 36.0) / 2 = 6.85 deg.
    expected = report["thoracolumbar_chord_extension_deg"]
    assert expected["reference_mean_deg"] == pytest.approx(6.85, abs=1e-12)
    sd = np.hypot(11.2, 10.1) / 2.0
    assert expected["reference_sd_deg"] == pytest.approx(sd, abs=1e-12)
    assert expected["model_deg"] == pytest.approx(0.0, abs=1e-9)
    assert expected["z_score"] == pytest.approx(-6.85 / sd, abs=1e-9)
    assert expected["within_2sd"] is True
    lumbar = report["lumbar_chord_forward_tilt_deg"]
    assert lumbar["comparable"] is False
    assert "waist" in lumbar["reason"]
    assert "Yukawa" in report["reference"]


# ---------------------------------------------------------- bodies and mass


def test_v2_trunk_reparents_de_leva_solids_and_drops_only_the_dummy(v1, doc) -> None:
    bodies = {b["name"]: b for b in doc["bodies"]}
    assert DUMMY not in bodies
    assert [s["name"].rsplit("/", 1)[-1] for s in bodies[v2.LUMBAR_BODY]["solids"]] == [
        "middle_trunk"
    ]
    assert [s["name"].rsplit("/", 1)[-1] for s in bodies[THORAX]["solids"]] == [
        "upper_trunk"
    ]
    assert doc["trunk_topology"]["locations_evidence"]["c7t1_in_pelvis_m"]


def test_relative_mass_error_is_below_1e_10(v1, doc) -> None:
    m1 = geo.spec_mass_properties(v1).mass_kg
    m2 = geo.spec_mass_properties(doc).mass_kg
    dropped = sum(b["mass_kg"] for b in doc["trunk_topology"]["dropped_bodies"])
    assert dropped == 0.3
    assert abs(m2 - (m1 - dropped)) / (m1 - dropped) <= 1e-10
    ledger = geo.human_mass_ledger(doc)
    # The issue's "true 78.0 kg" is not reached by dropping the dummy alone.
    assert ledger["subject_mass_kg"] == 78.0
    assert ledger["human_mass_kg"] == pytest.approx(78.7455, abs=5e-5)
    assert sum(ledger["excess_kg"].values()) == pytest.approx(
        ledger["human_mass_kg"] - 78.0, abs=1e-9
    )


def _random_non_trunk_q(doc: dict, rng: np.random.Generator) -> dict[str, float]:
    trunk = set(v2.V2_TRUNK.coordinates)
    return {
        n: float(rng.uniform(-0.4, 0.4))
        for n in doc["coordinate_order"]
        if n not in trunk and not n.startswith("Translation")
    }


def test_reference_pose_com_within_1_mm(v1, doc) -> None:
    rng = np.random.default_rng(SEED)
    poses = [{}] + [_random_non_trunk_q(doc, rng) for _ in range(5)]
    for q in poses:
        c1 = geo.spec_mass_properties(v1, q).com_m
        c2 = geo.spec_mass_properties(doc, q).com_m
        assert float(np.linalg.norm(c2 - c1)) <= 1e-3


def test_locked_aggregate_inertia_relative_error_below_1e_6(v1, doc) -> None:
    rng = np.random.default_rng(SEED + 1)
    for q in [{}] + [_random_non_trunk_q(doc, rng) for _ in range(5)]:
        i1 = geo.spec_mass_properties(v1, q, exclude_bodies=(DUMMY,)).inertia_com
        i2 = geo.spec_mass_properties(doc, q).inertia_com
        assert np.max(np.abs(i2 - i1)) / np.max(np.abs(i1)) <= 1e-6
    # The dummy itself is what the aggregate loses, and it is not negligible.
    full = geo.spec_mass_properties(v1).inertia_com
    i2 = geo.spec_mass_properties(doc).inertia_com
    assert np.max(np.abs(i2 - full)) / np.max(np.abs(full)) > 1e-6


def test_mujoco_compiled_mass_and_inertia_match_the_spec(doc) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(json.dumps(doc).encode("utf-8"))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    props = geo.spec_mass_properties(doc)
    total = float(np.sum(model.body_mass))
    assert abs(total - props.mass_kg) / props.mass_kg <= 1e-10
    com = data.subtree_com[0]
    assert float(np.linalg.norm(com - props.com_m)) <= 1e-9


def test_mass_properties_preconditions(doc) -> None:
    with pytest.raises(ValueError, match="exclude_bodies"):
        geo.spec_mass_properties(doc, exclude_bodies=("no such body",))
    with pytest.raises(TypeError):
        geo.spec_mass_properties([])  # type: ignore[arg-type]


# --------------------------------------------------------------------- ROM


def test_trunk_rom_is_soft_and_hard_ranges_are_report_only(doc) -> None:
    ranges = v2.v2_trunk_ranges_deg()
    assert not set(v2.V2_TRUNK.coordinates) & set(doc["coordinate_ranges_deg"])
    for name, bounds in ranges.items():
        assert tuple(doc["soft_limits_deg"][name]) == bounds
    assert doc["trunk_topology"]["hard_range_policy"].startswith("report_only")


# ------------------------------------------------------------ observability


def test_back_marker_observability_at_neutral(doc) -> None:
    report = geo.trunk_observability(doc)
    assert report["markers"] == list(geo.BACK_MARKERS)
    assert report["rank"] == 5
    assert report["null_space_dimension"] == 1
    null = np.asarray(report["null_directions"][0])
    names = list(v2.V2_TRUNK.coordinates)
    ia = names.index("lumbar_axial_rotation")
    it = names.index("thoracic_axial_rotation")
    assert abs(abs(null[ia]) - np.sqrt(0.5)) < 1e-6
    assert abs(null[ia] + null[it]) < 1e-6
    status = report["coordinate_status"]
    assert status["lumbar_axial_rotation"] == "degenerate"
    assert status["thoracic_axial_rotation"] == "degenerate"
    assert "identified" not in status.values()
    assert report["split_status"] == "regularised"
    assert report["distribution_source"] == "prior:bruno2015_regional@1"
    for sd in report["posterior_sd_deg"].values():
        assert np.isfinite(sd) and sd > 0.0


def test_observability_off_neutral_and_preconditions(doc) -> None:
    q = {"lumbar_flexion": 0.3, "thoracic_lateral_bending": 0.2}
    report = geo.trunk_observability(doc, q=q)
    # Structural: T12/L1 sits a fixed distance from L5/S1, so the thorax has
    # only five DOF relative to the pelvis in any pose (spin about the chord).
    assert report["rank"] == 5
    assert report["null_space_dimension"] == 1
    assert report["split_status"] == "regularised"
    with pytest.raises(ValueError, match="noise_m"):
        geo.trunk_observability(doc, noise_m=0.0)
    with pytest.raises(ValueError, match="marker"):
        geo.trunk_observability(doc, markers=("NoSuchMarker",))


def test_thorax_orientation_is_identified_while_the_split_is_not(doc) -> None:
    """The null spin moves the lumbar body only, so the thorax is observed."""
    for q in ({}, {"lumbar_flexion": 0.3, "thoracic_lateral_bending": 0.2}):
        report = geo.trunk_observability(doc, q=q)
        sd = report["thorax_orientation_data_sd_deg"]
        assert set(sd) == {"x", "y", "z"}
        for value in sd.values():
            assert np.isfinite(value) and value > 0.0
        if not q:  # at neutral the composed rotation is the regional sum
            np.testing.assert_allclose(
                sorted(sd.values()),
                sorted(report["regional_sum_data_sd_deg"].values()),
                rtol=1e-6,
            )
        share = report["null_share"]
        assert set(share) == set(v2.V2_TRUNK.coordinates)
        assert max(share.values()) == pytest.approx(
            float(np.max(np.abs(report["null_directions"][0]))), abs=1e-12
        )
