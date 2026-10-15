"""Drive a coupled segmental model from golf trunk input (SPN-9b, #12757).

Targets are generated from the fixture's own invented coupling law, so a
correct drive must recover the generating independent coordinates and, once the
``.mot`` is replayed natively on the coupled fixture, reproduce the target
trunk rotation.  Every number is invented; the evidence-file test records the
maximum residual and skips with a reason when the repository inputs are absent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    body_transform_in,
    set_coordinate_values,
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    GOLF_PELVIS,
    DriveOptions,
    DriveResult,
    chain_offsets_from_model,
    drive_segmental_model,
    export_mot,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_base_offset,
    fixture_level_names,
    fixture_neutral_transforms,
    fixture_terminal_offset,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    GolfTrunkInput,
    read_golf_humanoid_motion,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    LevelMapEntry,
    ModelLevelMap,
    load_level_maps,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.coupling import LinearCouplingLaw, level_rotations
from src.shared.python.spine.coupling_data import coupling_law
from src.shared.python.spine.distribution import ChainOffsets
from src.shared.python.spine.levels import SpinalLevel, levels_between
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]
EVID = REPO / "docs/development/opensim_tour_matching/evidence/os7_moco_g1/inputs"
EVID_STO = EVID / "ik_states_full.sto"
EVID_OSIM = EVID / "golf_humanoid_scaled_tour_markers.osim"
PHI = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)


@pytest.fixture(scope="module", autouse=True)
def _osim():
    return require_opensim()


def _law() -> LinearCouplingLaw:
    mats = {}
    for k, lv in enumerate(LUMBAR):
        w = 0.1 + 0.05 * k
        mats[lv] = np.array(
            [[w, 0.03 * (k + 1), 0.0], [0.0, w, 0.02 * (k + 1)], [0.0, 0.0, w / 2]]
        )
    return LinearCouplingLaw("invented", "1", mats, PHI, "invented, SPN-9b test")


def _phi(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.1, n)
    phi = np.stack(
        [0.5 * np.sin(2.1 * t), 0.35 * np.sin(1.3 * t + 0.4), 0.6 * np.sin(1.7 * t)],
        axis=1,
    )
    return t, phi


def _offsets(spec: FixtureSpec) -> ChainOffsets:
    return ChainOffsets(
        fixture_base_offset(spec)[:3, :3], fixture_terminal_offset(spec)[:3, :3]
    )


def _pelvis_pose(n: int) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = Rotation.from_euler(
            "XYZ", [0.05 * i, -0.03 * i, 0.04 * i]
        ).as_matrix()
        poses[i, :3, 3] = [0.1 * i, 0.9 + 0.01 * i, -0.02 * i]
    return poses


def _golf_input(spec: FixtureSpec, law: LinearCouplingLaw, n: int = 12):
    t, phi = _phi(n)
    neutral = fixture_neutral_transforms(spec)
    fixed = _offsets(spec)
    rots = np.stack(
        [
            fixed.base_rotation
            @ level_rotations(law, p, neutral).compose()[:3, :3]
            @ fixed.terminal_rotation
            for p in phi
        ]
    )
    golf = GolfTrunkInput(
        time_s=t,
        pelvis_to_thorax=rots,
        pelvis_pose_ground=_pelvis_pose(n),
        other_coordinates={},
        events={},
        source={"file": "synthetic.sto", "sha256": "0" * 64},
    )
    return golf, phi


@pytest.fixture(scope="module")
def coupled(tmp_path_factory):
    spec = FixtureSpec()
    law = _law()
    path = write_fixture_osim(
        spec, tmp_path_factory.mktemp("drive") / "fixture.osim", coupling=law
    )
    return spec, law, path


def _drive(path: Path, golf: GolfTrunkInput, **kw) -> DriveResult:
    options = DriveOptions(pelvis=FIXTURE_PELVIS, **kw)
    return drive_segmental_model(
        golf, model_path=path, model_id="fixture", law=_law(), options=options
    )


def test_recovers_generating_coordinates(coupled) -> None:
    spec, law, path = coupled
    golf, phi = _golf_input(spec, law)
    result = _drive(path, golf)
    assert np.max(result.residual_rad) < 1e-8
    assert result.max_residual_rad < 1e-8
    assert result.valid.all()
    np.testing.assert_allclose(result.phi, phi, atol=1e-6)
    assert result.independent_names == PHI
    assert result.law_id == "invented" and result.law_version == "1"
    assert result.kinematics.distribution_source == "coupling:invented@1"
    assert result.input_source["sha256"] == "0" * 64
    assert len(result.model_sha256) == 64


def test_chain_offsets_are_read_natively(coupled) -> None:
    spec, _law_, path = coupled
    fixed = chain_offsets_from_model(path, "fixture")
    expected = _offsets(spec)
    np.testing.assert_allclose(fixed.base_rotation, expected.base_rotation, atol=1e-9)
    np.testing.assert_allclose(
        fixed.terminal_rotation, expected.terminal_rotation, atol=1e-9
    )


def test_mot_columns_and_native_replay(coupled, tmp_path) -> None:
    spec, law, path = coupled
    golf, _phi_ = _golf_input(spec, law)
    result = _drive(path, golf)
    mot = export_mot(result, tmp_path / "drive.mot")
    osim = import_opensim()
    table = osim.TimeSeriesTable(str(mot))
    labels = [str(x) for x in table.getColumnLabels()]
    expected = [c for n in fixture_level_names(spec) for c in n.coordinates]
    expected += list(FIXTURE_PELVIS.rotation) + list(FIXTURE_PELVIS.translation)
    assert sorted(labels) == sorted(expected)
    assert table.getTableMetaDataAsString("inDegrees") == "no"
    assert int(table.getNumRows()) == golf.time_s.shape[0]
    times = [table.getIndependentColumn()[r] for r in range(golf.time_s.size)]
    np.testing.assert_allclose(times, golf.time_s, atol=1e-12)
    # End-to-end: replay the .mot on the coupled fixture and compare the
    # thorax_proxy-in-pelvis rotation with the golf target and the pelvis pose.
    model = osim.Model(str(path))
    state = model.initSystem()
    for frame in range(0, golf.time_s.size, 3):
        row = {lab: float(table.getDependentColumn(lab)[frame]) for lab in labels}
        set_coordinate_values(model, state, row)
        thorax = body_transform_in(model, state, "thorax_proxy", "pelvis")
        np.testing.assert_allclose(
            thorax[:3, :3], golf.pelvis_to_thorax[frame], atol=1e-8
        )
        body = model.getBodySet().get("pelvis")
        pelvis = transform_to_matrix(body.getTransformInGround(state))
        np.testing.assert_allclose(pelvis, golf.pelvis_pose_ground[frame], atol=1e-8)


def test_unreachable_target_reports_residual_never_rescales(coupled) -> None:
    spec, law, path = coupled
    golf, _phi_ = _golf_input(spec, law)
    poor = LinearCouplingLaw(
        "axial_only",
        "1",
        {lv: np.diag([0.0, 0.0, 0.2]) for lv in LUMBAR},
        PHI,
        "invented",
    )
    options = DriveOptions(pelvis=FIXTURE_PELVIS, tolerance_rad=1e-6)
    result = drive_segmental_model(
        golf, model_path=path, model_id="fixture", law=poor, options=options
    )
    assert result.max_residual_rad > 1e-3
    assert np.max(result.residual_rad) == result.max_residual_rad
    assert not result.valid.all()
    assert result.kinematics.distribution_source == "coupling:axial_only@1"
    # The law is untouched: level angles are exactly the law's expansion.
    expect = poor.expand(result.phi[4])
    for names in fixture_level_names(spec):
        got = [result.coordinates[c][4] for c in names.coordinates]
        np.testing.assert_allclose(got, expect[names.level], atol=1e-12)


def test_law_extracted_from_model_matches_given_law(coupled) -> None:
    spec, law, path = coupled
    golf, phi = _golf_input(spec, law)
    result = drive_segmental_model(
        golf,
        model_path=path,
        model_id="fixture",
        options=DriveOptions(pelvis=FIXTURE_PELVIS),
    )
    assert result.max_residual_rad < 1e-8
    np.testing.assert_allclose(result.phi, phi, atol=1e-6)
    assert result.kinematics.distribution_source.startswith("coupling:")


def test_pelvis_encodings() -> None:
    assert GOLF_PELVIS.rotation == ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
    assert GOLF_PELVIS.sequence == "ZXY"
    assert FIXTURE_PELVIS.sequence == "XYZ"


def test_preconditions(coupled) -> None:
    spec, law, path = coupled
    golf, _phi_ = _golf_input(spec, law)
    with pytest.raises(TypeError, match="GolfTrunkInput"):
        drive_segmental_model("nope", model_path=path, model_id="fixture", law=law)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model file not found"):
        drive_segmental_model(
            golf, model_path=path.with_name("missing.osim"), model_id="fixture"
        )
    with pytest.raises(ValueError, match="unknown model_id"):
        drive_segmental_model(golf, model_path=path, model_id="nope", law=law)
    short = LinearCouplingLaw("short", "1", {LUMBAR[0]: np.eye(3) * 0.1}, PHI, "inv")
    with pytest.raises(ValueError, match="levels"):
        drive_segmental_model(golf, model_path=path, model_id="fixture", law=short)
    with pytest.raises(ValueError, match="tolerance_rad"):
        DriveOptions(tolerance_rad=-1.0)
    with pytest.raises(ValueError, match="proper rotation"):
        DriveOptions(torso_alignment=np.ones((3, 3)))
    with pytest.raises(TypeError, match="DriveResult"):
        export_mot("x", Path("x.mot"))  # type: ignore[arg-type]


def test_export_rejects_wrong_suffix(coupled, tmp_path) -> None:
    spec, law, path = coupled
    golf, _phi_ = _golf_input(spec, law, n=4)
    result = _drive(path, golf)
    with pytest.raises(ValueError, match=r"\.mot"):
        export_mot(result, tmp_path / "x.txt")


def test_torso_alignment_is_applied(coupled) -> None:
    spec, law, path = coupled
    golf, phi = _golf_input(spec, law, n=6)
    align = Rotation.from_euler("Y", 0.2).as_matrix()
    shifted = replace(golf, pelvis_to_thorax=golf.pelvis_to_thorax @ align.T)
    result = _drive(path, shifted, torso_alignment=align)
    assert result.max_residual_rad < 1e-8
    np.testing.assert_allclose(result.phi, phi, atol=1e-6)


@pytest.mark.skipif(
    not (EVID_STO.is_file() and EVID_OSIM.is_file()),
    reason="evidence inputs (ik_states_full.sto / golf model) not present",
)
def test_evidence_swing_drives_full_fixture(tmp_path) -> None:
    spec = FixtureSpec(levels=tuple(SpinalLevel))
    law = coupling_law("bruno2015_regional")
    path = write_fixture_osim(spec, tmp_path / "full.osim", coupling=law)
    base = load_level_maps()["fixture"]
    level_map = ModelLevelMap(
        verified=True,
        pelvis_body=base.pelvis_body,
        terminal_body=base.terminal_body,
        levels={
            n.level: LevelMapEntry(n.joint, n.coordinates)
            for n in fixture_level_names(spec)
        },
    )
    golf = read_golf_humanoid_motion(EVID_STO, EVID_OSIM)
    stride = max(1, golf.time_s.size // 40)
    golf = replace(
        golf,
        time_s=golf.time_s[::stride],
        pelvis_to_thorax=golf.pelvis_to_thorax[::stride],
        pelvis_pose_ground=golf.pelvis_pose_ground[::stride],
        other_coordinates={},
    )
    options = DriveOptions(pelvis=FIXTURE_PELVIS, level_map=level_map)
    result = drive_segmental_model(
        golf, model_path=path, model_id="fixture", law=law, options=options
    )
    assert np.all(np.isfinite(result.residual_rad))
    assert all(np.all(np.isfinite(v)) for v in result.coordinates.values())
    print(f"EVIDENCE max residual rad = {result.max_residual_rad:.6e}")
