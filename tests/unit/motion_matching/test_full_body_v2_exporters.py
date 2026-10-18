"""A full-body-v2 fixture round-trips through every exporter's validator (HRG-D1).

Each exporter dispatches on the spec's schema id through the one registry
(``full_body_schema``). The v2 fixture uses the declared HRG-D1 stand-in
joint locations; geometry, mass and cross-engine qualification are HRG-D2/Q1.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    upper_body_slice,
)
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SEED = 1317


@pytest.fixture(scope="module")
def v2_doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return v2.derive_full_body_v2_spec(
        v1, v2.segment_boundary_locations(v1), provenance="HRG-D1 exporter fixture"
    )


@pytest.fixture(scope="module")
def v2_bytes(v2_doc) -> bytes:
    return json.dumps(v2_doc, sort_keys=True).encode("utf-8")


def _random_q(doc: dict) -> dict[str, float]:
    """Seeded rotations on every coordinate; translations stay at zero."""
    rng = np.random.default_rng(SEED)
    return {
        name: 0.0 if name.startswith("Translation") else float(rng.uniform(-0.4, 0.4))
        for name in doc["coordinate_order"]
    }


def _spec_frames(doc: dict, q: dict[str, float]) -> dict[str, np.ndarray]:
    poses = spec_body_poses(doc, q)
    return {
        f["name"]: poses[f["body"]] @ np.asarray(f["placement"], dtype=float)
        for f in doc["frames"]
    }


def test_mujoco_export_compiles_in_v2_order_and_matches_spec_fk(v2_doc, v2_bytes):
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(v2_bytes)
    assert meta["coordinate_order"] == v2_doc["coordinate_order"]
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    q = _random_q(v2_doc)
    for name, value in q.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    expected = _spec_frames(v2_doc, q)
    for site, frame in (("Hub", "Hub"), ("Spine", "Spine"), ("Torso", "Torso")):
        sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"][site]
        )
        np.testing.assert_allclose(
            data.site_xpos[sid], expected[frame][:3, 3], atol=1e-9
        )


def test_mujoco_visual_layer_is_not_implemented_for_v2(v2_bytes) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    with pytest.raises(
        SchemaFeatureUnavailableError, match=r"HRG-D3 \(tracked: #13174\)"
    ):
        export_full_body_mjcf(v2_bytes, visual=True)


def test_urdf_bundle_accepts_v2(v2_doc, v2_bytes) -> None:
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    _, sidecar = export_full_body_urdf(v2_bytes)
    assert list(sidecar["coordinate_order"]) == v2_doc["coordinate_order"]


@pytest.mark.requires_drake
def test_drake_model_accepts_v2(v2_doc) -> None:
    pytest.importorskip("pydrake.all")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    model = FullBodyDrakeModel(v2_doc)
    assert model.coordinate_order == tuple(v2_doc["coordinate_order"])
    q = _random_q(v2_doc)
    expected = _spec_frames(v2_doc, q)
    got = model.frame_poses(q)
    for name in ("Hub", "Spine", "Torso"):
        np.testing.assert_allclose(got[name], expected[name], atol=1e-9)


@pytest.mark.requires_pinocchio
def test_pinocchio_model_accepts_v2_and_builds_the_v2_upper_model(v2_doc) -> None:
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(v2_doc)
    assert model.coordinate_order == tuple(v2_doc["coordinate_order"])
    q = _random_q(v2_doc)
    expected = _spec_frames(v2_doc, q)
    got = model.frame_poses(q)
    for name in ("Hub", "Spine", "Torso"):
        np.testing.assert_allclose(np.asarray(got[name]), expected[name], atol=1e-9)
    upper = model.upper_body_model()  # HRG-D2: the regional-trunk upper body
    full_q = dict.fromkeys(v2_doc["coordinate_order"], 0.0)
    full_q.update(dict.fromkeys(v2.V2_TRUNK.coordinates, 0.1))
    names = upper_body_slice(v2_doc)["coordinate_order"]
    upper_q = {c: full_q[c] for c in names}
    full = model.frame_poses(full_q)
    for name, pose in upper.frame_poses(upper_q).items():
        np.testing.assert_allclose(np.asarray(pose), np.asarray(full[name]), atol=1e-12)


def test_opensim_export_and_parity_adapter_accept_v2(v2_doc, v2_bytes) -> None:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml, meta = export_full_body_osim(v2_bytes, club_geometry_ref=None)
    assert "lumbar_flexion" in xml and "TorsoInput" not in xml
    osim = require_opensim()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v2.osim"
        path.write_text(xml, encoding="utf-8")
        model = osim.Model(str(path))
        model.initSystem()
        names = {
            model.getCoordinateSet().get(i).getName()
            for i in range(model.getCoordinateSet().getSize())
        }
    assert set(v2.V2_TRUNK.coordinates) <= names
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        OpenSimFullBodyParityAdapter,
    )

    adapter = OpenSimFullBodyParityAdapter(v2_bytes)
    assert adapter.coordinate_order == tuple(v2_doc["coordinate_order"])
    mass = sum(s["mass_kg"] for b in v2_doc["bodies"] for s in b["solids"])
    assert adapter.mass_kg == pytest.approx(mass, rel=1e-9)


def test_opensim_musculoskeletal_graft_stays_v1_only(v2_bytes) -> None:
    from src.engines.physics_engines.opensim.python.musculoskeletal_graft import (
        spec_document,
    )

    # Deferred to HRG-Q1 (decision recorded by HRG-D2, #13173).
    with pytest.raises(
        SchemaFeatureUnavailableError, match=r"HRG-Q1 \(tracked: #13176\)"
    ):
        spec_document(v2_bytes)


def test_myosuite_spec_kinematics_accepts_v2(v2_doc, v2_bytes) -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.myofullbody.mapping import SpecKinematics

    kin = SpecKinematics(v2_bytes)
    assert kin.coordinate_order == tuple(v2_doc["coordinate_order"])


def test_polynomial_actuation_accepts_v2(v2_doc) -> None:
    from src.shared.python.motion_matching.polynomial_actuation import (
        FullBodyPolynomialControl,
    )

    control = FullBodyPolynomialControl.from_spec(v2_doc, duration_s=1.0)
    assert set(v2.V2_TRUNK.coordinates) <= set(control.actuated_names)


def test_v1_only_operations_reject_v2_explicitly(v2_doc) -> None:
    from src.shared.python.motion_matching.pelvis_frame import relocate_pelvis

    with pytest.raises(ValueError, match="full-body-v1"):
        relocate_pelvis(v2_doc, (0.0, -0.09, 0.0), (0.0, 0.09, 0.0))


def test_exports_leave_the_v2_identity_unchanged(v2_doc, v2_bytes) -> None:
    assert canonical_sha256(json.loads(v2_bytes)) == canonical_sha256(v2_doc)
