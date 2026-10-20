"""Humanoid character builder and model-library grip binding admission tests (HRG-GS-M13, #13335).

Acceptance criteria:
1. When a club is composed in spec_export, model_bundle, full_body_mjcf,
   character_builder, or model_converter.urdf_exporter, a valid grip-binding/v1
   manifest is emitted and validated via binding_manifest.
2. A clubless export that requests grip binding fails closed with
   SchemaFeatureUnavailableError and a non-empty reason (never an empty manifest).
3. Model-library admission: every advertised model-library asset with hands
   (M0 engine inventory + launcher model tiles + bundled ModelLibrary manifest)
   is either bound (manifest exists and validates) or reason-coded in
   grip_capability_inventory.json (100% coverage).
4. Neutral preload of exported URDF/MJCF is within +-0.5 mm of
   DEFAULT_TOPOLOGY_PRELOAD_M (or layout preload) across all grip styles.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import (
    HandClubBinding,
    evaluate_contacts,
)
from src.shared.python.grip_contact.binding_manifest import (
    MANIFEST_SCHEMA,
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.capability_inventory import (
    enumerate_advertised_hand_bearing_assets,
    load_grip_capability_inventory,
    verify_model_library_admission,
)
from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import (
    build_topology_model,
)
from src.shared.python.humanoid_character_builder import spec_export
from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from src.shared.python.humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
    generate_humanoid_urdf,
)
from src.shared.python.model_generation.export.model_bundle import (
    export_full_body_urdf,
    export_model_bundle,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec
from src.tools.character_builder.core import CharacterBuilderModel
from src.tools.model_converter.schema_validator import validate_canonical_model
from src.tools.model_converter.urdf_exporter import (
    build_canonical_grip_binding_manifest,
    export_urdf,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_YAML = (
    ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)


# ---------------------------------------------------------------------------
# 1. Composed club exports emit valid grip-binding/v1 manifests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_id", ["tour_average_male", "junior", "anthro_driver"])
def test_spec_export_emits_valid_grip_binding_manifest(preset_id: str) -> None:
    """spec_export builds and embeds a valid grip-binding/v1 manifest when club is composed."""
    character = spec_export.compile_character(preset_id, {})
    manifest = spec_export.build_grip_binding_manifest(character)
    validate_manifest(manifest)

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["model_hash"] == f"sha256:{character.sha256}"
    assert manifest["club_body"] == "club"
    assert manifest["lead_hand_body"] == "hand_l"
    assert manifest["trail_hand_body"] == "hand_r"

    summary = spec_export.build_summary(character)
    assert "grip_binding" in summary
    assert summary["grip_binding"] == manifest

    interface = GripInterface.from_spec(character.document)
    for style in GripStyle:
        for trig in (False, True):
            topo = contact_topology(GripSpec(style=style, trigger_finger=trig))
            binding = binding_from_manifest(manifest, topo, interface)
            assert isinstance(binding, HandClubBinding)
            validate_resolved_binding(binding, topo, interface)
            assert len(binding.sites) == len(topo.sites)


def test_exporters_embed_and_write_grip_binding_sidecar(tmp_path: Path) -> None:
    """model_bundle, full_body_urdf, full_body_mjcf, character_builder, and urdf_exporter emit grip-binding/v1."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    character = spec_export.compile_character("tour_average_male", {})
    spec_bytes = character.json_text.encode("utf-8")

    # 1. export_model_bundle
    bundle = export_model_bundle(spec_bytes)
    assert bundle.sidecar is not None
    assert "grip_binding" in bundle.sidecar
    validate_manifest(bundle.sidecar["grip_binding"])
    assert (
        bundle.sidecar["grip_binding"]["model_hash"]
        == f"sha256:{bundle.manifest.model_sha256}"
    )

    # 2. export_full_body_urdf
    _, urdf_meta = export_full_body_urdf(spec_bytes)
    assert "grip_binding" in urdf_meta
    validate_manifest(urdf_meta["grip_binding"])

    # 3. export_full_body_mjcf
    _, mjcf_meta = export_full_body_mjcf(spec_bytes)
    assert "grip_binding" in mjcf_meta
    validate_manifest(mjcf_meta["grip_binding"])

    # 4. CharacterBuilderModel.export writes sidecar <stem>.grip_binding.json
    cb_model = CharacterBuilderModel("junior")
    urdf_path = cb_model.export("urdf", tmp_path)
    sidecar_gb_path = tmp_path / f"{urdf_path.stem}.grip_binding.json"
    assert sidecar_gb_path.is_file()
    gb_data = json.loads(sidecar_gb_path.read_text(encoding="utf-8"))
    validate_manifest(gb_data)

    direct_gb_path = cb_model.export_grip_binding(tmp_path / "direct")
    assert direct_gb_path.is_file()
    validate_manifest(json.loads(direct_gb_path.read_text(encoding="utf-8")))

    # 5. model_converter.urdf_exporter with composed club_shaft
    canonical = validate_canonical_model(CANONICAL_YAML)
    canon_manifest = build_canonical_grip_binding_manifest(canonical)
    validate_manifest(canon_manifest)
    assert canon_manifest["club_body"] == "club_shaft"
    assert canon_manifest["lead_hand_body"] == "hand_left"
    assert canon_manifest["trail_hand_body"] == "hand_right"

    canon_urdf_path = tmp_path / "golfer.urdf"
    export_urdf(canonical, out_path=canon_urdf_path, emit_grip_binding=True)
    canon_gb_path = tmp_path / "golfer.grip_binding.json"
    assert canon_gb_path.is_file()
    validate_manifest(json.loads(canon_gb_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 2. Clubless exports fail closed with SchemaFeatureUnavailableError
# ---------------------------------------------------------------------------


def test_clubless_spec_export_fails_closed() -> None:
    """A clubless spec document requesting grip binding fails closed with SchemaFeatureUnavailableError."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    character = spec_export.compile_character("tour_average_male", {})
    clubless_doc = dict(character.document)
    clubless_doc.pop("club", None)
    clubless_doc.pop("closure", None)

    with pytest.raises(SchemaFeatureUnavailableError) as exc_info:
        spec_export.build_grip_binding_manifest(clubless_doc)
    assert getattr(exc_info.value, "reason", "").strip() != ""
    assert "club" in str(exc_info.value).lower()

    clubless_bytes = json.dumps(clubless_doc).encode("utf-8")
    with pytest.raises(SchemaFeatureUnavailableError) as exc_bundle:
        export_model_bundle(clubless_bytes, require_grip_binding=True)
    assert getattr(exc_bundle.value, "reason", "").strip() != ""

    with pytest.raises(SchemaFeatureUnavailableError) as exc_mjcf:
        export_full_body_mjcf(clubless_bytes, require_grip_binding=True)
    assert getattr(exc_mjcf.value, "reason", "").strip() != ""


def test_clubless_humanoid_urdf_generator_fails_closed() -> None:
    """Standalone clubless HumanoidURDFGenerator fails closed when grip binding is requested."""
    params = BodyParameters(height_m=1.80, mass_kg=80.0)
    gen = HumanoidURDFGenerator(URDFGeneratorConfig(require_grip_binding=True))
    with pytest.raises(SchemaFeatureUnavailableError) as exc_info:
        gen.generate(params)
    assert getattr(exc_info.value, "reason", "").strip() != ""

    with pytest.raises(SchemaFeatureUnavailableError) as exc_fn:
        generate_humanoid_urdf(params, require_grip_binding=True)
    assert getattr(exc_fn.value, "reason", "").strip() != ""


def test_clubless_canonical_urdf_exporter_fails_closed() -> None:
    """CanonicalModel without club_shaft fails closed when grip binding is requested."""
    import dataclasses

    canonical = validate_canonical_model(CANONICAL_YAML)
    clubless = dataclasses.replace(
        canonical,
        segments=tuple(
            s for s in canonical.segments if s.name not in ("club_shaft", "club_head")
        ),
    )
    with pytest.raises(SchemaFeatureUnavailableError) as exc_info:
        build_canonical_grip_binding_manifest(clubless)
    assert getattr(exc_info.value, "reason", "").strip() != ""

    with pytest.raises(SchemaFeatureUnavailableError):
        export_urdf(clubless, require_grip_binding=True)


# ---------------------------------------------------------------------------
# 3. Model-library admission (100% coverage of advertised hand-bearing assets)
# ---------------------------------------------------------------------------


def test_model_library_admission_100_percent_coverage() -> None:
    """Every advertised hand-bearing asset across engine inventory, launcher tiles, and bundled library is bound or reason-coded."""
    inv = load_grip_capability_inventory()
    advertised = enumerate_advertised_hand_bearing_assets(ROOT)
    assert len(advertised) >= 18
    assert "humanoid_character_builder" in advertised
    assert "character_builder" in advertised
    assert "mujoco_humanoid" in advertised

    admission = verify_model_library_admission(inv, ROOT)
    assert set(admission.keys()) == set(advertised)
    for asset_id, status in admission.items():
        assert status in ("bound", "reason_coded"), (
            f"Asset {asset_id!r} has invalid admission status {status!r}"
        )

    # Specifically humanoid_character_builder has a committed binding manifest and available pad grip
    assert admission["humanoid_character_builder"] == "bound"
    cb_cap = inv.models["humanoid_character_builder"]
    assert cb_cap.features["physical_pad_grip"].available is True

    # Handless bundled library models are explicitly excluded with reasons
    for handless_id in ("simple_humanoid", "simple_arm", "simple_quadruped"):
        assert inv.is_excluded(handless_id), (
            f"Handless bundled model {handless_id} must be excluded"
        )
        excl = inv.get_excluded(handless_id)
        assert excl is not None and bool(excl.reason.strip())


# ---------------------------------------------------------------------------
# 4. Neutral preload within +-0.5 mm on shared geometry, URDF, and MJCF
# ---------------------------------------------------------------------------


def _assert_neutral_preload_at_club_pose(
    character: spec_export.CompiledCharacter,
    manifest: dict[str, Any],
    club_pose: np.ndarray,
) -> None:
    """Verify every bound pad's penetration at neutral pose is within +-0.5 mm of DEFAULT_TOPOLOGY_PRELOAD_M."""
    interface = GripInterface.from_spec(character.document)
    club_state = RigidBodyState(
        rotation=club_pose[:3, :3],
        position_m=club_pose[:3, 3],
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    hand_states = hand_frame_states(club_state, interface)
    body_states = {
        manifest["club_body"]: club_state,
        manifest["lead_hand_body"]: hand_states["L"],
        manifest["trail_hand_body"]: hand_states["R"],
    }

    for style in GripStyle:
        for trig in (False, True):
            topo = contact_topology(GripSpec(style=style, trigger_finger=trig))
            binding = binding_from_manifest(manifest, topo, interface)
            topo_model = build_topology_model(interface, topo)
            report = evaluate_contacts(topo_model, binding, body_states)
            assert len(report.pair_wrenches) > 0
            for pair_name, pw in report.pair_wrenches.items():
                assert abs(pw.penetration_m - DEFAULT_TOPOLOGY_PRELOAD_M) <= 5.0e-4, (
                    f"{style.value} trig={trig} {pair_name}: penetration "
                    f"{pw.penetration_m * 1e3:.4f} mm outside +-0.5 mm of "
                    f"{DEFAULT_TOPOLOGY_PRELOAD_M * 1e3:.4f} mm"
                )


def test_neutral_preload_shared_geometry_within_half_mm() -> None:
    """Neutral preload of bound pads is within +-0.5 mm in shared geometry."""
    character = spec_export.compile_character("tour_average_male", {})
    manifest = spec_export.build_grip_binding_manifest(character)
    _assert_neutral_preload_at_club_pose(character, manifest, np.eye(4))


@pytest.mark.requires_pinocchio
def test_neutral_preload_exported_urdf_pinocchio() -> None:
    """Neutral preload on the exported URDF (evaluated via Pinocchio FK) is within +-0.5 mm."""
    pin = pytest.importorskip("pinocchio")
    character = spec_export.compile_character("tour_average_male", {})
    bundle = export_model_bundle(character.json_text.encode("utf-8"))
    assert bundle.sidecar is not None
    manifest = bundle.sidecar["grip_binding"]
    body_links = bundle.sidecar["body_links"]
    closure = bundle.sidecar["closure"]

    model = pin.buildModelFromXML(bundle.urdf_xml)
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    club_spec_body = closure["body_b"]
    trail_spec_body = closure["body_a"]
    for spec_body in (club_spec_body, trail_spec_body):
        assert spec_body in body_links
        link_name = body_links[spec_body]
        assert model.existFrame(link_name)

    club_link = body_links[club_spec_body]
    t_club = np.array(data.oMf[model.getFrameId(club_link)].homogeneous, dtype=float)
    _assert_neutral_preload_at_club_pose(character, manifest, t_club)


@pytest.mark.requires_mujoco
def test_neutral_preload_exported_mjcf_mujoco() -> None:
    """Neutral preload on the exported MJCF (evaluated via MuJoCo FK) is within +-0.5 mm."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    character = spec_export.compile_character("tour_average_male", {})
    xml, meta = export_full_body_mjcf(character.json_text.encode("utf-8"))
    manifest = meta["grip_binding"]
    closure = character.document["closure"]

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    club_spec_body = closure["body_b"]
    trail_spec_body = closure["body_a"]
    for spec_body in (club_spec_body, trail_spec_body):
        body_id = model.body(spec_body).id
        assert body_id >= 0

    club_id = model.body(club_spec_body).id
    t_club = np.eye(4)
    t_club[:3, :3] = data.xmat[club_id].reshape(3, 3)
    t_club[:3, 3] = data.xpos[club_id]
    _assert_neutral_preload_at_club_pose(character, manifest, t_club)


# ---------------------------------------------------------------------------
# 5. AST size & Law of Demeter budgets on touched files
# ---------------------------------------------------------------------------


def _attribute_chain_depth(node: ast.AST) -> int:
    depth = 0
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            depth += 1
            cur = cur.value
        elif isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            break
    return depth


TOUCHED_FILES = (
    ROOT / "src/shared/python/grip_contact/binding_manifest.py",
    ROOT / "src/shared/python/grip_contact/capability_inventory.py",
    ROOT / "src/shared/python/humanoid_character_builder/spec_export.py",
    ROOT / "src/tools/character_builder/core.py",
    ROOT / "src/tools/model_converter/urdf_exporter.py",
)


@pytest.mark.parametrize("path", TOUCHED_FILES, ids=lambda p: p.name)
def test_touched_files_size_and_lod_budgets(path: Path) -> None:
    """All touched source files obey <= 1200 lines, functions <= 100 lines and <= 8 params, LOD <= 2 dots."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= 1200, f"{path.name} has {len(lines)} lines > 1200"

    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            depth = _attribute_chain_depth(node)
            assert depth <= 2, (
                f"{path.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                f"{ast.unparse(node)}"
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.end_lineno is not None
            length = node.end_lineno - node.lineno + 1
            assert length <= 100, f"{path.name}:{node.name} is {length} lines > 100"
            args = node.args
            n_params = (
                len(args.posonlyargs)
                + len(args.args)
                + len(args.kwonlyargs)
                + (1 if args.vararg else 0)
                + (1 if args.kwarg else 0)
            )
            if args.args and args.args[0].arg in ("self", "cls"):
                n_params -= 1
            assert n_params <= 8, f"{path.name}:{node.name} has {n_params} params > 8"
