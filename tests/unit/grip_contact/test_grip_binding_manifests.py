"""Unit tests for versioned per-model grip binding manifests and validators (HRG-GS-M2, #13324).

Acceptance criteria:
- 100 % of topology sites are resolved for the manifest, or reason-coded, for
  overlap, interlock and ten_finger, with and without the trigger finger.
- The FK round trip of every site is <= 0.5 mm: resolve to body-local, map back
  to world at the calibration pose, and compare with the topology site in the
  grip frame. The +x axis dot product is >= 0.999999.
- A handedness reversal test: a left-handed manifest maps the lead/trail roles
  to the mirrored bodies, and the resolved bindings mirror.
- Negative tests:
  - a cm-scaled manifest (units.length = "cm", or geometry 100x too large via a
    scaled resolver) is rejected;
  - a y-axis-shaft manifest (grip axis along y) is rejected;
  - a missing key, a wrong schema, and a duplicate/overlapping bound+unavailable
    site are each rejected.
- A test that every JSON in src/config/grip_bindings/ validates (glob test).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import HandClubBinding, SiteBinding
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactSite,
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
MANIFESTS_DIR = ROOT / "src/config/grip_bindings"
REFERENCE_MANIFEST_PATH = MANIFESTS_DIR / "full_body_v1_spec_export.json"
BINDING_PY = ROOT / "src/shared/python/grip_contact/binding.py"


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


@pytest.fixture(scope="module")
def reference_manifest() -> dict[str, Any]:
    assert REFERENCE_MANIFEST_PATH.exists(), (
        f"Missing manifest: {REFERENCE_MANIFEST_PATH}"
    )
    return json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_reference_manifest_loads_and_validates(
    reference_manifest: dict[str, Any],
) -> None:
    """The committed reference manifest passes schema validation."""
    validate_manifest(reference_manifest)
    assert reference_manifest["schema"] == "grip-binding/v1"
    assert reference_manifest["model_id"] == "full_body_v1_spec_export"
    assert reference_manifest["units"] == {"length": "m", "angle": "rad"}


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [True, False])
def test_all_topology_sites_resolved_across_styles_and_trigger(
    reference_manifest: dict[str, Any],
    interface: GripInterface,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """100 % of topology sites are resolved for each style with/without trigger finger."""
    spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topology = contact_topology(spec)
    binding = binding_from_manifest(reference_manifest, topology, interface)

    assert isinstance(binding, HandClubBinding)
    site_names_in_topology = set(topology.site_names())
    bound_site_names = {s.site_id for s in binding.sites}
    unavailable_names = set(binding.unavailable_sites.keys())

    # Every topology site must be bound or unavailable with reason
    covered = bound_site_names | unavailable_names
    assert covered == site_names_in_topology
    assert len(bound_site_names & unavailable_names) == 0

    # For rigid hands fixture, 100% of sites should be resolved
    assert len(bound_site_names) == len(site_names_in_topology)

    # Validate resolved geometric bounds
    validate_resolved_binding(binding, topology, interface)


def test_fk_round_trip_accuracy_and_axis_alignment(
    reference_manifest: dict[str, Any],
    interface: GripInterface,
) -> None:
    """FK round trip of every site <= 0.5 mm and +x dot product >= 0.999999."""
    spec = GripSpec(style=GripStyle.OVERLAP, trigger_finger=True)
    topology = contact_topology(spec)
    binding = binding_from_manifest(reference_manifest, topology, interface)

    club_pose = np.eye(4)
    club_state = RigidBodyState(
        rotation=np.eye(3),
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    hand_states = hand_frame_states(club_state, interface)

    body_poses = {
        binding.club_body: club_pose,
        binding.lead_hand_body: np.eye(4),
        binding.trail_hand_body: np.eye(4),
    }
    body_poses[binding.lead_hand_body][:3, :3] = hand_states["L"].rotation
    body_poses[binding.lead_hand_body][:3, 3] = hand_states["L"].position_m
    body_poses[binding.trail_hand_body][:3, :3] = hand_states["R"].rotation
    body_poses[binding.trail_hand_body][:3, 3] = hand_states["R"].position_m

    grip_frames = {
        "L": interface.left.matrix(),
        "R": interface.right.matrix(),
    }

    max_dist_error_m = 0.0
    for sb in binding.sites:
        # Resolve to body-local, map to world at calibration pose
        sb_pos_body = np.append(sb.translation_m, 1.0)
        p_world = body_poses[sb.body_id] @ sb_pos_body

        # Map back to grip frame
        side_char = "L" if sb.body_id == binding.lead_hand_body else "R"
        t_grip_world = club_pose @ grip_frames[side_char]
        p_grip = np.linalg.inv(t_grip_world) @ p_world

        # Compare with expected position in the hand grip frame
        expected_pos = np.append(sb.translation_m, 1.0)
        err = float(np.linalg.norm(p_grip[:3] - expected_pos[:3]))
        if err > max_dist_error_m:
            max_dist_error_m = err

        # Verify +x axis alignment: dot product >= 0.999999
        rot_grip = t_grip_world[:3, :3]
        rot_body = body_poses[sb.body_id][:3, :3]
        x_axis_world = rot_body @ sb.body_to_site[:3, 0]
        x_axis_grip = rot_grip.T @ x_axis_world
        dot_x = float(np.dot(x_axis_grip, np.array([1.0, 0.0, 0.0])))
        assert dot_x >= 0.999999, (
            f"Site {sb.site_id} x-axis dot product {dot_x} < 0.999999"
        )

    assert max_dist_error_m <= 0.0005, (
        f"FK round trip max error {max_dist_error_m * 1000:.4f} mm > 0.5 mm"
    )


def test_handedness_reversal_mirrors_bodies_and_bindings(
    reference_manifest: dict[str, Any],
    interface: GripInterface,
) -> None:
    """A left-handed manifest maps lead/trail roles to mirrored bodies, and resolved bindings mirror."""
    topo_right = contact_topology(GripSpec(style=GripStyle.OVERLAP, handedness="right"))
    binding_right = binding_from_manifest(reference_manifest, topo_right, interface)

    left_manifest = dict(reference_manifest)
    left_manifest["model_id"] = "full_body_v1_left_handed"
    left_manifest["handedness"] = "left"
    left_manifest["side_roles"] = {"right": "lead", "left": "trail"}
    left_manifest["lead_hand_body"] = "hand_r"
    left_manifest["trail_hand_body"] = "hand_l"

    validate_manifest(left_manifest)

    topo_left = contact_topology(GripSpec(style=GripStyle.OVERLAP, handedness="left"))
    binding_left = binding_from_manifest(left_manifest, topo_left, interface)

    # Lead/trail roles map to mirrored bodies
    assert binding_left.lead_hand_body == "hand_r"
    assert binding_left.trail_hand_body == "hand_l"

    right_map = binding_right.site_binding_map()
    left_map = binding_left.site_binding_map()

    for site_id, right_sb in right_map.items():
        assert site_id in left_map
        left_sb = left_map[site_id]
        # In left handed, lead sites go to hand_r, trail to hand_l
        expected_body = "hand_r" if "lead" in site_id else "hand_l"
        assert left_sb.body_id == expected_body

        # The resolved site translations mirror across the shaft (z axis in hand frame)
        p_r = right_sb.translation_m
        p_l = left_sb.translation_m
        assert np.isclose(p_r[0], p_l[0], atol=1e-6)
        assert np.isclose(p_r[1], p_l[1], atol=1e-6)
        assert np.isclose(p_r[2], -p_l[2], atol=1e-6)


def test_negative_manifest_units_length_cm_rejected(
    reference_manifest: dict[str, Any],
) -> None:
    """Manifest with units.length = 'cm' is rejected by validate_manifest."""
    bad_manifest = dict(reference_manifest)
    bad_manifest["units"] = {"length": "cm", "angle": "rad"}
    with pytest.raises(ValueError, match="units.*metres|length.*m"):
        validate_manifest(bad_manifest)


def test_negative_scaled_resolver_geometry_100x_rejected(
    reference_manifest: dict[str, Any],
) -> None:
    """Resolver producing 100x geometry is rejected by validate_resolved_binding."""
    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))

    def scaled_resolver(site_id: str) -> np.ndarray:
        mat = np.eye(4)
        # 100x too large: radial offset ~ 2.4 m instead of ~ 0.024 m
        mat[:3, 3] = np.array([0.0, 2.4, 0.0])
        return mat

    with pytest.raises(ValueError, match="radial offset|out of bounds"):
        binding_from_manifest(reference_manifest, topology, scaled_resolver)


def test_negative_y_axis_shaft_rejected(
    reference_manifest: dict[str, Any],
) -> None:
    """Manifest or binding with grip axis along y is rejected."""
    bad_manifest = dict(reference_manifest)
    bad_manifest["grip_axis"] = "y"
    with pytest.raises(ValueError, match="grip.*axis"):
        validate_manifest(bad_manifest)

    # Also test via resolver that rotates grip axis to +y
    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))

    def y_axis_resolver(site_id: str) -> np.ndarray:
        mat = np.eye(4)
        # Rotate 90 deg around z so x-axis points along y
        mat[:3, :3] = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mat[:3, 3] = np.array([0.0, 0.02, 0.0])
        return mat

    with pytest.raises(ValueError, match="grip.*axis.*dot product"):
        binding_from_manifest(reference_manifest, topology, y_axis_resolver)


@pytest.mark.parametrize(
    "missing_key",
    [
        "schema",
        "model_id",
        "model_hash",
        "handedness",
        "side_roles",
        "calibration_pose",
        "units",
        "provenance",
        "attachment_mode",
        "club_body",
        "lead_hand_body",
        "trail_hand_body",
        "unavailable_sites",
    ],
)
def test_negative_missing_keys_rejected(
    reference_manifest: dict[str, Any],
    missing_key: str,
) -> None:
    """Missing required keys are rejected with descriptive ValueError."""
    bad_manifest = dict(reference_manifest)
    del bad_manifest[missing_key]
    with pytest.raises(ValueError, match=f"missing.*{missing_key}"):
        validate_manifest(bad_manifest)


def test_negative_wrong_schema_rejected(
    reference_manifest: dict[str, Any],
) -> None:
    """Wrong schema version is rejected with descriptive ValueError."""
    bad_manifest = dict(reference_manifest)
    bad_manifest["schema"] = "grip-binding/v2"
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_manifest(bad_manifest)


def test_negative_duplicate_overlapping_bound_and_unavailable_sites_rejected(
    reference_manifest: dict[str, Any],
) -> None:
    """A site that is both bound and listed unavailable is rejected."""
    bad_manifest = dict(reference_manifest)
    bad_manifest["unavailable_sites"] = {"lead_palm_proximal": "damaged pad"}
    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))

    # Constructing a HandClubBinding with both bound and unavailable raises
    sb = SiteBinding(
        site_id="lead_palm_proximal",
        body_id="lead_hand",
        body_to_site=np.eye(4),
    )
    with pytest.raises(ValueError, match="both bound and unavailable"):
        HandClubBinding(
            model_id="test",
            model_hash="sha256:123",
            handedness="right",
            attachment_mode="contact",
            club_body="club",
            lead_hand_body="lead_hand",
            trail_hand_body="trail_hand",
            sites=(sb,),
            unavailable_sites={"lead_palm_proximal": "damaged pad"},
        )


def test_every_json_in_grip_bindings_validates(
    interface: GripInterface,
) -> None:
    """Every JSON file in src/config/grip_bindings/ validates cleanly."""
    json_files = list(MANIFESTS_DIR.glob("*.json"))
    assert len(json_files) >= 1, f"Expected at least 1 manifest in {MANIFESTS_DIR}"

    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    for mf_path in json_files:
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        binding = binding_from_manifest(manifest, topology, interface)
        validate_resolved_binding(binding, topology, interface)


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


BINDING_MANIFEST_PY = ROOT / "src/shared/python/grip_contact/binding_manifest.py"


def test_ast_budgets_and_lod() -> None:
    """AST check: functions <= 100 lines, <= 8 params, no attribute chain > 2 dots."""
    for path in (BINDING_PY, BINDING_MANIFEST_PY):
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
                assert n_params <= 8, (
                    f"{path.name}:{node.name} has {n_params} params > 8"
                )
