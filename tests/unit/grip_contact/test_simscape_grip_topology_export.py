"""Unit tests for Simscape GolfSwing3D grip topology export and allocated reporting (HRG-GS-M11, #13333).

Acceptance criteria:
1. Deterministic ``gs3dx_grip_topology.json`` export generated in Python with
   ``--check`` mode that fails on drift, preserving ``gs3dx_appearance.v1.json``
   byte-identical for v1 consumers.
2. Two runs give byte-identical output (SHA-256 hash determinism).
3. 100 % of topology pair IDs and site IDs are preserved in the export across all
   styles (overlap, interlock, ten_finger, with/without trigger finger), plus
   posed finger geometry from G4 ``build_grip_hand_mesh``.
4. Python allocation from ``simscape-closed-loop-snapshot/v1`` recorded states
   allocates pad and hand-hand forces via existing G3 ``hand_hand_allocation``
   and H1 ``palm_thumb`` with zero injected forces.
5. Thin MATLAB consumer ``gs3dx_read_grip_topology.m`` and deferred R2025b
   validation record naming ``C:/Program Files/MATLAB/R2025b/bin/matlab.exe``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import apply_report
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import (
    METHOD as ALLOCATION_LABEL,
    AllocatedHandContacts,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.palm_thumb import PalmThumbForce
from src.shared.python.grip_contact.simscape_grip_topology import (
    DEFAULT_GRIP_TOPOLOGY_JSON,
    GRIP_TOPOLOGY_SCHEMA_VERSION,
    SIMSCAPE_BINDING_MANIFEST_PATH,
    SimscapeAllocatedFrameReport,
    SimscapeAllocatedSeriesReport,
    SimscapeGripReportingAdapter,
    allocate_simscape_export,
    allocate_simscape_snapshot,
    build_grip_topology_payload,
    default_grip_interface,
    export_grip_topology,
    render_grip_topology_json,
    validate_grip_topology_payload,
)
from src.shared.python.loop_nullspace.simscape_export import (
    SCHEMA_ID as SIMSCAPE_SNAPSHOT_SCHEMA_ID,
    load_simscape_export,
    parse_simscape_export,
)
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import GripPose, Hand
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import (
    NAMED_GRIP_PRESETS,
    GripSpec,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APPEARANCE_V1_JSON = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "appearance"
    / "gs3dx_appearance.v1.json"
)
MATLAB_READER_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "tools"
    / "gs3dx_read_grip_topology.m"
)
DEFERRED_CATALOG_PATH = (
    REPO_ROOT / "docs" / "development" / "deferred-catalog-bundle.json"
)
PLANAR_FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "simscape"
    / "synthetic_planar_closed_loop_v1.json"
)
MODULE_PATH = (
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "grip_contact"
    / "simscape_grip_topology.py"
)
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_simscape_grip_topology.py"


def _build_synthetic_weld6_export_payload() -> dict[str, Any]:
    """Build a synthetic 6-DOF WELD6 ``simscape-closed-loop-snapshot/v1`` payload.

    Uses a 12-coordinate closed loop (6 lead + 6 trail actuated joints) where
    the trail hand closes onto the grip frame via a 6-row ``WELD6`` constraint.
    """
    n = 12
    m = 6
    coord_order = [f"lead_j{i}" for i in range(6)] + [f"trail_j{i}" for i in range(6)]
    mass_mat = np.diag([1.2, 1.1, 1.0, 0.5, 0.4, 0.3, 1.2, 1.1, 1.0, 0.5, 0.4, 0.3])
    selection = np.eye(n)
    # Canonical 6x12 closure Jacobian: V_trail - V_lead in grip frame
    j_lead = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.08, -0.04],
            [0.0, 1.0, 0.0, -0.08, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.04, -0.05, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    j_trail = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, -0.07, 0.03],
            [0.0, 1.0, 0.0, 0.07, 0.0, -0.06],
            [0.0, 0.0, 1.0, -0.03, 0.06, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    j_c = np.hstack([-j_lead, j_trail])
    assert j_c.shape == (m, n)

    q0 = np.zeros(n)
    v0 = np.zeros(n)
    # Acceleration in the velocity/acceleration nullspace of J_c so J_c a + Jdot_v = 0
    a0 = np.zeros(n)
    # Bias vector h that creates a realistic non-zero grip closure load across the hands
    h0 = np.array(
        [18.0, -25.0, 12.0, 1.4, -0.9, 0.8, -14.0, 22.0, -10.0, -1.1, 0.7, -0.6],
        dtype=float,
    )
    jdot_v0 = np.zeros(m)

    return {
        "schema": SIMSCAPE_SNAPSHOT_SCHEMA_ID,
        "provenance": {
            "provenance_kind": "synthetic_test_fixture",
            "model_scope": "verification_rig",
            "matlab_release": None,
            "matlab_version": None,
            "model_file": "synthetic_gs3dx_weld6_fixture",
            "model_sha256": None,
            "exported_at": None,
            "exporter": "test_simscape_grip_topology_export",
            "note": "Synthetic 6-DOF WELD6 closed-loop test fixture",
        },
        "topology": {
            "closure_kind": "WELD6",
            "closing_side": "trail",
            "holding_side": "lead",
            "coordinate_order": coord_order,
            "actuated_indices": list(range(n)),
            "unactuated_indices": [],
            "loop_support_indices": list(range(n)),
        },
        "inertias": {
            "bare_club": {
                "mass": 0.313,
                "com": [0.45, 0.0, 0.0],
                "inertia": [
                    [0.0005, 0.0, 0.0],
                    [0.0, 0.045, 0.0],
                    [0.0, 0.0, 0.045],
                ],
            },
            "welded_hand_inertias": [
                {
                    "mass": 0.42,
                    "com": [-0.08, 0.0, 0.0],
                    "inertia": [
                        [0.0005, 0.0, 0.0],
                        [0.0, 0.0008, 0.0],
                        [0.0, 0.0, 0.0008],
                    ],
                }
            ],
        },
        "states": [
            {
                "q": q0.tolist(),
                "v": v0.tolist(),
                "a": a0.tolist(),
                "M": mass_mat.tolist(),
                "h": h0.tolist(),
                "J_c": j_c.tolist(),
                "Jdot_v": jdot_v0.tolist(),
                "grip_frame": {
                    "rotation": np.eye(3).tolist(),
                    "origin": [0.0, 0.0, 0.0],
                },
            },
            {
                "q": q0.tolist(),
                "v": v0.tolist(),
                "a": a0.tolist(),
                "M": mass_mat.tolist(),
                "h": (1.25 * h0).tolist(),
                "J_c": j_c.tolist(),
                "Jdot_v": jdot_v0.tolist(),
                "grip_frame": {
                    "rotation": np.eye(3).tolist(),
                    "origin": [0.0, 0.0, 0.0],
                },
            },
        ],
    }


def test_grip_topology_schema_validates_and_hash_is_deterministic() -> None:
    """Two runs of the export produce byte-identical JSON and pass schema validation."""
    payload_1 = build_grip_topology_payload()
    payload_2 = build_grip_topology_payload()
    validate_grip_topology_payload(payload_1)
    validate_grip_topology_payload(payload_2)

    json_1 = render_grip_topology_json(payload_1)
    json_2 = render_grip_topology_json(payload_2)
    assert json_1 == json_2

    sha_1 = hashlib.sha256(json_1.encode("utf-8")).hexdigest()
    sha_2 = hashlib.sha256(json_2.encode("utf-8")).hexdigest()
    assert sha_1 == sha_2
    assert payload_1["schema_version"] == GRIP_TOPOLOGY_SCHEMA_VERSION


def test_committed_grip_topology_json_matches_generator_and_check_mode(
    tmp_path: Path,
) -> None:
    """Committed gs3dx_grip_topology.json is up-to-date and --check detects drift."""
    assert DEFAULT_GRIP_TOPOLOGY_JSON.is_file(), (
        f"Missing committed {DEFAULT_GRIP_TOPOLOGY_JSON}"
    )
    assert export_grip_topology(DEFAULT_GRIP_TOPOLOGY_JSON, check=True) == 0

    out_file = tmp_path / "gs3dx_grip_topology.json"
    assert export_grip_topology(out_file, check=False) == 0
    assert out_file.read_text(encoding="utf-8") == DEFAULT_GRIP_TOPOLOGY_JSON.read_text(
        encoding="utf-8"
    )
    assert export_grip_topology(out_file, check=True) == 0

    # Drift detection
    drifted = json.loads(out_file.read_text(encoding="utf-8"))
    drifted["model_id"] = "drifted_model_id"
    out_file.write_text(json.dumps(drifted, indent=2) + "\n", encoding="utf-8")
    assert export_grip_topology(out_file, check=True) == 1

    # Missing file fails check mode
    missing_file = tmp_path / "missing.json"
    assert export_grip_topology(missing_file, check=True) == 1


def test_v1_appearance_export_remains_byte_identical() -> None:
    """v1 consumer file gs3dx_appearance.v1.json remains 100% byte-identical."""
    from scripts.export_simscape_appearance import (
        DEFAULT_OUTPUT,
        export,
        main,
        render_json,
    )

    assert DEFAULT_OUTPUT == APPEARANCE_V1_JSON
    assert main(["--check"]) == 0
    expected = render_json(export())
    actual = APPEARANCE_V1_JSON.read_text(encoding="utf-8")
    assert actual == expected


@pytest.mark.parametrize(
    "style",
    [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER],
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_100_percent_topology_pair_and_site_ids_preserved_in_export(
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """100 % of ContactTopology pair IDs and site IDs are preserved in the export."""
    payload = build_grip_topology_payload()
    variant_key = f"{style.value}_trigger" if trigger_finger else style.value
    assert variant_key in payload["styles"]
    style_entry = payload["styles"][variant_key]

    topology = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    expected_pair_ids = list(topology.pair_names())
    expected_site_ids = list(topology.site_names())

    assert style_entry["pair_ids"] == expected_pair_ids
    exported_pair_ids = [p["id"] for p in style_entry["pairs"]]
    assert exported_pair_ids == expected_pair_ids
    assert set(exported_pair_ids) == set(expected_pair_ids)

    exported_site_ids = [s["name"] for s in style_entry["sites"]]
    assert exported_site_ids == expected_site_ids

    expected_hg = [p.name for p in topology.hand_grip_pairs()]
    expected_hh = [p.name for p in topology.hand_hand_pairs()]
    assert style_entry["hand_grip_pair_ids"] == expected_hg
    assert style_entry["hand_hand_pair_ids"] == expected_hh

    # Every pair in Simscape's welded/closed-loop export is marked allocated
    for p_entry in style_entry["pairs"]:
        assert p_entry["simscape_label"] == "allocated"


@pytest.mark.parametrize(
    "style",
    [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER],
)
def test_posed_finger_geometry_in_export_matches_g4_hand_mesh(
    style: GripStyle,
) -> None:
    """Exported posed finger geometry matches G4 build_grip_hand_mesh."""
    payload = build_grip_topology_payload()
    style_entry = payload["styles"][style.value]
    posed_hands = style_entry["posed_hands"]

    for hand_enum in (Hand.LEAD, Hand.TRAIL):
        hand_key = hand_enum.value
        assert hand_key in posed_hands
        exported_hand = posed_hands[hand_key]

        pose = GripPose(style=style, trigger_finger=False)
        mesh = build_grip_hand_mesh(
            hand_enum, grip_radius_m=payload["grip_radius_m"], pose=pose
        )
        assert len(exported_hand["components"]) == len(mesh.components)
        assert exported_hand["v_azimuth_deg"] == pytest.approx(mesh.v_azimuth_deg)
        assert exported_hand["thumb_azimuth_deg"] == pytest.approx(
            mesh.thumb_azimuth_deg
        )
        assert exported_hand["distance_below_butt_m"] == pytest.approx(
            mesh.distance_below_butt_m
        )

        for finger_name, tip_coords in exported_hand["fingertips_in_grip_m"].items():
            expected_tip = mesh.fingertip_in_grip_m(finger_name)
            np.testing.assert_allclose(tip_coords, expected_tip, atol=1e-8)


def test_presets_and_simscape_binding_manifest_validate() -> None:
    """Named presets and committed simscape_gs3dx_human.json manifest validate."""
    payload = build_grip_topology_payload()
    for preset_name in NAMED_GRIP_PRESETS:
        assert preset_name in payload["presets"]

    assert SIMSCAPE_BINDING_MANIFEST_PATH.is_file()
    manifest = json.loads(SIMSCAPE_BINDING_MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["model_id"] == "simscape_gs3dx_human"
    assert manifest["attachment_mode"] == "weld"

    interface = default_grip_interface()
    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    binding = binding_from_manifest(manifest, topology, interface)
    assert binding.attachment_mode == "weld"
    assert len(binding.sites) == len(topology.sites)


@pytest.mark.parametrize(
    "style",
    [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER],
)
def test_allocation_from_weld6_simscape_snapshot_zero_injected_forces(
    style: GripStyle,
) -> None:
    """Allocate pad and hand-hand forces from a WELD6 Simscape snapshot with zero injected forces."""
    raw_payload = _build_synthetic_weld6_export_payload()
    export_obj = parse_simscape_export(raw_payload)
    topology = contact_topology(GripSpec(style=style))

    series_report = allocate_simscape_export(
        export_obj,
        topology=topology,
        test_only=True,
    )
    assert isinstance(series_report, SimscapeAllocatedSeriesReport)
    assert len(series_report.frames) == 2
    assert series_report.injected_force_count == 0
    assert series_report.max_injected_force_n == 0.0
    assert series_report.max_injected_moment_nm == 0.0

    for frame_rep in series_report.frames:
        assert isinstance(frame_rep, SimscapeAllocatedFrameReport)
        assert frame_rep.label == ALLOCATION_LABEL
        assert isinstance(frame_rep.hand_hand_contacts, AllocatedHandContacts)
        assert isinstance(frame_rep.palm_thumb_contact, PalmThumbForce)
        assert frame_rep.hand_hand_contacts.available is True
        assert frame_rep.palm_thumb_contact.available is True

        # Zero injected forces check on both frame report and GripStepReport
        assert frame_rep.injected_body_wrenches == {}
        assert frame_rep.max_injected_force_n == 0.0
        assert frame_rep.max_injected_moment_nm == 0.0
        for force_vec, moment_vec in frame_rep.step_report.body_wrenches().values():
            np.testing.assert_allclose(force_vec, np.zeros(3), atol=1e-15)
            np.testing.assert_allclose(moment_vec, np.zeros(3), atol=1e-15)

        # 100 % of topology pairs are present and labelled "allocated"
        expected_pairs = set(topology.pair_names())
        assert set(frame_rep.pair_labels.keys()) == expected_pairs
        for p_name, lbl in frame_rep.pair_labels.items():
            assert lbl == "allocated", f"pair {p_name} must be allocated, got {lbl}"

        # Every hand-grip pad pair has finite, non-negative allocated normal force
        for pair in topology.hand_grip_pairs():
            fn = frame_rep.normal_force_n[pair.name]
            ft = frame_rep.tangential_force_n[pair.name]
            ms = frame_rep.spin_moment_nm[pair.name]
            assert np.isfinite(fn) and fn >= 0.0
            assert np.isfinite(ft) and ft >= 0.0
            assert np.isfinite(ms) and ms >= 0.0

        # Active hand-hand pairs reconstruct closure wrench within 1e-9 N
        assert frame_rep.hand_hand_contacts.reproduction_error_n is not None
        assert frame_rep.hand_hand_contacts.reproduction_error_n <= 1e-9


def test_allocation_from_checked_in_planar_simscape_fixture() -> None:
    """Allocate forces from the checked-in synthetic_planar_closed_loop_v1.json fixture."""
    assert PLANAR_FIXTURE_PATH.is_file()
    export_obj = load_simscape_export(PLANAR_FIXTURE_PATH)
    topology = contact_topology(GripSpec(style=GripStyle.OVERLAP))

    series_report = allocate_simscape_export(
        export_obj,
        topology=topology,
        test_only=True,
    )
    assert len(series_report.frames) == len(export_obj.states)
    assert series_report.injected_force_count == 0
    assert series_report.max_injected_force_n == 0.0
    for frame_rep in series_report.frames:
        assert frame_rep.injected_body_wrenches == {}
        assert frame_rep.hand_hand_contacts.available is True
        assert frame_rep.palm_thumb_contact.available is True


def test_simscape_reporting_adapter_forbids_wrench_injection() -> None:
    """SimscapeGripReportingAdapter never feeds a force back into the model."""
    adapter = SimscapeGripReportingAdapter()
    caps = adapter.capabilities()
    assert caps.supports_wrench_application is False
    assert caps.supports_physical_pairs is False
    assert caps.native_contact is False

    raw_payload = _build_synthetic_weld6_export_payload()
    export_obj = parse_simscape_export(raw_payload)
    frame_rep = allocate_simscape_export(export_obj, test_only=True).frames[0]

    with pytest.raises(SchemaFeatureUnavailableError):
        apply_report(adapter, frame_rep.step_report)

    with pytest.raises(SchemaFeatureUnavailableError):
        adapter.apply_wrenches({"club": (np.ones(3), np.zeros(3))})


def test_matlab_thin_consumer_and_deferred_r2025b_validation() -> None:
    """Thin MATLAB reader exists with no law re-implementation and R2025b deferral is cataloged."""
    assert MATLAB_READER_PATH.is_file(), f"Missing {MATLAB_READER_PATH}"
    m_code = MATLAB_READER_PATH.read_text(encoding="utf-8")
    assert "gs3dx_grip_topology.json" in m_code
    assert "jsondecode" in m_code
    assert GRIP_TOPOLOGY_SCHEMA_VERSION in m_code
    # Must not re-implement Hunt-Crossley or Coulomb contact laws in MATLAB
    for forbidden in ("sphere_capsule", "capsule_capsule", "hunt_crossley", "stribeck"):
        assert forbidden not in m_code.lower()

    assert DEFERRED_CATALOG_PATH.is_file()
    catalog = json.loads(DEFERRED_CATALOG_PATH.read_text(encoding="utf-8"))
    items = catalog.get("deferred_items", [])
    m11_items = [
        item
        for item in items
        if item.get("id") == "HRG-GS-M11" or item.get("issue") == 13333
    ]
    assert len(m11_items) == 1, "Expected exactly one HRG-GS-M11 (#13333) deferred item"
    entry = m11_items[0]
    assert entry["issue"] == 13333
    assert entry["status"] == "deferred"
    assert "C:/Program Files/MATLAB/R2025b/bin/matlab.exe" in entry["executable"]
    assert "R2025b" in entry["host_requirement"]


def test_negative_validation_errors() -> None:
    """Invalid payloads and inputs raise TypeError or ValueError."""
    with pytest.raises(TypeError):
        validate_grip_topology_payload("not_a_dict")  # type: ignore[arg-type]

    bad_schema = build_grip_topology_payload()
    bad_schema["schema_version"] = "simscape-grip-topology/v999"
    with pytest.raises(ValueError, match="schema_version"):
        validate_grip_topology_payload(bad_schema)

    missing_styles = build_grip_topology_payload()
    missing_styles["styles"] = {}
    with pytest.raises(ValueError, match="styles"):
        validate_grip_topology_payload(missing_styles)

    with pytest.raises(TypeError):
        allocate_simscape_snapshot("not_a_snapshot")  # type: ignore[arg-type]


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


def test_ast_quality_gates() -> None:
    """Module and script obey <= 1200 lines, <= 100 lines/fn, <= 8 params, LOD <= 2."""
    for path in (MODULE_PATH, SCRIPT_PATH):
        assert path.is_file(), f"Missing {path}"
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
