"""MyoSuite golfer scene composition (MS-51, #10344).

Composes the pinned MyoHub ``myo_sim`` MyoBody-simple-upper MJCF with a
club from :mod:`club_models`, dual-grip weld sites, and four foot contact
spheres documented against the shared Hunt–Crossley contact parameters.

Design contracts
----------------
- Public entry points validate preconditions and fail closed when the
  pinned ``myo_sim`` checkout is missing.
- ``myosuite`` / MuJoCo SDKs are **not** imported here (Law of Demeter);
  native load lives in tests and ``model_inventory`` native helpers.
- Scene generation is pure XML composition over existing assets (DRY).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import enum
import hashlib
import json
import logging
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final

import numpy as np
from numpy import linalg as npla

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    SchemaFeatureUnavailableError,
    evaluate_contacts,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import (
    AllocatedPairForce,
    AllocationPrior,
    allocate_pair_force,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    map_site_to_hand_frame,
)
from src.shared.python.loop_nullspace.evidence import (
    GripFrame as NullspaceGripFrame,
)
from src.shared.python.loop_nullspace.grip_split import grasp_map
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.motion_matching.contact_law import ContactParameters
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    resolve_visual_pose,
)

try:
    import mujoco

    from src.engines.physics_engines.mujoco.python.grip_efc import (
        DEFAULT_GRIP_WELDS,
        grip_analysis_from_efc,
    )
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        WeldedGripBodyError,
        _calibration_grip_frames,
        _sample_body_pose_and_vel,
        _state_to_pose_and_vel,
    )
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]
    DEFAULT_GRIP_WELDS = {"L": "grip_weld_l", "R": "grip_weld_r"}

    class WeldedGripBodyError(SchemaFeatureUnavailableError, ValueError):  # type: ignore[no-redef]
        """Raised when physical wrenches or modes are requested on welded grip bodies."""


logger = logging.getLogger(__name__)

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[5]
_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_MANIFESTS_DIR: Final[Path] = REPO_ROOT_DEFAULT / "src" / "config" / "grip_bindings"
ADMITTED_MYOSUITE_GOLFER_MODELS: Final[tuple[str, ...]] = (
    "myosuite_golfer_driver",
    "myosuite_golfer_iron",
)
_MYOSUITE_MODEL_ID_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "driver": "myosuite_golfer_driver",
        "golfer_myobody_driver": "myosuite_golfer_driver",
        "iron": "myosuite_golfer_iron",
        "golfer_myobody_iron": "myosuite_golfer_iron",
    }
)
MYOSUITE_MANIFEST_PATHS: Final[Mapping[str, Path]] = MappingProxyType(
    {
        model_id: _MANIFESTS_DIR / f"{model_id}.json"
        for model_id in ADMITTED_MYOSUITE_GOLFER_MODELS
    }
)
_MYOSUITE_CONSUMER: Final[str] = "myosuite.golfer_scene"
_MYOSUITE_CHILD: Final[str] = "HRG-GS-M7"
_MYOSUITE_ISSUE: Final[int] = 13329

# Scenes live at shared/models/myosuite/golf/body so nested myo_sim includes
# that use ../../myo_sim/... resolve the same way as stock body/*.xml.
_DEFAULT_MODELS_DIR = (
    REPO_ROOT_DEFAULT / "shared" / "models" / "myosuite" / "golf" / "body"
)
COORDINATE_MAP_PATH = Path(__file__).with_name("coordinate_map_anthro.json")

# Gitlink recorded for shared/models/myosuite/myo_sim (not .gitmodules).
MYO_SIM_PIN_SHA = "33f3ded946f55adbdcf963c99999587aadaf975f"
MYO_SIM_RELATIVE = Path("shared/models/myosuite/myo_sim")

_PLACEHOLDER_MYOBODY = (
    _ENGINE_ROOT.parent / "mujoco" / "myo_sim" / "body" / "myobody.xml"
)

# Four MS-51 foot contacts (heel + forefoot per foot); not the six-sphere
# anthro document set. Positions are in the calcn body frame (metres).
_FOOT_CONTACTS: tuple[tuple[str, str, tuple[float, float, float], float], ...] = (
    ("heel_r", "calcn_r", (0.01, -0.005, 0.0), 0.035),
    ("forefoot_r", "calcn_r", (0.16, -0.005, 0.0), 0.03),
    ("heel_l", "calcn_l", (0.01, -0.005, 0.0), 0.035),
    ("forefoot_l", "calcn_l", (0.16, -0.005, 0.0), 0.03),
)

_DEFAULT_CONTACT = ContactParameters(
    stiffness_n_m=50_000.0,
    dissipation_s_m=2.0,
    static_friction=0.9,
    dynamic_friction=0.8,
    viscous_friction=0.0,
    transition_velocity_m_s=0.05,
)


class ClubKind(enum.Enum):
    """Flagship club variants for generated golfer scenes."""

    DRIVER = "driver"
    IRON = "iron"


@dataclass(frozen=True)
class GolferScenePaths:
    """Paths written by :func:`generate_golfer_scene`."""

    driver: Path
    iron: Path
    receipt: Path


@dataclass(frozen=True)
class GolferScene:
    """Resolved scene assets for kinematic replay and inventory smoke."""

    xml_path: Path
    marker_sites: dict[str, dict[str, Any]]
    topology_note: str
    parity_budget_qualified: bool = False
    club: ClubKind = ClubKind.DRIVER
    myo_sim_pin: str = MYO_SIM_PIN_SHA

    @property
    def is_placeholder(self) -> bool:
        name = _path_basename_lower(self.xml_path)
        return name in {"myobody.xml", "myoupperbody.xml"} or "placeholder" in name


def _repo_root(repo_root: Path | None) -> Path:
    return Path(repo_root) if repo_root is not None else REPO_ROOT_DEFAULT


def _myo_sim_root(repo_root: Path) -> Path:
    return repo_root / MYO_SIM_RELATIVE


def _require_myo_sim(repo_root: Path) -> Path:
    root = _myo_sim_root(repo_root)
    body = root / "body" / "myobody_simpleupper.xml"
    if not body.is_file():
        raise FileNotFoundError(
            f"Pinned myo_sim missing at {root} (expected pin {MYO_SIM_PIN_SHA}). "
            "Run scripts/setup_myosuite_models.ps1 (or .sh)."
        )
    return root


def _rel_uri(from_dir: Path, target: Path) -> str:
    """POSIX relative path from ``from_dir`` to ``target`` for MJCF includes."""
    import os

    return Path(
        os.path.relpath(Path(target).resolve(), Path(from_dir).resolve())
    ).as_posix()


def _club_spec(kind: ClubKind) -> ClubSpec:
    if kind is ClubKind.DRIVER:
        return CLUBS["driver"]
    return CLUBS["iron7"]


def _inject_hand_grip_sites(arm_chain_text: str, hand_name: str, site_name: str) -> str:
    """Insert a grip site immediately inside the named hand body."""
    pattern = rf'(<body name="{re.escape(hand_name)}"[^>]*>)'
    replacement = (
        rf'\1\n                    <site name="{site_name}" '
        rf'pos="0 -0.05 0" size="0.005"/>'
    )
    updated, n = re.subn(pattern, replacement, arm_chain_text, count=1)
    if n != 1:
        raise ValueError(f"Failed to inject grip site into {hand_name}")
    return updated


def _write_arm_chains_with_grip_sites(
    myo_sim: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Copy simple arm chains and inject dual-grip sites on the hands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    right_src = myo_sim / "arm" / "assets" / "myoarm_simpleR_chain.xml"
    left_src = myo_sim / "arm" / "assets" / "myoarm_simpleL_chain.xml"
    right_dst = out_dir / "myoarm_simpleR_chain_grip.xml"
    left_dst = out_dir / "myoarm_simpleL_chain_grip.xml"
    right_dst.write_text(
        _inject_hand_grip_sites(
            right_src.read_text(encoding="utf-8"), "hand_r", "grip_site_hand_r"
        ),
        encoding="utf-8",
    )
    left_dst.write_text(
        _inject_hand_grip_sites(
            left_src.read_text(encoding="utf-8"), "hand_l", "grip_site_hand_l"
        ),
        encoding="utf-8",
    )
    return right_dst, left_dst


def _write_upper_chain_with_local_arms(
    myo_sim: Path, out_dir: Path, right_arm: Path, left_arm: Path
) -> Path:
    """Rewrite myoupperbody_chain includes to the grip-injected arm chains."""
    src = myo_sim / "body" / "assets" / "myoupperbody_chain.xml"
    text = src.read_text(encoding="utf-8")
    text = text.replace(
        'file="../../myo_sim/arm/assets/myoarm_simpleR_chain.xml"',
        f'file="{right_arm.name}"',
    )
    text = text.replace(
        'file="../../myo_sim/arm/assets/myoarm_simpleL_chain.xml"',
        f'file="{left_arm.name}"',
    )
    # Keep stock ../../myo_sim/head/... path so MuJoCo resolves it against the
    # main scene directory (golf/body), matching myoupperbody_assets includes.
    dst = out_dir / "myoupperbody_chain_grip.xml"
    dst.write_text(text, encoding="utf-8")
    return dst


def _club_grip_site(name: str, club: ClubSpec, hand: Hand) -> str:
    """Grip site on the club body (origin at the head, shaft toward -y).

    The hand sits ``distance_below_butt_m`` below the butt, so its distance
    from the head is the club length minus that (shared grip pose, OSV-2).
    """
    y = -(club.length_m - DEFAULT_GRIP_POSE.distance_below_butt_m(hand))
    return f'      <site name="{name}" pos="0 {y:.6g} 0" size="0.006"/>'


def _worldbody_addons(club: ClubSpec, contact: ContactParameters) -> str:
    """Club + contact sphere bodies placed inside ``<worldbody>``."""
    head = club.head_half_size_m
    head_type = "ellipsoid" if club.head_shape == "ellipsoid" else "box"
    solref = f"{1.0 / contact.stiffness_n_m:.6g} {contact.dissipation_s_m:.6g}"
    friction = (
        f"{contact.static_friction:.6g} {contact.dynamic_friction:.6g} "
        f"{contact.viscous_friction:.6g}"
    )
    contact_bodies: list[str] = []
    for name, _parent, _pos, radius in _FOOT_CONTACTS:
        contact_bodies.append(
            "\n".join(
                [
                    f'    <body name="ud_contact_{name}">',
                    "      <freejoint/>",
                    (
                        f'      <geom name="contact_{name}" type="sphere" '
                        f'size="{radius:.6g}" mass="0.001" '
                        f'rgba="0.2 0.8 0.3 0.35" '
                        f'contype="0" conaffinity="0" '
                        f'solref="{solref}" friction="{friction}"/>'
                    ),
                    "    </body>",
                ]
            )
        )
    club_y = -club.length_m
    return "\n".join(
        [
            '    <body name="golf_club" pos="0.3 0 1.0">',
            '      <freejoint name="club_free"/>',
            (
                f'      <geom name="club_shaft" type="capsule" '
                f'fromto="0 {club_y:.6g} 0 0 0 0" '
                f'size="{club.shaft_radius_m:.6g}" '
                f'mass="{club.shaft_mass_kg + club.grip_mass_kg:.6g}" '
                f'rgba="0.75 0.75 0.8 1"/>'
            ),
            (
                f'      <geom name="club_head" type="{head_type}" '
                f'size="{head[0]:.6g} {head[1]:.6g} {head[2]:.6g}" '
                f'pos="0 0 0" mass="{club.head_mass_kg:.6g}" '
                f'rgba="0.1 0.1 0.15 1"/>'
            ),
            _club_grip_site("grip_site_club_r", club, Hand.TRAIL),
            _club_grip_site("grip_site_club_l", club, Hand.LEAD),
            "    </body>",
            *contact_bodies,
        ]
    )


def _equality_fragment() -> str:
    """Dual-grip site welds plus foot-contact body welds (outside worldbody)."""
    contact_welds: list[str] = []
    for name, parent, pos, _radius in _FOOT_CONTACTS:
        px, py, pz = pos
        contact_welds.append(
            f'    <weld name="contact_weld_{name}" body1="{parent}" '
            f'body2="ud_contact_{name}" '
            f'relpose="{px:.6g} {py:.6g} {pz:.6g} 1 0 0 0"/>'
        )
    return "\n".join(
        [
            "  <equality>",
            (
                '    <weld name="grip_weld_r" site1="grip_site_hand_r" '
                'site2="grip_site_club_r"/>'
            ),
            (
                '    <weld name="grip_weld_l" site1="grip_site_hand_l" '
                'site2="grip_site_club_l"/>'
            ),
            *contact_welds,
            "  </equality>",
        ]
    )


def _build_scene_xml(
    *,
    myo_sim: Path,
    out_dir: Path,
    upper_chain: Path,
    club: ClubSpec,
    model_name: str,
    contact: ContactParameters,
) -> str:
    """Compose a golfer MJCF mirroring myobody_simpleupper + club/contacts."""
    meshdir = _rel_uri(out_dir, myo_sim)
    scene = _rel_uri(out_dir, myo_sim / "scene" / "myosuite_scene.xml")
    upper_assets = _rel_uri(
        out_dir, myo_sim / "body" / "assets" / "myoupperbody_assets.xml"
    )
    leg_assets = _rel_uri(out_dir, myo_sim / "leg" / "assets" / "myolegs_assets.xml")
    leg_tendon = _rel_uri(out_dir, myo_sim / "leg" / "assets" / "myolegs_tendon.xml")
    leg_muscle = _rel_uri(out_dir, myo_sim / "leg" / "assets" / "myolegs_muscle.xml")
    leg_chain = _rel_uri(out_dir, myo_sim / "leg" / "assets" / "myolegs_chain.xml")
    upper_chain_from_scene = _rel_uri(out_dir, upper_chain)
    return "\n".join(
        [
            f'<mujoco model="{model_name}">',
            f'  <include file="{scene}"/>',
            f'  <include file="{upper_assets}"/>',
            f'  <include file="{leg_assets}"/>',
            f'  <include file="{leg_tendon}"/>',
            f'  <include file="{leg_muscle}"/>',
            f'  <compiler angle="radian" meshdir="{meshdir}" texturedir="{meshdir}"/>',
            "  <worldbody>",
            '    <body name="root" pos="0 0 1" euler="0 0 -1.57">',
            f'      <include file="{upper_chain_from_scene}"/>',
            f'      <include file="{leg_chain}"/>',
            "      <freejoint/>",
            "    </body>",
            _worldbody_addons(club, contact),
            "  </worldbody>",
            _equality_fragment(),
            "</mujoco>",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_basename_lower(path: Path) -> str:
    """Return the lowercased final path component (Law of Demeter helper)."""
    return path.name.lower()


def _golfer_scene_receipt_payload(
    *,
    params: ContactParameters,
    driver_path: Path,
    iron_path: Path,
    driver_club: ClubSpec,
    iron_club: ClubSpec,
) -> dict[str, Any]:
    """Build the MS-51 golfer scene receipt document (pure data)."""
    return {
        "schema_version": "myosuite-golfer-scene/1",
        "issue": 10344,
        "ms": "MS-51",
        "myo_sim_pin": MYO_SIM_PIN_SHA,
        "myo_sim_path": MYO_SIM_RELATIVE.as_posix(),
        "base_model": "body/myobody_simpleupper.xml",
        "contact_law": "hunt_crossley_coulomb_shared",
        "contact_parameters": params.as_document(),
        "foot_contacts": [
            {
                "name": name,
                "body": body,
                "position_m": list(pos),
                "radius_m": radius,
            }
            for name, body, pos, radius in _FOOT_CONTACTS
        ],
        "grip": {
            "closure": "dual_site_weld",
            "sites": [
                "grip_site_hand_r",
                "grip_site_hand_l",
                "grip_site_club_r",
                "grip_site_club_l",
            ],
            "welds": ["grip_weld_r", "grip_weld_l"],
        },
        "clubs": {
            "driver": {
                "name": driver_club.name,
                "length_m": driver_club.length_m,
                "total_mass_kg": driver_club.total_mass_kg,
                "xml": driver_path.name,
                "sha256": _sha256(driver_path),
            },
            "iron": {
                "name": iron_club.name,
                "length_m": iron_club.length_m,
                "total_mass_kg": iron_club.total_mass_kg,
                "xml": iron_path.name,
                "sha256": _sha256(iron_path),
            },
        },
        "qualification": {
            "native_load_required": True,
            "parity_budget_qualified": False,
            "note": (
                "Generated topology with dual-grip welds and four foot contacts. "
                "A partial coordinate map is diagnostic only; 15 mm marker parity "
                "and G1 dynamics are not claimed by MS-51."
            ),
        },
    }


@precondition(lambda: True, "generate_golfer_scene entry")
@postcondition(
    lambda result: result.driver.is_file() and result.iron.is_file(),
    "driver and iron scenes must exist after generation",
)
def generate_golfer_scene(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    contact: ContactParameters | None = None,
) -> GolferScenePaths:
    """Generate driver/iron golfer MJCF scenes and a receipt JSON."""
    root = _repo_root(repo_root)
    myo_sim = _require_myo_sim(root)
    if output_root is not None:
        out_dir = Path(output_root)
    else:
        out_dir = root / "shared" / "models" / "myosuite" / "golf" / "body"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    right_arm, left_arm = _write_arm_chains_with_grip_sites(myo_sim, assets_dir)
    upper_chain = _write_upper_chain_with_local_arms(
        myo_sim, assets_dir, right_arm, left_arm
    )
    params = contact or _DEFAULT_CONTACT
    driver_path = out_dir / "golfer_myobody_driver.xml"
    iron_path = out_dir / "golfer_myobody_iron.xml"
    driver_path.write_text(
        _build_scene_xml(
            myo_sim=myo_sim,
            out_dir=out_dir,
            upper_chain=upper_chain,
            club=_club_spec(ClubKind.DRIVER),
            model_name="golfer_myobody_driver",
            contact=params,
        ),
        encoding="utf-8",
    )
    iron_path.write_text(
        _build_scene_xml(
            myo_sim=myo_sim,
            out_dir=out_dir,
            upper_chain=upper_chain,
            club=_club_spec(ClubKind.IRON),
            model_name="golfer_myobody_iron",
            contact=params,
        ),
        encoding="utf-8",
    )
    receipt_path = out_dir / "golfer_myobody_receipt.json"
    driver_club = _club_spec(ClubKind.DRIVER)
    iron_club = _club_spec(ClubKind.IRON)
    receipt = _golfer_scene_receipt_payload(
        params=params,
        driver_path=driver_path,
        iron_path=iron_path,
        driver_club=driver_club,
        iron_club=iron_club,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Generated MyoSuite golfer scenes at %s (pin %s)", out_dir, MYO_SIM_PIN_SHA
    )
    return GolferScenePaths(driver=driver_path, iron=iron_path, receipt=receipt_path)


def _marker_sites_from_map() -> dict[str, dict[str, Any]]:
    if not COORDINATE_MAP_PATH.is_file():
        return {}
    doc = json.loads(COORDINATE_MAP_PATH.read_text(encoding="utf-8"))
    markers = doc.get("marker_sites") or {}
    return dict(markers) if isinstance(markers, dict) else {}


@precondition(lambda: True, "resolve_golfer_scene entry")
def resolve_golfer_scene(
    marker_sites: dict[str, dict[str, Any]] | None = None,
    *,
    club: ClubKind = ClubKind.DRIVER,
    models_dir: Path | None = None,
    repo_root: Path | None = None,
) -> GolferScene:
    """Return the best available MyoSuite golfer MJCF and marker site table.

    Prefers the MS-51 generated driver/iron scene. Falls back to the vendored
    placeholder only when generation assets are absent (fail-closed note).
    """
    root = _repo_root(repo_root)
    models = Path(models_dir) if models_dir is not None else _DEFAULT_MODELS_DIR
    preferred = models / (
        "golfer_myobody_driver.xml"
        if club is ClubKind.DRIVER
        else "golfer_myobody_iron.xml"
    )
    sites = marker_sites or _marker_sites_from_map()
    if preferred.is_file():
        return GolferScene(
            xml_path=preferred,
            marker_sites=dict(sites),
            topology_note=(
                f"MS-51 generated {preferred.name} on "
                f"myo_sim@{MYO_SIM_PIN_SHA[:12]} "
                "(myobody_simpleupper + club welds + four foot contacts). "
                "Parity budget not claimed."
            ),
            parity_budget_qualified=False,
            club=club,
            myo_sim_pin=MYO_SIM_PIN_SHA,
        )

    try:
        paths = generate_golfer_scene(repo_root=root, output_root=models)
        xml = paths.driver if club is ClubKind.DRIVER else paths.iron
        return GolferScene(
            xml_path=xml,
            marker_sites=dict(sites),
            topology_note=(
                f"MS-51 on-demand generated {xml.name} on "
                f"myo_sim@{MYO_SIM_PIN_SHA[:12]}. Parity budget not claimed."
            ),
            parity_budget_qualified=False,
            club=club,
            myo_sim_pin=MYO_SIM_PIN_SHA,
        )
    except FileNotFoundError:
        logger.warning("myo_sim unavailable; falling back to placeholder MyoBody")

    if not _PLACEHOLDER_MYOBODY.is_file():
        raise FileNotFoundError(
            f"MyoSuite body MJCF missing: {_PLACEHOLDER_MYOBODY}. "
            "Run scripts/setup_myosuite_models.ps1 (or .sh)."
        )
    return GolferScene(
        xml_path=_PLACEHOLDER_MYOBODY,
        marker_sites=dict(sites)
        or {
            "WaistLeft": {"body": "pelvis", "pos": [-0.12, 0.0, 0.0]},
            "WaistRight": {"body": "pelvis", "pos": [0.12, 0.0, 0.0]},
            "BackTop": {"body": "torso", "pos": [0.0, 0.0, 0.35]},
        },
        topology_note=(
            "Vendored MyoBody placeholder MJCF — MS-51 myo_sim pin / generated "
            "golfer scene not available in this checkout."
        ),
        parity_budget_qualified=False,
        club=club,
    )


@dataclass(frozen=True)
class BimanualGraspSynthesis:
    """Bimanual grasp synthesis result ``G y = w_club`` (HRG-GS-M7, #13329)."""

    grasp_matrix: np.ndarray
    hand_wrench_vector: np.ndarray
    club_wrench_vector: np.ndarray
    residual_inf_n: float


_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _require_mujoco_sdk() -> Any:
    if mujoco is None:  # pragma: no cover
        raise ImportError("mujoco SDK is required for MyoSuiteGolferGripAdapter")
    return mujoco


def _raise_myosuite_error(cls_err: Any, model_id: str, feature: str) -> None:
    raise cls_err(
        model_id,
        consumer=_MYOSUITE_CONSUMER,
        feature=feature,
        child=_MYOSUITE_CHILD,
        issue=_MYOSUITE_ISSUE,
    )


def _canonical_myosuite_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    resolved = _MYOSUITE_MODEL_ID_ALIASES.get(model_id, model_id)
    if resolved not in ADMITTED_MYOSUITE_GOLFER_MODELS:
        raise ValueError(f"Unknown MyoSuite golfer model_id {model_id!r}")
    return resolved


def _load_myosuite_manifest(
    model_id: str, manifest_override: Mapping[str, Any] | Path | None = None
) -> dict[str, Any]:
    if manifest_override is None:
        path = MYOSUITE_MANIFEST_PATHS[_canonical_myosuite_model_id(model_id)]
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(manifest_override, Path):
        raw = json.loads(manifest_override.read_text(encoding="utf-8"))
    elif isinstance(manifest_override, Mapping):
        raw = dict(manifest_override)
    else:
        raise TypeError(
            f"manifest must be Mapping, Path, or None, got {type(manifest_override).__name__}"
        )
    validate_manifest(raw)
    return dict(raw)


def _validate_bound_bodies_exist(
    model: Any, model_id: str, bodies: tuple[str, ...]
) -> None:
    mj = _require_mujoco_sdk()
    body_obj = mj.mjtObj.mjOBJ_BODY
    for body_id in bodies:
        if int(mj.mj_name2id(model, body_obj, body_id)) < 0:
            _raise_myosuite_error(
                SchemaFeatureUnavailableError,
                model_id,
                f"missing required model body {body_id!r}",
            )


def _resolve_grip_weld_ids(model: Any) -> dict[str, int]:
    mj = _require_mujoco_sdk()
    eq_obj = mj.mjtObj.mjOBJ_EQUALITY
    return {
        s: int(mj.mj_name2id(model, eq_obj, n))
        for s, n in DEFAULT_GRIP_WELDS.items()
        if int(mj.mj_name2id(model, eq_obj, n)) >= 0
    }


def _configure_grip_welds(
    model: Any, data: Any, weld_ids: Mapping[str, int], *, active: bool
) -> None:
    flag = 1 if active else 0
    eq_active0, eq_active = model.eq_active0, data.eq_active
    for eq_id in weld_ids.values():
        eq_active0[eq_id] = flag
        eq_active[eq_id] = flag


def _hand_wrench_namespace(side: str, hw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        hand=side,
        force_n=np.asarray(hw.force_on_club_n, dtype=float).copy(),
        torque_nm=np.asarray(hw.torque_on_club_nm, dtype=float).copy(),
        point_m=np.asarray(hw.point_m, dtype=float).copy(),
    )


class MyoSuiteGolferGripAdapter:
    """MyoSuite golfer :class:`ModelGripAdapter` binding ``hand_l``, ``hand_r``, and ``golf_club`` (HRG-GS-M7, #13329)."""

    def __init__(
        self,
        model: Any,
        data: Any | None = None,
        *,
        manifest: Mapping[str, Any] | Path | None = None,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: bool = False,
    ) -> None:
        mj = _require_mujoco_sdk()
        spec = grip_spec if grip_spec is not None else GripSpec()
        if not isinstance(spec, GripSpec):
            raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")
        if not isinstance(contact_mode, bool):
            raise TypeError("contact_mode must be a bool")

        resolved_manifest = _load_myosuite_manifest("myosuite_golfer_driver", manifest)
        self._manifest = resolved_manifest
        self.model_id = str(resolved_manifest["model_id"])
        self.club_body = str(resolved_manifest["club_body"])
        self.lead_hand_body = str(resolved_manifest["lead_hand_body"])
        self.trail_hand_body = str(resolved_manifest["trail_hand_body"])
        self.model = model
        self.data = data if data is not None else mj.MjData(model)
        _validate_bound_bodies_exist(
            self.model,
            self.model_id,
            (self.club_body, self.lead_hand_body, self.trail_hand_body),
        )
        self._grip_weld_ids = _resolve_grip_weld_ids(self.model)
        self.contact_mode = bool(contact_mode)
        _configure_grip_welds(
            self.model, self.data, self._grip_weld_ids, active=not self.contact_mode
        )
        mj.mj_forward(self.model, self.data)

        self.grip_spec = spec
        self.visual_pose = resolve_visual_pose(spec)
        self.topology = contact_topology(spec)
        self._interface = interface
        self._prescribed_states: BodyStates | None = None
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cal_grip_frames = _calibration_grip_frames(
            self.model, self.club_body, self.lead_hand_body, self.trail_hand_body
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "myosuite_golfer_driver",
        *,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: bool = False,
        models_dir: Path | None = None,
        repo_root: Path | None = None,
    ) -> MyoSuiteGolferGripAdapter:
        """Load an admitted MyoSuite golfer scene (driver or iron) into a :class:`MyoSuiteGolferGripAdapter`."""
        mj = _require_mujoco_sdk()
        canonical_id = _canonical_myosuite_model_id(model_id)
        manifest = _load_myosuite_manifest(canonical_id)
        club_kind = (
            ClubKind.DRIVER if canonical_id.endswith("_driver") else ClubKind.IRON
        )
        scene = resolve_golfer_scene(
            club=club_kind, models_dir=models_dir, repo_root=repo_root
        )
        if scene.is_placeholder:
            raise FileNotFoundError(
                f"MyoSuite golfer scene {canonical_id!r} requires pinned myo_sim assets"
            )
        cache_key = (str(scene.xml_path), bool(contact_mode))
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = mj.MjModel.from_xml_path(str(scene.xml_path))
        mj_model = _MODEL_CACHE[cache_key]
        return cls(
            mj_model,
            mj.MjData(mj_model),
            manifest=manifest,
            grip_spec=grip_spec,
            interface=interface,
            contact_mode=contact_mode,
        )

    @property
    def grip_weld_ids(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._grip_weld_ids))

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    def active_grip_weld_names(self) -> tuple[str, ...]:
        """Return the names of currently active grip weld equalities in ``data.eq_active``."""
        eq_active = self.data.eq_active
        return tuple(
            DEFAULT_GRIP_WELDS[s]
            for s in ("L", "R")
            if s in self._grip_weld_ids and int(eq_active[self._grip_weld_ids[s]]) != 0
        )

    def _club_grip_frames(self) -> dict[str, np.ndarray]:
        iface = self._interface
        if iface is not None:
            left_f, right_f = iface.left, iface.right
            return {"L": left_f.matrix(), "R": right_f.matrix()}
        return dict(self._cal_grip_frames)

    def capabilities(self) -> GripCapabilities:
        """Return declared :class:`GripCapabilities` for site-weld or contact mode."""
        avail = (self.club_body, self.lead_hand_body, self.trail_hand_body)
        if self.contact_mode:
            return GripCapabilities(
                attachment_modes=("contact", "contact_v1", "bushing"),
                supports_physical_pairs=True,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=avail,
            )
        welded_reason = (
            f"club body {self.club_body!r} is site-welded to hands "
            f"({self.lead_hand_body!r}, {self.trail_hand_body!r}) via "
            f"grip_weld_l and grip_weld_r in MJCF {self.model_id!r}; "
            "pass contact_mode=True to disable grip welds and enable physical pad contact"
        )
        return GripCapabilities(
            attachment_modes=("weld",),
            supports_physical_pairs=False,
            supports_wrench_application=False,
            native_contact=False,
            available_bodies=avail,
            unavailable_reason=welded_reason,
        )

    def resolve_binding(
        self, topology: ContactTopology, *, attachment_mode: str | None = None
    ) -> HandClubBinding:
        """Resolve and validate a :class:`HandClubBinding` against :meth:`capabilities`."""
        manifest = dict(self._manifest)
        if attachment_mode is not None:
            manifest["attachment_mode"] = attachment_mode
        elif self.contact_mode:
            manifest["attachment_mode"] = "contact"
        grip_frames = self._club_grip_frames()
        binding = binding_from_manifest(manifest, topology, grip_frames)
        validate_resolved_binding(binding, topology, grip_frames)
        validate_binding(topology, binding, self.capabilities())
        return binding

    def set_prescribed_grip_states(
        self,
        *,
        hand_states: Mapping[str, BushingState],
        club_state: RigidBodyState | BushingState,
    ) -> None:
        """Set prescribed hand and club grip states for M3 harness or hold evaluation."""
        for side_key in ("L", "R"):
            if side_key not in hand_states:
                raise ValueError(f"hand_states missing side {side_key!r}")
        c_pose, c_vel = _state_to_pose_and_vel(club_state)
        l_pose, l_vel = _state_to_pose_and_vel(hand_states["L"])
        r_pose, r_vel = _state_to_pose_and_vel(hand_states["R"])
        poses = {
            self.club_body: c_pose,
            self.lead_hand_body: l_pose,
            self.trail_hand_body: r_pose,
        }
        vels = {
            self.club_body: c_vel,
            self.lead_hand_body: l_vel,
            self.trail_hand_body: r_vel,
        }
        self._prescribed_states = BodyStates(poses, velocities=vels)

    def clear_prescribed_grip_states(self) -> None:
        self._prescribed_states = None

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for bound bodies."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        if self._prescribed_states is not None:
            return self._prescribed_states
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id in binding.required_bodies():
            poses[body_id], velocities[body_id] = _sample_body_pose_and_vel(
                self.model, self.data, body_id
            )
        return BodyStates(poses, velocities=velocities)

    def extract_weld_closure(self) -> Any:
        """Extract per-hand weld constraint wrenches from ``data.efc_force`` via :func:`grip_analysis_from_efc`."""
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        grip = grip_analysis_from_efc(self.model, self.data)
        left_hw, right_hw = grip.left, grip.right
        if left_hw is None or right_hw is None:
            reason = grip.unavailable_reason or "grip weld efc_force unavailable"
            return SimpleNamespace(available=False, reason=reason, wrenches=[])
        w_left = _hand_wrench_namespace("lead", left_hw)
        w_right = _hand_wrench_namespace("trail", right_hw)
        return SimpleNamespace(
            available=True, reason="", wrenches=[w_left, w_right], grip_analysis=grip
        )

    def synthesise_closure_from_net_wrench(
        self, force_n: np.ndarray | None = None, torque_nm: np.ndarray | None = None
    ) -> Any:
        """Build a closure stub for allocated hand-hand reporting in contact mode."""
        f_vec = np.asarray(
            force_n if force_n is not None else [250.0, -120.0, 80.0], dtype=float
        )
        t_vec = np.asarray(
            torque_nm if torque_nm is not None else [6.0, -4.0, 2.0], dtype=float
        )
        w_item = SimpleNamespace(
            force_n=f_vec.copy(),
            torque_nm=t_vec.copy(),
            point_m=np.zeros(3, dtype=float),
        )
        return SimpleNamespace(available=True, reason="", wrenches=[w_item])

    def allocate_weld_pad_forces(
        self,
        topology: ContactTopology | None = None,
        *,
        prior: AllocationPrior | None = None,
    ) -> dict[str, AllocatedPairForce]:
        """Allocate per-hand site-weld constraint wrenches across pad-grip pairs."""
        topo = topology if topology is not None else self.topology
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        grip = grip_analysis_from_efc(self.model, self.data)
        left_hw, right_hw = grip.left, grip.right
        if left_hw is None or right_hw is None:
            reason = grip.unavailable_reason or "grip weld efc_force unavailable"
            return {
                p.name: AllocatedPairForce(
                    available=False,
                    reason=reason,
                    pair_name=p.name,
                    label=p.allocated_label,
                )
                for p in topo.hand_grip_pairs()
            }
        mu = float(_DEFAULT_CONTACT.static_friction)
        results: dict[str, AllocatedPairForce] = {}
        for role, side_char, hw in (("lead", "L", left_hw), ("trail", "R", right_hw)):
            side_pairs = [
                p for p in topo.hand_grip_pairs() if topo.site(p.site_a).hand == role
            ]
            n_side = max(1, len(side_pairs))
            f_vec = np.asarray(hw.force_on_club_n, dtype=float)
            t_vec = np.asarray(hw.torque_on_club_nm or (0.0, 0.0, 0.0), dtype=float)
            ref_pt, c_mag = np.asarray(hw.point_m, dtype=float), float(npla.norm(f_vec))
            for pair in side_pairs:
                mapped = map_site_to_hand_frame(topo, topo.site(pair.site_a), side_char)
                centre = np.asarray(mapped.centre_m, dtype=float)
                rad = np.array([centre[0], 0.0, centre[2]], dtype=float)
                rad_norm = float(npla.norm(rad))
                normal = (
                    rad / rad_norm
                    if rad_norm > 1e-12
                    else np.array([0.0, 0.0, 1.0], dtype=float)
                )
                weight = (
                    float(prior.weights[pair.name])
                    if prior is not None and pair.name in prior.weights
                    else 1.0 / n_side
                )
                results[pair.name] = allocate_pair_force(
                    force_on_lead_n=weight * f_vec,
                    torque_on_lead_nm=weight * t_vec,
                    reference_point_m=ref_pt,
                    patch=(ref_pt + centre, normal),
                    static_friction=mu,
                    pair_name=pair.name,
                    label=pair.allocated_label,
                    closure_force_n=c_mag,
                )
        return results

    def evaluate_contacts(
        self,
        topology_model: TopologyContactModel,
        binding: HandClubBinding | None = None,
        *,
        closure: Any | None = None,
        prior: AllocationPrior | None = None,
    ) -> GripStepReport:
        """Evaluate grip contacts at current sampled states, extracting weld ``efc_force`` in weld mode."""
        rb = (
            binding
            if binding is not None
            else self.resolve_binding(topology_model.topology)
        )
        rc = (
            self.extract_weld_closure()
            if closure is None and not self.contact_mode
            else closure
        )
        return evaluate_contacts(
            topology_model, rb, self.sample_states(rb), closure=rc, prior=prior
        )

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply physical body wrenches to ``data.xfrc_applied`` in contact mode."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        if not self.contact_mode:
            _raise_myosuite_error(
                WeldedGripBodyError,
                self.model_id,
                f"cannot apply physical wrenches to welded club/hand bodies in {self.model_id!r}",
            )
        active_welds = self.active_grip_weld_names()
        if active_welds:
            _raise_myosuite_error(
                WeldedGripBodyError,
                self.model_id,
                f"cannot apply shared-law wrenches while grip weld equalities {active_welds} are active",
            )
        mj = _require_mujoco_sdk()
        body_obj = mj.mjtObj.mjOBJ_BODY
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        xfrc, xpos, xipos = self.data.xfrc_applied, self.data.xpos, self.data.xipos
        for body_id, pair in wrenches.items():
            f_arr, m_arr = (
                np.asarray(pair[0], dtype=float),
                np.asarray(pair[1], dtype=float),
            )
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            b_idx = int(mj.mj_name2id(self.model, body_obj, body_id))
            if b_idx < 0:
                raise ValueError(f"Body {body_id!r} not found in MjModel")
            arm = np.asarray(xpos[b_idx], dtype=float) - np.asarray(
                xipos[b_idx], dtype=float
            )
            xfrc[b_idx, :3] = f_arr
            xfrc[b_idx, 3:] = m_arr + np.cross(arm, f_arr)
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())
        self._applied_wrenches = cleaned

    @postcondition(
        lambda result: result.residual_inf_n <= 1e-9,
        "bimanual grasp synthesis residual ||G y - w_club||_inf <= 1e-9 N",
    )
    def synthesize_bimanual_grasp(
        self,
        report: GripStepReport | None = None,
        *,
        binding: HandClubBinding | None = None,
    ) -> BimanualGraspSynthesis:
        """Synthesize bimanual grasp map ``G y = w_club`` and verify residual ``<= 1e-9 N``."""
        if not self.contact_mode:
            mj = _require_mujoco_sdk()
            mj.mj_forward(self.model, self.data)
            grip = grip_analysis_from_efc(self.model, self.data)
            left_hw, right_hw = grip.left, grip.right
            if (
                left_hw is None
                or right_hw is None
                or grip.net_force_n is None
                or grip.couple_at_midpoint_nm is None
            ):
                raise ValueError(
                    grip.unavailable_reason
                    or "grip weld efc_force unavailable for synthesis"
                )
            r_l, r_r = (
                np.asarray(left_hw.point_m, dtype=float),
                np.asarray(right_hw.point_m, dtype=float),
            )
            origin = 0.5 * (r_l + r_r)
            g_mat = grasp_map(
                NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=origin),
                r_l,
                r_r,
            )
            y_vec = np.concatenate(
                [
                    left_hw.force_on_club_n,
                    left_hw.torque_on_club_nm or (0.0, 0.0, 0.0),
                    right_hw.force_on_club_n,
                    right_hw.torque_on_club_nm or (0.0, 0.0, 0.0),
                ]
            )
            w_club = np.concatenate([grip.net_force_n, grip.couple_at_midpoint_nm])
            return BimanualGraspSynthesis(
                g_mat, y_vec, w_club, float(np.max(np.abs(g_mat @ y_vec - w_club)))
            )

        if report is None:
            raise ValueError(
                "report is required for bimanual grasp synthesis in contact mode"
            )
        rb = (
            binding
            if binding is not None
            else self.resolve_binding(self.topology, attachment_mode="contact")
        )
        states = self.sample_states(rb)
        r_l = states[self.lead_hand_body][:3, 3].copy()
        r_r = states[self.trail_hand_body][:3, 3].copy()
        r_c = states[self.club_body][:3, 3].copy()
        origin = 0.5 * (r_l + r_r)
        g_mat = grasp_map(
            NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=origin),
            r_l,
            r_r,
        )
        bw = report.body_wrenches()
        f_hl, m_hl = bw.get(self.lead_hand_body, (np.zeros(3), np.zeros(3)))
        f_hr, m_hr = bw.get(self.trail_hand_body, (np.zeros(3), np.zeros(3)))
        f_c, m_c = bw.get(self.club_body, (np.zeros(3), np.zeros(3)))
        y_vec = np.concatenate([-f_hl, -m_hl, -f_hr, -m_hr])
        w_club = np.concatenate([f_c, m_c + np.cross(r_c - origin, f_c)])
        return BimanualGraspSynthesis(
            g_mat, y_vec, w_club, float(np.max(np.abs(g_mat @ y_vec - w_club)))
        )


__all__ = [
    "ADMITTED_MYOSUITE_GOLFER_MODELS",
    "COORDINATE_MAP_PATH",
    "MYOSUITE_MANIFEST_PATHS",
    "MYO_SIM_PIN_SHA",
    "MYO_SIM_RELATIVE",
    "BimanualGraspSynthesis",
    "ClubKind",
    "GolferScene",
    "GolferScenePaths",
    "MyoSuiteGolferGripAdapter",
    "WeldedGripBodyError",
    "generate_golfer_scene",
    "resolve_golfer_scene",
]
