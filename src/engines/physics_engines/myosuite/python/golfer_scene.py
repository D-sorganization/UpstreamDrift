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

import enum
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.motion_matching.contact_law import ContactParameters

logger = logging.getLogger(__name__)

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[5]
_ENGINE_ROOT = Path(__file__).resolve().parents[1]
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


def _worldbody_addons(club: ClubSpec, contact: ContactParameters) -> str:
    """Club + contact sphere bodies placed inside ``<worldbody>``."""
    shaft_half = 0.5 * club.length_m
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
    club_y = -shaft_half
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
            '      <site name="grip_site_club_r" pos="0 -0.95 0" size="0.006"/>',
            '      <site name="grip_site_club_l" pos="0 -0.88 0" size="0.006"/>',
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


__all__ = [
    "COORDINATE_MAP_PATH",
    "MYO_SIM_PIN_SHA",
    "MYO_SIM_RELATIVE",
    "ClubKind",
    "GolferScene",
    "GolferScenePaths",
    "generate_golfer_scene",
    "resolve_golfer_scene",
]
