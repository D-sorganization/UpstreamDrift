"""Flexible golf shaft as a discretised rigid-link chain (R6, #12686).

The club of the full-body specification is one rigid body by default
(``club.shaft.model = "rigid"``). :func:`apply_shaft_chain` replaces its
exposed shaft (grip end to head) by ``N`` rigid links joined by spring-damper
joints: two bending axes per joint and optional torsion. This module is
engine-neutral: it writes only specification bodies, joints and per-primitive
spring constants, so the MuJoCo export (``native_mjcf``) and the later
Drake, Pinocchio, OpenSim and Simscape ports (R11) read the same numbers.

Profile (owner decision, #12686): a generic STIFF flex, flagged
``default_not_measured``. The section is a tapered tube whose wall thickens
toward the tip so the linear density stays the club's (shaft mass over the
exposed span, which keeps the chain's mass identical to the rigid club). The
modulus is calibrated so the clamped shaft plus head vibrates at the
``SHAFT_FLEX_DATA`` frequency of the flex (STIFF: 280 cpm) in the standard
5 in butt clamp. The stations are the vendored Tools ``ShaftStation`` records.

Lumping rule (midpoint): the exposed span ``L`` is cut into ``N`` equal
segments ``ds = L / N``; joint ``i`` sits at the segment midpoint
``s_i = (i + 1/2) ds`` measured from the grip end, with bending stiffness
``k_i = EI(s_i) / ds`` (and ``GJ(s_i) / ds`` in torsion). The beam compliance
``int (L - s)^2 / EI ds`` is then integrated by the midpoint rule (second
order in ``ds``). Link ``i`` is the shaft piece between joints ``i`` and
``i + 1`` (the last link ends at the tip and carries the head); the piece
from the grip end to joint 0 stays on the club body. The rigid solid rod is
partitioned exactly, so mass, centre of mass and inertia equal the rigid
club's. Joint damping is stiffness proportional, ``c_i = beta k_i`` with
``beta = 2 zeta / omega_1``, ``omega_1`` the FEM first bending frequency of
the grip-clamped span with the head.

Frames: every link body uses the club body frame (shaft along ``-y`` from the
head to the grip, origin at the head); at zero deflection all link frames
coincide with the club frame, so solid and frame placements keep their rigid
values. Joint ``i`` has its base and follower at the joint point; the
primitives are ``Rx`` and ``Rz`` (bending about the club ``x`` and ``z``
axes) and ``Ry`` (torsion about the shaft). The ``Clubhead`` frame moves to
the tip link, so head markers and the face centre follow the deflected head.

Sources: Milne & Davis (1992), J. Biomech. 25(9):975-983 and MacKenzie &
Sprigings (2009), Proc. IMechE P 223:13-19 (role of shaft flexibility; no
numbers taken from them); MuJoCo XML reference, joint ``stiffness``,
``springref`` and ``damping``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.motion_matching.club_models import (
    CLUB_BODY_SUFFIX,
    HEAD_HEIGHT_M,
    ClubSpec,
    _placement,
    _rod_inertia,
    club_solids,
)

SHAFT_MODELS: tuple[str, ...] = ("rigid", "chain")
DEFAULT_SHAFT_MODEL = "rigid"
#: Engines whose spec adapters apply the chain's joint springs (#12686).
CHAIN_ENGINES: frozenset[str] = frozenset({"mujoco"})
PROFILE_STATUS = "default_not_measured"
DEFAULT_FLEX = "Stiff"
#: Links of the default chain: the smallest N whose static tip deflection and
#: first frequency both change by < 1 % from N to 2N (convergence study,
#: #12686). N = 8 would also pass but its stiff damped modes leave the
#: classical RK4 stability region at the 0.125 ms same-input substep.
DEFAULT_LINKS = 4
#: Standard frequency-analyser butt clamp (5 in) used for the CPM calibration.
CPM_CLAMP_M = 0.127
BUTT_DIAMETER_M = 0.0152  # 0.600 in, generic
TIP_DIAMETER_M = {"driver": 0.0084, "iron7": 0.0094}  # 0.335 / 0.370 in
DENSITY_KG_M3 = {"driver": 1750.0, "iron7": 7850.0}  # graphite / steel
POISSON_RATIO = 0.3  # generic isotropic estimate for G = E / 2(1 + nu)
DAMPING_RATIO = 0.02  # ShaftProperties default structural damping
PROFILE_STATIONS = 41
FEM_ELEMENTS = 64
COORDINATE_PREFIX = "shaft"
SOURCES: tuple[str, ...] = (
    "Milne R.D., Davis J.P. (1992) The role of the shaft in the golf swing. "
    "J. Biomech. 25(9):975-983",
    "MacKenzie S.J., Sprigings E.J. (2009) Understanding the role of shaft "
    "stiffness in the golf swing. Proc. IMechE Part P 223:13-19",
    "MuJoCo XML reference: joint stiffness, springref, damping",
)


class ShaftChainUnsupportedError(ValueError):
    """A chain-shaft spec was handed to an engine without chain support."""


@dataclass(frozen=True)
class ShaftChain:
    """Engine-neutral chain parameters (SI units, ``s`` from the grip end).

    ``joint_s_m`` are the joint points, ``link_bounds_m`` each link's shaft
    piece, ``bend_stiffness`` / ``twist_stiffness`` the spring constants
    (N m/rad, ``twist_stiffness`` empty without torsion) and ``damping_beta_s``
    the stiffness-proportional damping factor.
    """

    club: ClubSpec
    n_links: int
    joint_s_m: tuple[float, ...]
    link_bounds_m: tuple[tuple[float, float], ...]
    bend_stiffness: tuple[float, ...]
    twist_stiffness: tuple[float, ...]
    damping_beta_s: float
    stiffness_scale: float
    record: dict[str, Any]

    def __post_init__(self) -> None:
        n = self.n_links
        if n < 1 or len(self.joint_s_m) != n or len(self.link_bounds_m) != n:
            raise ValueError("a chain needs one joint and one link per segment")
        if len(self.bend_stiffness) != n or len(self.twist_stiffness) not in (0, n):
            raise ValueError("one bending (and optional torsion) spring per joint")
        springs = (*self.bend_stiffness, *self.twist_stiffness)
        if not all(math.isfinite(k) and k > 0 for k in springs):
            raise ValueError("spring constants must be finite and positive")
        if not (math.isfinite(self.damping_beta_s) and self.damping_beta_s >= 0):
            raise ValueError("damping factor must be finite and >= 0")

    @property
    def span_m(self) -> float:
        return self.club.length_m - self.club.grip_length_m

    @property
    def torsion(self) -> bool:
        return bool(self.twist_stiffness)

    def coordinates(self) -> tuple[str, ...]:
        """Chain coordinate names in spec order (appended after the body's)."""
        return tuple(
            c for i in range(self.n_links) for c in _joint_coordinates(self, i)
        )


def _joint_coordinates(chain: ShaftChain, i: int) -> tuple[str, ...]:
    names = (f"{COORDINATE_PREFIX}_bend_x_{i}", f"{COORDINATE_PREFIX}_bend_z_{i}")
    return (*names, f"{COORDINATE_PREFIX}_twist_{i}") if chain.torsion else names


# --- generic flex profile ---------------------------------------------------


def flex_frequency_hz(flex: str = DEFAULT_FLEX) -> float:
    """Clamped club frequency of ``flex`` from ``SHAFT_FLEX_DATA`` (cpm / 60)."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.club_configurations import (  # noqa: E501
        SHAFT_FLEX_DATA,
    )

    if flex not in SHAFT_FLEX_DATA:
        raise ValueError(f"unknown flex {flex!r}; known {sorted(SHAFT_FLEX_DATA)}")
    return float(SHAFT_FLEX_DATA[flex]["stiffness_cpm"]) / 60.0


def tube_properties(club: ClubSpec, start_m: float, youngs_modulus: float) -> Any:
    """Constant-area tapered tube from ``start_m`` (from the butt) to the tip.

    Outer diameter tapers linearly from the butt to the tip over the full club
    length; the wall makes the section area ``A = m_shaft / (L_span rho)`` at
    every station, so the linear density is the club's. Returns the physics
    package ``ShaftProperties`` (station 0 at ``start_m``, the clamp).
    """
    from src.shared.python.physics.flexible_shaft import ShaftProperties

    if club.name not in TIP_DIAMETER_M:
        raise ValueError(f"no generic shaft geometry for club {club.name!r}")
    rho = DENSITY_KG_M3[club.name]
    area = club.shaft_mass_kg / ((club.length_m - club.grip_length_m) * rho)
    s = np.linspace(start_m, club.length_m, PROFILE_STATIONS)
    diameter = BUTT_DIAMETER_M + (TIP_DIAMETER_M[club.name] - BUTT_DIAMETER_M) * (
        s / club.length_m
    )
    wall = (diameter - np.sqrt(diameter**2 - 4.0 * area / math.pi)) / 2.0
    return ShaftProperties(
        length=club.length_m - start_m,
        outer_diameter=diameter,
        wall_thickness=wall,
        station_positions=s - start_m,
        youngs_modulus=float(youngs_modulus),
        density=rho,
        damping_ratio=DAMPING_RATIO,
    )


def clamped_head_frequencies(
    properties: Any,
    club: ClubSpec,
    head_inertia_kg_m2: float = 0.0,
    n_elements: int = FEM_ELEMENTS,
) -> np.ndarray:
    """FEM bending frequencies (Hz, lowest three) of the clamped shaft + head.

    Reuses ``FiniteElementShaftModel`` (Euler-Bernoulli, consistent mass,
    clamped at station 0) and adds the head on the tip node: its mass on the
    tip translation (its centre of mass lies on the shaft line at the tip)
    and ``head_inertia_kg_m2``, its inertia about the bending axis, on the tip
    slope.
    """
    from scipy.linalg import eigh

    from src.shared.python.physics.flexible_shaft import FiniteElementShaftModel

    fem = FiniteElementShaftModel(n_elements)
    fem.initialize(properties)
    mass = fem.M.copy()
    mass[-2, -2] += club.head_mass_kg
    mass[-1, -1] += head_inertia_kg_m2
    eigenvalues = eigh(fem.K, mass, eigvals_only=True)
    return np.sqrt(eigenvalues[:3]) / (2.0 * math.pi)


def fem_tip_deflection(
    properties: Any, force_n: float, n_elements: int = FEM_ELEMENTS
) -> float:
    """FEM static tip deflection (m) of the clamped shaft under a tip load."""
    from src.shared.python.physics.flexible_shaft import FiniteElementShaftModel

    fem = FiniteElementShaftModel(n_elements)
    fem.initialize(properties)
    state = fem.compute_static_solution(properties.length, force_n)
    return float(state.deflections[-1])


def calibrated_modulus_pa(club: ClubSpec, flex: str = DEFAULT_FLEX) -> float:
    """Modulus giving the flex's cpm for the 5 in butt clamp plus head.

    Frequencies scale with ``sqrt(E)`` (stiffness linear in ``E``, mass
    independent of it), so one FEM solve at a trial modulus fixes ``E``.
    """
    trial = 100e9
    f_trial = clamped_head_frequencies(tube_properties(club, CPM_CLAMP_M, trial), club)
    return trial * (flex_frequency_hz(flex) / float(f_trial[0])) ** 2


def span_properties(club: ClubSpec, flex: str = DEFAULT_FLEX) -> Any:
    """Calibrated profile of the exposed span (grip end clamped to the tip)."""
    return tube_properties(club, club.grip_length_m, calibrated_modulus_pa(club, flex))


def station_profile(properties: Any, club: ClubSpec) -> Any:
    """The span profile as vendored Tools ``ShaftStation`` records.

    ``EI`` about both transverse axes is ``E I`` (a round tube), ``GJ`` is
    ``G 2I`` with ``G = E / 2(1 + nu)``; the profile is flagged
    ``default_not_measured`` in its provenance.
    """
    # Canonical spelling only: ``src.shared.python.golf_club`` is a second
    # module tree whose import re-runs Tools' import-time variable registration.
    from shared.python.golf_club.shaft_profile import (
        ShaftProfile,
        ShaftProfileProvenance,
        ShaftStation,
    )
    from src.shared.python.physics.flexible_shaft import (
        compute_EI_profile,
        compute_mass_profile,
    )

    ei = compute_EI_profile(properties)
    shear = properties.youngs_modulus / (2.0 * (1.0 + POISSON_RATIO))
    gj = shear * 2.0 * ei / properties.youngs_modulus
    density = compute_mass_profile(properties)
    stations = tuple(
        ShaftStation(
            position_m=float(s),
            outer_diameter_m=float(d),
            inner_diameter_m=float(d - 2.0 * t),
            linear_density_kg_m=float(rho),
            ei_about_x_n_m2=float(e),
            ei_about_y_n_m2=float(e),
            gj_n_m2=float(g),
            damping_ratio=properties.damping_ratio,
        )
        for s, d, t, rho, e, g in zip(
            properties.station_positions,
            properties.outer_diameter,
            properties.wall_thickness,
            density,
            ei,
            gj,
            strict=True,
        )
    )
    return ShaftProfile(
        shaft_id=f"generic_{club.name}_shaft",
        frame_id="club_span_from_grip_end",
        raw_length_m=properties.length,
        cut_length_m=properties.length,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=stations,
        provenance=ShaftProfileProvenance(
            source_name=f"generic tapered tube ({PROFILE_STATUS})",
            measurement_method="modulus calibrated to SHAFT_FLEX_DATA cpm (#12686)",
            uncertainty_note="not a measured shaft; capture-club shafts unknown",
        ),
    )


# --- chain parameters --------------------------------------------------------


def build_shaft_chain(
    club: ClubSpec,
    n_links: int = DEFAULT_LINKS,
    *,
    flex: str = DEFAULT_FLEX,
    torsion: bool = False,
    stiffness_scale: float = 1.0,
) -> ShaftChain:
    """Lump the calibrated profile into ``n_links`` links (midpoint rule).

    ``stiffness_scale`` multiplies every spring (the rigid-limit check uses
    1e6); damping scales with its square root so modal damping ratios stay.
    """
    if isinstance(n_links, bool) or not isinstance(n_links, int) or n_links < 1:
        raise ValueError("n_links must be a positive integer")
    if not (math.isfinite(stiffness_scale) and stiffness_scale > 0):
        raise ValueError("stiffness_scale must be finite and positive")
    properties = span_properties(club, flex)
    profile = station_profile(properties, club)
    ds = properties.length / n_links
    joints = tuple((i + 0.5) * ds for i in range(n_links))
    stations = [profile.station_at(s) for s in joints]
    bend = tuple(stiffness_scale * st.ei_about_x_n_m2 / ds for st in stations)
    twist = (
        tuple(stiffness_scale * st.gj_n_m2 / ds for st in stations) if torsion else ()
    )
    omega = 2.0 * math.pi * float(clamped_head_frequencies(properties, club)[0])
    beta = 2.0 * DAMPING_RATIO / omega / math.sqrt(stiffness_scale)
    bounds = tuple(
        (s, joints[i + 1] if i + 1 < n_links else properties.length)
        for i, s in enumerate(joints)
    )
    record = _chain_record(club, n_links, properties, (flex, torsion, stiffness_scale))
    return ShaftChain(
        club, n_links, joints, bounds, bend, twist, beta, stiffness_scale, record
    )


def _chain_record(
    club: ClubSpec, n_links: int, properties: Any, options: tuple[str, bool, float]
) -> dict[str, Any]:
    flex, torsion, scale = options
    return {
        "model": "chain",
        "n_links": n_links,
        "torsion": torsion,
        "stiffness_scale": scale,
        "flex": flex,
        "flex_cpm": 60.0 * flex_frequency_hz(flex),
        "profile_status": PROFILE_STATUS,
        "profile": {
            "kind": "constant-area tapered tube, linear EI interpolation",
            "youngs_modulus_pa": properties.youngs_modulus,
            "density_kg_m3": properties.density,
            "butt_diameter_m": BUTT_DIAMETER_M,
            "tip_diameter_m": TIP_DIAMETER_M[club.name],
            "cpm_clamp_m": CPM_CLAMP_M,
            "poisson_ratio": POISSON_RATIO,
            "damping_ratio": DAMPING_RATIO,
        },
        "span_m": properties.length,
        "lumping": "midpoint joints s_i=(i+1/2)L/N, k_i=EI(s_i)/(L/N), c_i=beta k_i",
        "axes": "Rx, Rz bending about club x, z; Ry torsion; links in club frame",
        "sources": list(SOURCES),
    }


# --- specification transform --------------------------------------------------


def shaft_model(spec: Mapping[str, Any]) -> str:
    """``club.shaft.model`` of ``spec``; ``"rigid"`` when the record is absent."""
    shaft = (spec.get("club") or {}).get("shaft")
    model = DEFAULT_SHAFT_MODEL if shaft is None else shaft.get("model")
    if model not in SHAFT_MODELS:
        raise ValueError(f"unknown club.shaft.model {model!r}")
    return str(model)


def passive_coordinates(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Chain coordinates of ``spec`` (unactuated springs); empty when rigid."""
    if shaft_model(spec) == "rigid":
        return ()
    return tuple(spec["club"]["shaft"]["coordinates"])


def require_engine_support(spec: Mapping[str, Any] | bytes, engine: str) -> None:
    """DbC: refuse a chain-shaft spec for an engine that cannot apply it.

    An engine without chain support would replay the shaft joints as free
    or rigid joints; that is a different model, so it is an error, never a
    silent rigid replay.

    Raises:
        ShaftChainUnsupportedError: ``spec`` is a chain spec and ``engine`` is
            not in :data:`CHAIN_ENGINES`.
    """
    document = json.loads(spec) if isinstance(spec, bytes | bytearray) else spec
    if shaft_model(document) == "chain" and engine not in CHAIN_ENGINES:
        raise ShaftChainUnsupportedError(
            f"spec has club.shaft.model='chain' but engine {engine!r} has no "
            f"chain-shaft support (supported: {sorted(CHAIN_ENGINES)}); "
            "refusing a silent rigid replay (#12686)"
        )


def _club_body(document: Mapping[str, Any]) -> dict[str, Any]:
    body = next(
        (b for b in document["bodies"] if b["name"].endswith(CLUB_BODY_SUFFIX)), None
    )
    if body is None:
        raise ValueError("document has no club body")
    return body


def _shaft_piece(
    prefix: str, chain: ShaftChain, bounds: tuple[float, float], label: str
) -> dict[str, Any]:
    """Solid for the rod piece ``bounds`` (from the grip end) of the shaft."""
    start, end = bounds
    length = end - start
    mass = chain.club.shaft_mass_kg * length / chain.span_m
    centre_y = -chain.span_m + 0.5 * (start + end)
    return {
        "name": f"{prefix}/Shaft {label}",
        "mass_kg": mass,
        "placement": _placement((0.0, centre_y, HEAD_HEIGHT_M)),
        "com_m": [0.0, 0.0, 0.0],
        "inertia_com_kg_m2": _rod_inertia(mass, length, chain.club.shaft_radius_m),
    }


def _translation(point_y: float) -> list[list[float]]:
    return _placement((0.0, point_y, HEAD_HEIGHT_M))


def _chain_joint(prefix: str, chain: ShaftChain, i: int, parent: str) -> dict[str, Any]:
    point = _translation(-chain.span_m + chain.joint_s_m[i])
    springs = [chain.bend_stiffness[i], chain.bend_stiffness[i]]
    kinds = ["Rx", "Rz"]
    if chain.torsion:
        springs.append(chain.twist_stiffness[i])
        kinds.append("Ry")
    primitives = [
        {
            "primitive": kind,
            "coordinate": name,
            "stiffness_n_m_rad": k,
            "damping_n_m_s_rad": chain.damping_beta_s * k,
            "springref_rad": 0.0,
        }
        for kind, name, k in zip(
            kinds, _joint_coordinates(chain, i), springs, strict=True
        )
    ]
    return {
        "name": f"{prefix}/Shaft Joint {i}",
        "parent": parent,
        "child": f"{prefix}/Shaft Link {i}",
        "parent_to_base": point,
        "child_to_follower": point,
        "primitives": primitives,
    }


def _check_rigid_club(body: Mapping[str, Any], chain: ShaftChain) -> dict[str, Any]:
    """Precondition: the club body carries ``chain.club``'s rigid solids."""
    expected = {s["name"]: s for s in club_solids(body["name"], chain.club)}
    present = {s["name"]: s for s in body["solids"] if s["name"] in expected}
    if set(present) != set(expected) or any(
        not np.allclose(present[n]["placement"], expected[n]["placement"], atol=1e-12)
        or not math.isclose(
            present[n]["mass_kg"], expected[n]["mass_kg"], abs_tol=1e-15
        )
        for n in expected
    ):
        raise ValueError(f"club body does not carry the rigid {chain.club.name} solids")
    return present


def apply_shaft_chain(document: Mapping[str, Any], chain: ShaftChain) -> dict[str, Any]:
    """Return a copy of ``document`` with the club shaft replaced by ``chain``.

    Preconditions: ``document`` is a rigid-shaft spec whose club body carries
    exactly ``club_solids(prefix, chain.club)`` (``ValueError`` otherwise).
    Postconditions: ``club.shaft.model == "chain"``; the chain bodies and
    joints and their coordinates are appended (spec order unchanged before
    them); total mass, centre of mass and inertia at zero deflection equal
    the rigid club's; the ``Clubhead`` frame is on the tip link.
    """
    if shaft_model(document) != "rigid":
        raise ValueError("document already has a flexible shaft")
    doc = json.loads(json.dumps(document))
    body = _club_body(doc)
    prefix = body["name"]
    rigid = _check_rigid_club(body, chain)
    head, shaft = rigid[f"{prefix}/Clubhead"], rigid[f"{prefix}/Rigid Shaft"]
    root_piece = _shaft_piece(prefix, chain, (0.0, chain.joint_s_m[0]), "Root")
    body["solids"] = [
        s if s is not shaft else root_piece for s in body["solids"] if s is not head
    ]
    parent = prefix
    for i, bounds in enumerate(chain.link_bounds_m):
        link_solids = [_shaft_piece(prefix, chain, bounds, f"Link {i}")]
        if i == chain.n_links - 1:
            link_solids.append(head)
        link: dict[str, Any] = {
            "name": f"{prefix}/Shaft Link {i}",
            "solids": link_solids,
        }
        doc["bodies"].append(link)
        doc["joints"].append(_chain_joint(prefix, chain, i, parent))
        parent = link["name"]
    for frame in doc["frames"]:
        if frame["body"] == prefix and frame["name"] == "Clubhead":
            frame["body"] = parent
    doc["coordinate_order"] = list(doc["coordinate_order"]) + list(chain.coordinates())
    doc["club"]["shaft"] = {
        **chain.record,
        "coordinates": list(chain.coordinates()),
        "tip_link": parent,
        "club_body": prefix,
    }
    return doc


def clamped_shaft_document(
    chain: ShaftChain, *, gravity: Sequence[float] = (0.0, 0.0, 0.0)
) -> dict[str, Any]:
    """Grip-clamped shaft plus head as a minimal spec (reference studies).

    The club body is welded to the world (a joint with no primitives) with
    only the root shaft piece; the links and joints are those of
    :func:`apply_shaft_chain`. Used by the static, modal and render checks.
    """
    prefix = f"clamped_{chain.club.name}{CLUB_BODY_SUFFIX}"
    head = next(
        s for s in club_solids(prefix, chain.club) if s["name"].endswith("Clubhead")
    )
    bodies = [
        {"name": "world", "solids": []},
        {
            "name": prefix,
            "solids": [_shaft_piece(prefix, chain, (0.0, chain.joint_s_m[0]), "Root")],
        },
    ]
    joints = [
        {
            "name": f"{prefix}/Clamp",
            "parent": "world",
            "child": prefix,
            "parent_to_base": np.eye(4).tolist(),
            "child_to_follower": np.eye(4).tolist(),
            "primitives": [],
        }
    ]
    parent = prefix
    for i, bounds in enumerate(chain.link_bounds_m):
        solids = [_shaft_piece(prefix, chain, bounds, f"Link {i}")]
        if i == chain.n_links - 1:
            solids.append(head)
        bodies.append({"name": f"{prefix}/Shaft Link {i}", "solids": solids})
        joints.append(_chain_joint(prefix, chain, i, parent))
        parent = f"{prefix}/Shaft Link {i}"
    return {
        "gravity_m_s2": list(gravity),
        "coordinate_order": list(chain.coordinates()),
        "bodies": bodies,
        "joints": joints,
        "tip_body": parent,
        "tip_point_m": [0.0, 0.0, HEAD_HEIGHT_M],
    }
