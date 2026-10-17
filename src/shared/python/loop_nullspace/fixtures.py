"""Closed-loop test fixtures with NSX-00 goldens (NSX-12, issue #12557).

Every fixture returns a validated ``ClosedLoopSnapshot`` (NSX-10 contracts)
together with the ``GoldenValues`` record loaded from the NSX-00 JSON in
``tests/fixtures/loop_nullspace/goldens``.  The goldens come only from that
JSON, which is produced by the sympy derivations in
``docs/research/loop_nullspace/derivations``; the snapshot matrices here are
computed by the independent numpy model in ``planar_model`` (or Pinocchio, see
``spatial_fixture``) so a test comparing the two checks two code paths.

Non-claim: these are structural fixtures with generic geometry, not golfer
models, and they say nothing about physiological plausibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    planar_dynamics,
)

Array = NDArray[np.float64]
Actuation = Literal["full", "club_unactuated"]

__all__ = [
    "Actuation",
    "AnalyticEnvelopeCase",
    "EnvelopeBound",
    "FixtureCase",
    "GoldenValues",
    "PlanarGolferConfig",
    "analytic_envelope_case",
    "generic_planar_config",
    "load_golden",
    "planar_golfer_snapshot",
    "spatial_two_arm_club_snapshot",
    "straight_arm_config",
    "straight_arm_snapshot",
]

GOLDENS_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
)
CLOSURE_ATOL = 1e-9
_ACTUATED = {"full": (0, 1, 2, 3, 4), "club_unactuated": (0, 1, 3, 4)}
_PLANAR_GOLDEN = {
    "full": "planar_two_arm_full",
    "club_unactuated": "planar_two_arm_club_unactuated",
}
_COORDINATES = (
    "lead_shoulder",
    "lead_elbow",
    "club_angle",
    "trail_shoulder",
    "trail_elbow",
)


@dataclass(frozen=True)
class GoldenValues:
    """One NSX-00 golden record: file stem, optional variant and the payload.

    ``payload`` is the parsed JSON object (the variant sub-object when a
    variant is selected); treat it as read-only.
    """

    name: str
    payload: dict[str, Any]
    variant: str | None = None
    content_hash: str = field(default="", repr=False)

    def get(self, *path: str) -> Any:
        """Return the JSON value at ``path`` (``KeyError`` names the miss)."""
        node: Any = self.payload
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"golden {self.name!r} has no entry {'/'.join(path)!r}")
            node = node[key]
        return node

    def array(self, *path: str) -> Array:
        """Return the numeric JSON value at ``path`` as a float64 array."""
        return np.asarray(self.get(*path), dtype=np.float64)

    def fractions(self, *path: str) -> list[Any]:
        """Return an exact-rational entry (nested lists of ``Fraction``)."""

        def conv(node: Any) -> Any:
            if isinstance(node, list):
                return [conv(x) for x in node]
            return Fraction(node)

        return conv(self.get(*path))  # type: ignore[no-any-return]


def load_golden(name: str, variant: str | None = None) -> GoldenValues:
    """Load ``<name>.json`` from the NSX-00 goldens directory.

    Args:
        name: File stem, e.g. ``"planar_two_arm_full"``.
        variant: Key under ``"variants"`` for multi-variant goldens.

    Raises:
        FileNotFoundError: If the golden file does not exist.
        ValueError: If ``variant`` is not present or the schema tag is wrong.
    """
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"golden name must be a plain file stem, got {name!r}")
    path = GOLDENS_DIR / f"{name}.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema") != "loop-nullspace-golden/1":
        raise ValueError(
            f"golden {name!r} has unexpected schema {data.get('schema')!r}"
        )
    if variant is not None:
        variants = data.get("variants", {})
        if variant not in variants:
            raise ValueError(f"golden {name!r} has no variant {variant!r}")
        data = variants[variant]
    return GoldenValues(name, data, variant, hashlib.sha256(raw).hexdigest())


class FixtureCase(NamedTuple):
    """A snapshot and the golden it must reproduce (``None`` if not pinned)."""

    snapshot: ClosedLoopSnapshot
    golden: GoldenValues | None


# ------------------------------------------------------------------ planar --


@dataclass(frozen=True)
class PlanarGolferConfig:
    """A planar state ``(q, v, a)``; ``name`` links it to a pinned golden."""

    q: Array
    v: Array
    a: Array
    name: str = "custom"

    def __post_init__(self) -> None:
        for key in ("q", "v", "a"):
            arr = np.array(getattr(self, key), dtype=np.float64, copy=True)
            if arr.shape != (5,) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"PlanarGolferConfig.{key} must be a finite (5,) vector"
                )
            arr.flags.writeable = False
            object.__setattr__(self, key, arr)


def generic_planar_config() -> PlanarGolferConfig:
    """The closed, non-singular NSX-00 configuration (state from the golden)."""
    state = load_golden("planar_two_arm_full").get("state")
    return PlanarGolferConfig(state["q"], state["v"], state["a"], name="generic")


def straight_arm_config() -> PlanarGolferConfig:
    """The collinear NSX-00 configuration (``J_c`` has rank 1)."""
    state = load_golden("planar_straight_arm_rank_deficient", "full").get("state")
    return PlanarGolferConfig(state["q"], state["v"], state["a"], name="straight_arm")


def _check_actuation(actuation: str) -> None:
    if actuation not in _ACTUATED:
        raise ValueError(
            f"actuation must be one of {sorted(_ACTUATED)}, got {actuation!r}"
        )


def _check_closed(dyn: dict[str, Array], cfg: PlanarGolferConfig) -> None:
    jc = dyn["Jc"]
    residuals = {
        "position closure Phi(q)": dyn["phi"],
        "velocity closure J_c v": jc @ cfg.v,
        "acceleration closure J_c a + Jdot v": jc @ cfg.a + dyn["Jdot_v"],
    }
    for label, res in residuals.items():
        if float(np.max(np.abs(res))) > CLOSURE_ATOL:
            raise ValueError(
                f"config {cfg.name!r} violates the {label} (max "
                f"{float(np.max(np.abs(res))):.3e} > {CLOSURE_ATOL:g})"
            )


def _loop_support(jc: Array) -> tuple[int, ...]:
    return tuple(int(j) for j in np.flatnonzero(np.max(np.abs(jc), axis=0) > 1e-12))


def _planar_snapshot(
    cfg: PlanarGolferConfig,
    actuation: Actuation,
    params: PlanarGolferParams,
    source: str,
    content_hash: str,
) -> ClosedLoopSnapshot:
    dyn = planar_dynamics(cfg.q, cfg.v, params)
    _check_closed(dyn, cfg)
    actuated = _ACTUATED[actuation]
    unact = tuple(i for i in range(5) if i not in actuated)
    topo = LoopTopology(
        ClosureKind.PLANAR_POINT2,
        "trail",
        "lead",
        actuated,
        unact,
        _loop_support(dyn["Jc"]),
        _COORDINATES,
    )
    grip = np.array([dyn["grip"][0], dyn["grip"][1], 0.0])
    axis = np.array([np.sin(cfg.q[:3].sum()), -np.cos(cfg.q[:3].sum()), 0.0])
    moment = params.club_mass * params.club_length**2 / 12.0
    club = SpatialInertia(
        params.club_mass,
        (params.club_com_from_lead_hand - params.trail_grip_from_lead_hand) * axis,
        moment * (np.eye(3) - np.outer(axis, axis)),
    )
    return ClosedLoopSnapshot(
        q=cfg.q,
        v=cfg.v,
        a=cfg.a,
        M=dyn["M"],
        h=dyn["h"],
        S=np.eye(5)[list(actuated)],
        J_c=dyn["Jc"],
        Jdot_v=dyn["Jdot_v"],
        grip_frame=GripFrame(np.eye(3), grip),
        bare_club=club,
        welded_hand_inertias=(),
        model_provenance=ModelProvenance(
            "planar_two_arm_golfer_numpy", "1", content_hash
        ),
        input_provenance=InputProvenance(
            "synthetic", f"nsx-00-golden:{source}", f"actuation={actuation}"
        ),
        topology=topo,
    )


def planar_golfer_snapshot(
    config: PlanarGolferConfig | None = None,
    *,
    actuation: Actuation = "full",
    closure: ClosureKind = ClosureKind.PLANAR_POINT2,
) -> FixtureCase:
    """Planar two-arm golfer snapshot (fixture (a)).

    Args:
        config: State to evaluate; default is ``generic_planar_config()``.
        actuation: ``"full"`` (``S = I``) or ``"club_unactuated"`` (the club
            angle ``q3`` carries no actuator).
        closure: Only ``PLANAR_POINT2`` is pinned by NSX-00.

    Returns:
        ``FixtureCase(snapshot, golden)``; ``golden`` is the NSX-00 record for
        a named config (``generic`` / ``straight_arm``) and ``None`` for a
        custom state.

    Raises:
        ValueError: Unknown ``actuation``, a ``closure`` other than
            ``PLANAR_POINT2``, or a state that violates the position-,
            velocity- or acceleration-level closure.
    """
    _check_actuation(actuation)
    if closure is not ClosureKind.PLANAR_POINT2:
        raise ValueError(
            f"NSX-00 pins only PLANAR_POINT2 for the planar golfer, got {closure.name}"
        )
    cfg = config if config is not None else generic_planar_config()
    base = load_golden(_PLANAR_GOLDEN[actuation])
    params = PlanarGolferParams.from_golden(base)
    snap = _planar_snapshot(cfg, actuation, params, base.name, base.content_hash)
    golden = base if cfg.name == "generic" else None
    if cfg.name == "straight_arm":
        golden = load_golden("planar_straight_arm_rank_deficient", actuation)
    return FixtureCase(snap, golden)


def straight_arm_snapshot(*, actuation: Actuation = "full") -> FixtureCase:
    """Rank-deficient collinear configuration (fixture (b); ``rank J_c = 1``)."""
    return planar_golfer_snapshot(straight_arm_config(), actuation=actuation)


# ---------------------------------------------------------------- envelope --


@dataclass(frozen=True)
class EnvelopeBound:
    """Exact closed-form bounds of one quantity over the multiplier polygon."""

    lower: Fraction
    upper: Fraction
    sign_class: str


@dataclass(frozen=True)
class AnalyticEnvelopeCase:
    """Three torque capacities over a 2-multiplier closure, solved exactly.

    ``tau(lam) = tau_ID - J_c^T lam`` with ``|tau_i| <= capacity_i`` defines
    the polygon ``polytope_g @ lam <= polytope_h``; ``bounds`` holds the exact
    min/max of ``tau_1..3`` and ``lam_1..2`` over it (NSX-00 section on the
    analytic envelope).
    """

    J_c_T: Array
    tau_ID: Array
    capacity: Array
    polytope_g: Array
    polytope_h: Array
    row_labels: tuple[str, ...]
    vertices: list[tuple[Fraction, ...]]
    bounds: dict[str, EnvelopeBound]
    golden: GoldenValues

    def affine_functional(self, name: str) -> tuple[Array, float]:
        """Return ``(coeff, const)`` with quantity ``name = const + coeff . lam``."""
        if name.startswith("lam_"):
            coeff = np.zeros(2)
            coeff[int(name[-1]) - 1] = 1.0
            return coeff, 0.0
        if name.startswith("tau_"):
            idx = int(name[-1]) - 1
            return -self.J_c_T[idx], float(self.tau_ID[idx])
        raise ValueError(f"unknown envelope quantity {name!r}")


def analytic_envelope_case() -> AnalyticEnvelopeCase:
    """Closed-form envelope case (fixture (d)) read from the NSX-00 golden."""
    gold = load_golden("envelope_analytic_lp")
    static = GoldenValues(gold.name, gold.get("static"), "static", gold.content_hash)
    rows = static.get("rows")
    bounds = {
        key: EnvelopeBound(
            Fraction(ent["lower"]), Fraction(ent["upper"]), ent["sign_class"]
        )
        for key, ent in static.get("envelopes").items()
    }
    verts = [tuple(Fraction(c) for c in v) for v in static.get("vertices")]
    return AnalyticEnvelopeCase(
        J_c_T=np.array([[float(Fraction(x)) for x in r] for r in static.get("J_c_T")]),
        tau_ID=np.array([float(Fraction(x)) for x in static.get("tau_ID")]),
        capacity=np.array([float(Fraction(x)) for x in static.get("capacity")]),
        polytope_g=np.array(
            [[float(Fraction(x)) for x in r["g"]] for r in rows], dtype=float
        ),
        polytope_h=np.array([float(Fraction(r["h"])) for r in rows], dtype=float),
        row_labels=tuple(r["label"] for r in rows),
        vertices=verts,
        bounds=bounds,
        golden=gold,
    )


# ----------------------------------------------------------------- spatial --


def spatial_two_arm_club_snapshot(
    *,
    floating_root: bool = False,
    feet: bool = False,
    foot_points: int = 1,
    zero_root_pose: bool = False,
) -> FixtureCase:
    """Spatial 2 x 7-DOF arm chain with a rigid bare club and a WELD6 closure.

    Built with Pinocchio (fixture (c)); see ``spatial_fixture`` for the model.
    ``foot_points=2`` gives two rigid points per foot (NSX-60 lower-body variant);
    ``zero_root_pose`` puts the root at the origin (matches the symbolic golden).
    Raises ``ValueError`` if ``feet`` is requested without ``floating_root``.
    """
    from src.shared.python.loop_nullspace import spatial_fixture

    return spatial_fixture.build_case(
        floating_root=floating_root,
        feet=feet,
        foot_points=foot_points,
        zero_root_pose=zero_root_pose,
    )
