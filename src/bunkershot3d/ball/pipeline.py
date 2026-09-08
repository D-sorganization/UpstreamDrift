"""Pipeline integration for bunkershot3d ball model (issues #8613, #8657).

Provides the handoff between bunkershot3d and the existing
``SwingBallFlightPipeline`` infrastructure: a bunker shot in, a
:class:`~src.shared.python.physics.impact_model.PostImpactState` out, ready
for flight simulation.

Everything measured arrives through
:class:`~bunkershot3d.ball.splash.SandDelivery` -- the solver's impulse and
entry/exit speeds, and the metrics layer's divot mass. What remains on
:class:`BunkerShotState` is the club's declared geometry and mass and the lie,
none of which the solver measures. The old ``entry_depth_m``,
``sole_width_m`` and ``sole_length_m`` fields existed only to feed the deleted
box-volume estimate of displaced sand (issue #8657) and are gone with it.

Usage::

    state = BunkerShotState(club_loft_deg=56.0, ball_lie=..., delivery=...)
    result = compute_bunker_launch(state)
    post = to_post_impact_state(result, state)
    # Then use SwingBallFlightPipeline with post_state
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from ..provenance.hashing import canonical_json
from ..solvers.envelope import (
    Caveat,
    DimensionlessGroups,
    EnvelopeStatus,
    FeatureScale,
    ValidityVerdict,
)
from ..solvers.protocol import FidelityTier
from src.shared.python.core.contracts import require
from src.shared.python.physics.impact_model import PostImpactState

from .lie import BallLie, BallProperties
from .splash import (
    DEFAULT_MOMENTUM_TRANSFER,
    BallLaunchResult,
    MomentumTransfer,
    SandDelivery,
    compute_ball_launch_from_splash,
)

__all__ = [
    "HEAD_FRAME_TO_FLIGHT_TRANSFORM",
    "BunkerShotState",
    "ExitVectorProvenance",
    "HandoffFrame",
    "PostImpactEnvelope",
    "compute_bunker_launch",
    "post_impact_envelope",
    "to_post_impact_state",
]


@dataclass(slots=True)
class BunkerShotState:
    """Complete specification of a bunker shot.

    Attributes:
        club_loft_deg: Effective loft at delivery [degrees].
        ball_lie: Ball position and burial in sand.
        delivery: What the solver and metrics layer measured about the strike.
        club_mass_kg: Club head mass [kg].
        ball: Ball properties.
        transfer: The uncalibrated sand-to-ball partition parameters.
    """

    club_loft_deg: float
    ball_lie: BallLie
    delivery: SandDelivery
    club_mass_kg: float = 0.30  # 300g wedge head
    ball: BallProperties = field(default_factory=BallProperties)
    transfer: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER

    def __post_init__(self) -> None:
        """Validate the declared club properties.

        Raises:
            ValueError: If the delivery is not a measured strike. The launch
                may not fall back on a defaulted one.
        """
        require(0 < self.club_loft_deg < 90, "loft must be in (0, 90) degrees")
        require(self.club_mass_kg > 0, "club mass must be positive")
        if not isinstance(self.delivery, SandDelivery):
            raise ValueError(
                "a bunker shot must carry the strike the solver measured; got "
                f"{type(self.delivery).__name__}. Ball launch is derived from "
                "the delivered impulse and the divot mass (issue #8657), and "
                "neither has a sensible default"
            )


def compute_bunker_launch(state: BunkerShotState) -> BallLaunchResult:
    """Compute ball launch conditions from bunker shot.

    This is the main entry point for bunker shot physics. Thin/blade direct
    contact remains out of scope, so the splash transfer is always used.

    Args:
        state: Complete bunker shot specification.

    Returns:
        The launch, its validity verdict and the provenance of the partition.
    """
    return compute_ball_launch_from_splash(
        lie=state.ball_lie,
        ball=state.ball,
        delivery=state.delivery,
        club_loft_rad=math.radians(state.club_loft_deg),
        club_mass_kg=state.club_mass_kg,
        transfer=state.transfer,
    )


def to_post_impact_state(
    result: BallLaunchResult,
    state: BunkerShotState,
) -> PostImpactState:
    """Convert a launch result to the flight pipeline's input contract.

    The clubhead's state after the sand is the solver's, not an estimate: it
    left the bed at :attr:`~bunkershot3d.ball.splash.SandDelivery.exit_speed_m_s`
    after :attr:`~bunkershot3d.ball.splash.SandDelivery.contact_duration_s` of
    engagement. Only the *direction* it leaves in is a convention, taken from
    the loft as the launch direction is.

    Args:
        result: Ball launch result from the bunker shot.
        state: Original bunker shot state.

    Returns:
        PostImpactState ready for flight simulation.
    """
    delivery = state.delivery
    club_loft_rad = math.radians(state.club_loft_deg)
    ball_angular_velocity = np.array(result.ball_angular_velocity, dtype=float)

    if delivery.exit_velocity_m_s is not None:
        club_velocity = np.array(delivery.exit_velocity_m_s, dtype=float)
    else:
        club_speed_out = delivery.exit_speed_m_s
        club_velocity = np.array(
            [
                club_speed_out * math.cos(club_loft_rad * 0.3),  # Forward
                0.0,  # No lateral
                -club_speed_out * math.sin(club_loft_rad * 0.3),  # Down through sand
            ],
            dtype=float,
        )

    if delivery.exit_angular_velocity_rad_s is not None:
        clubhead_angular_velocity = np.array(
            delivery.exit_angular_velocity_rad_s, dtype=float
        )
    else:
        clubhead_angular_velocity = np.zeros(3, dtype=float)

    ball_ke = 0.5 * state.ball.mass_kg * result.ball_speed_m_s**2
    ball_rot_ke = (
        0.5 * state.ball.moi_kg_m2 * float(np.linalg.norm(ball_angular_velocity)) ** 2
    )

    return PostImpactState(
        ball_velocity=np.array(result.ball_velocity, dtype=float),
        ball_angular_velocity=ball_angular_velocity,
        clubhead_velocity=club_velocity,
        clubhead_angular_velocity=clubhead_angular_velocity,
        contact_duration=delivery.contact_duration_s,
        energy_transfer=float(ball_ke + ball_rot_ke),
        impact_location=np.zeros(2, dtype=float),
    )


HEAD_FRAME_TO_FLIGHT_TRANSFORM: tuple[tuple[float, float, float], ...] = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)
"""The declared rotation from the bunker solver's world frame to the flight
frame (issue #9542).

The F0 solver's world has the head travelling toward ``-x``
(:data:`~bunkershot3d.geometry.delivery.TRAVEL_AXIS_BODY`); the flight
pipeline takes ``[x=forward, y=left, z=up]``. Rotating pi about ``z`` maps
the travel axis ``-x`` onto forward ``+x`` and is a **proper** rotation
(``det = +1``) -- never the axis reflection a naive sign flip would be.
Axial (angular) vectors transform with the same rotation.
"""


class ExitVectorProvenance(StrEnum):
    """Whether a boundary vector is a measurement or a modelling convention.

    Issue #9542: the adapter used to synthesise a clubhead direction from the
    loft with zero spin and pass it off as the exit state. Nothing may be
    inferred about which kind a vector is; it is carried explicitly.
    """

    ACTUAL_EXIT_STATE = "actual_exit_state"
    """Measured by the solver and supplied through the delivery."""

    MODELED_CONVENTION = "modeled_convention"
    """Synthesised by a documented convention. Never a measurement."""


class HandoffFrame(StrEnum):
    """The coordinate frame a handoff vector is expressed in."""

    BUNKER_HEAD_WORLD = "bunker_head_world"
    """The F0 solver's world frame: the head travels toward ``-x``."""

    BUNKER_LAUNCH = "bunker_launch"
    """The launch model's frame: ``+x`` toward the target, ``+z`` up. The
    convention already matches the flight pipeline's ``[x=forward,
    y=left, z=up]``, so its transform is the identity."""


_FRAME_TRANSFORMS: Mapping[HandoffFrame, tuple[tuple[float, float, float], ...]] = {
    HandoffFrame.BUNKER_HEAD_WORLD: HEAD_FRAME_TO_FLIGHT_TRANSFORM,
    HandoffFrame.BUNKER_LAUNCH: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


_PROVENANCE_FRAME: Mapping[ExitVectorProvenance, HandoffFrame] = {
    ExitVectorProvenance.ACTUAL_EXIT_STATE: HandoffFrame.BUNKER_HEAD_WORLD,
    ExitVectorProvenance.MODELED_CONVENTION: HandoffFrame.BUNKER_LAUNCH,
}


def _frozen_vector(values: Any) -> np.ndarray:
    """Return an owned, immutable ``(3,)`` float copy of ``values``."""
    array = np.array(values, dtype=float, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PostImpactEnvelope:
    """The versioned result envelope around the flight handoff (issue #9542).

    The legacy :func:`to_post_impact_state` boundary carries only seven
    fields and no statement of what they are. This envelope wraps that state
    with everything a downstream consumer needs to quote it: the validity
    verdict, the fidelity tier, whether each clubhead vector is a
    measurement or a convention, the frame each group lives in, and a digest
    of the whole payload.

    Attributes:
        schema_version: Envelope schema version.
        post_impact_state: The Tools-owned flight input, with owned
            immutable arrays.
        verdict: The launch verdict carried onto the boundary.
        fidelity_tier: ADR-0032 rung that produced the strike.
        clubhead_velocity_provenance: Actual versus modelled clubhead exit.
        clubhead_angular_velocity_provenance: Actual versus modelled spin.
        ball_launch_provenance: Always ``MODELED_CONVENTION`` -- the splash
            partition has no measured ball vector.
        clubhead_velocity_frame: Frame of the clubhead velocity vector.
        clubhead_angular_velocity_frame: Frame of the clubhead spin vector.
        ball_frame: Frame of the ball vectors.
        source_digest: SHA-256 of the canonical payload this envelope carries.
    """

    SCHEMA_VERSION = 1

    schema_version: int
    post_impact_state: PostImpactState
    verdict: ValidityVerdict
    fidelity_tier: FidelityTier
    clubhead_velocity_provenance: ExitVectorProvenance
    clubhead_angular_velocity_provenance: ExitVectorProvenance
    ball_launch_provenance: ExitVectorProvenance
    clubhead_velocity_frame: HandoffFrame
    clubhead_angular_velocity_frame: HandoffFrame
    ball_frame: HandoffFrame
    source_digest: str

    def __post_init__(self) -> None:
        """Refuse an envelope that mislabels its own contents.

        Raises:
            ValueError: If the schema version is unknown, a field is of the
                wrong type, a frame contradicts its provenance, or the
                digest is not a SHA-256 hex string.
        """
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported envelope schema_version {self.schema_version!r}; "
                f"expected {self.SCHEMA_VERSION}"
            )
        if not isinstance(self.verdict, ValidityVerdict):
            raise ValueError(
                "an envelope must carry the launch verdict, got "
                f"{type(self.verdict).__name__}"
            )
        if not isinstance(self.fidelity_tier, FidelityTier):
            raise ValueError("fidelity_tier must be a FidelityTier")
        _require_provenance_pair(
            "clubhead_velocity",
            self.clubhead_velocity_provenance,
            self.clubhead_velocity_frame,
        )
        _require_provenance_pair(
            "clubhead_angular_velocity",
            self.clubhead_angular_velocity_provenance,
            self.clubhead_angular_velocity_frame,
        )
        _require_provenance_pair(
            "ball_launch", self.ball_launch_provenance, self.ball_frame
        )
        if len(self.source_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_digest
        ):
            raise ValueError("source_digest must be a 64-character SHA-256 digest")

    def clubhead_velocity_in_flight_frame(self) -> np.ndarray:
        """Return the clubhead exit velocity expressed in the flight frame."""
        return _to_flight_frame(
            self.post_impact_state.clubhead_velocity, self.clubhead_velocity_frame
        )

    def clubhead_angular_velocity_in_flight_frame(self) -> np.ndarray:
        """Return the clubhead exit spin expressed in the flight frame."""
        return _to_flight_frame(
            self.post_impact_state.clubhead_angular_velocity,
            self.clubhead_angular_velocity_frame,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe payload, digest included, for save/reload."""
        post = self.post_impact_state
        return {
            "schema_version": self.schema_version,
            "post_impact_state": {
                "ball_velocity": post.ball_velocity.tolist(),
                "ball_angular_velocity": post.ball_angular_velocity.tolist(),
                "clubhead_velocity": post.clubhead_velocity.tolist(),
                "clubhead_angular_velocity": post.clubhead_angular_velocity.tolist(),
                "contact_duration": post.contact_duration,
                "energy_transfer": post.energy_transfer,
                "impact_location": post.impact_location.tolist(),
            },
            "verdict": _verdict_payload(self.verdict),
            "fidelity_tier": self.fidelity_tier.value,
            "clubhead_velocity_provenance": self.clubhead_velocity_provenance.value,
            "clubhead_angular_velocity_provenance": (
                self.clubhead_angular_velocity_provenance.value
            ),
            "ball_launch_provenance": self.ball_launch_provenance.value,
            "clubhead_velocity_frame": self.clubhead_velocity_frame.value,
            "clubhead_angular_velocity_frame": self.clubhead_angular_velocity_frame.value,
            "ball_frame": self.ball_frame.value,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PostImpactEnvelope:
        """Rebuild an envelope from :meth:`to_dict` output.

        Args:
            payload: The serialized envelope.

        Returns:
            The reloaded envelope.

        Raises:
            ValueError: If the schema version is unknown, the payload was
                tampered with (digest mismatch), or a field is malformed.
        """
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported envelope schema_version "
                f"{payload.get('schema_version')!r}; expected {cls.SCHEMA_VERSION}"
            )
        body = {key: value for key, value in payload.items() if key != "source_digest"}
        digest = _payload_digest(body)
        if digest != payload.get("source_digest"):
            raise ValueError(
                "the envelope payload does not match its source_digest; it was "
                "changed after it was written"
            )
        post_payload = payload["post_impact_state"]
        return cls(
            schema_version=payload["schema_version"],
            post_impact_state=PostImpactState(
                ball_velocity=_frozen_vector(post_payload["ball_velocity"]),
                ball_angular_velocity=_frozen_vector(
                    post_payload["ball_angular_velocity"]
                ),
                clubhead_velocity=_frozen_vector(post_payload["clubhead_velocity"]),
                clubhead_angular_velocity=_frozen_vector(
                    post_payload["clubhead_angular_velocity"]
                ),
                contact_duration=float(post_payload["contact_duration"]),
                energy_transfer=float(post_payload["energy_transfer"]),
                impact_location=_frozen_vector(post_payload["impact_location"]),
            ),
            verdict=_verdict_from_payload(payload["verdict"]),
            fidelity_tier=FidelityTier(payload["fidelity_tier"]),
            clubhead_velocity_provenance=ExitVectorProvenance(
                payload["clubhead_velocity_provenance"]
            ),
            clubhead_angular_velocity_provenance=ExitVectorProvenance(
                payload["clubhead_angular_velocity_provenance"]
            ),
            ball_launch_provenance=ExitVectorProvenance(
                payload["ball_launch_provenance"]
            ),
            clubhead_velocity_frame=HandoffFrame(payload["clubhead_velocity_frame"]),
            clubhead_angular_velocity_frame=HandoffFrame(
                payload["clubhead_angular_velocity_frame"]
            ),
            ball_frame=HandoffFrame(payload["ball_frame"]),
            source_digest=digest,
        )


def _require_provenance_pair(
    group: str, provenance: ExitVectorProvenance, frame: HandoffFrame
) -> None:
    """Refuse a vector group whose frame contradicts its provenance."""
    if not isinstance(provenance, ExitVectorProvenance):
        raise ValueError(f"{group}_provenance must be an ExitVectorProvenance")
    if not isinstance(frame, HandoffFrame):
        raise ValueError(f"{group}_frame must be a HandoffFrame")
    expected = _PROVENANCE_FRAME[provenance]
    if frame is not expected:
        raise ValueError(
            f"{group}_frame {frame.value!r} contradicts its provenance "
            f"{provenance.value!r}; a {provenance.value} vector is expressed "
            f"in {expected.value}"
        )


def _payload_digest(body: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of an envelope payload without its digest."""
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _to_flight_frame(vector: Any, frame: HandoffFrame) -> np.ndarray:
    """Express ``vector`` from its handoff ``frame`` in the flight frame."""
    return np.array(_FRAME_TRANSFORMS[frame], dtype=float) @ np.asarray(
        vector, dtype=float
    )


def _verdict_payload(verdict: ValidityVerdict) -> dict[str, Any]:
    """Return the JSON-safe form of a validity verdict."""
    return {
        "status": verdict.status.value,
        "groups": [
            {
                "scale_name": group.scale.name,
                "scale_length_m": group.scale.length_m,
                "speed_m_s": group.speed_m_s,
                "grain_diameter_m": group.grain_diameter_m,
                "froude": group.froude,
                "micro_inertial_number": group.micro_inertial_number,
                "grain_size_ratio": group.grain_size_ratio,
                "continuum_size_ratio": group.continuum_size_ratio,
                "macro_inertial_number": group.macro_inertial_number,
                "element_size_m": group.element_size_m,
            }
            for group in verdict.groups
        ],
        "caveats": [caveat.value for caveat in verdict.caveats],
        "reasons": list(verdict.reasons),
        "governing_index": verdict.governing_index,
        "clamped_area_fraction": verdict.clamped_area_fraction,
        "details": dict(verdict.details),
    }


def _verdict_from_payload(payload: Mapping[str, Any]) -> ValidityVerdict:
    """Rebuild a validity verdict from :func:`_verdict_payload` output."""
    return ValidityVerdict(
        status=EnvelopeStatus(payload["status"]),
        groups=tuple(
            DimensionlessGroups(
                scale=FeatureScale(
                    name=group["scale_name"], length_m=group["scale_length_m"]
                ),
                speed_m_s=group["speed_m_s"],
                grain_diameter_m=group["grain_diameter_m"],
                froude=group["froude"],
                micro_inertial_number=group["micro_inertial_number"],
                grain_size_ratio=group["grain_size_ratio"],
                continuum_size_ratio=group["continuum_size_ratio"],
                macro_inertial_number=group["macro_inertial_number"],
                element_size_m=group["element_size_m"],
            )
            for group in payload["groups"]
        ),
        caveats=tuple(Caveat(caveat) for caveat in payload["caveats"]),
        reasons=tuple(payload["reasons"]),
        governing_index=payload["governing_index"],
        clamped_area_fraction=payload["clamped_area_fraction"],
        details=payload["details"],
    )


def post_impact_envelope(
    result: BallLaunchResult, state: BunkerShotState
) -> PostImpactEnvelope:
    """Build the versioned result envelope for one bunker launch (issue #9542).

    The legacy :func:`to_post_impact_state` adapter stays available for
    existing callers; the envelope is the boundary that can state what its
    numbers are, which frame they live in, and whether any of them was
    measured.

    Args:
        result: Ball launch result from the bunker shot.
        state: Original bunker shot state.

    Returns:
        The envelope around the flight handoff state.
    """
    delivery = state.delivery
    post = to_post_impact_state(result, state)
    velocity_provenance = (
        ExitVectorProvenance.ACTUAL_EXIT_STATE
        if delivery.exit_velocity_m_s is not None
        else ExitVectorProvenance.MODELED_CONVENTION
    )
    angular_provenance = (
        ExitVectorProvenance.ACTUAL_EXIT_STATE
        if delivery.exit_angular_velocity_rad_s is not None
        else ExitVectorProvenance.MODELED_CONVENTION
    )
    payload = PostImpactEnvelope(
        schema_version=PostImpactEnvelope.SCHEMA_VERSION,
        post_impact_state=PostImpactState(
            ball_velocity=_frozen_vector(post.ball_velocity),
            ball_angular_velocity=_frozen_vector(post.ball_angular_velocity),
            clubhead_velocity=_frozen_vector(post.clubhead_velocity),
            clubhead_angular_velocity=_frozen_vector(post.clubhead_angular_velocity),
            contact_duration=post.contact_duration,
            energy_transfer=post.energy_transfer,
            impact_location=_frozen_vector(post.impact_location),
        ),
        verdict=result.verdict,
        fidelity_tier=FidelityTier.F0,
        clubhead_velocity_provenance=velocity_provenance,
        clubhead_angular_velocity_provenance=angular_provenance,
        ball_launch_provenance=ExitVectorProvenance.MODELED_CONVENTION,
        clubhead_velocity_frame=_PROVENANCE_FRAME[velocity_provenance],
        clubhead_angular_velocity_frame=_PROVENANCE_FRAME[angular_provenance],
        ball_frame=HandoffFrame.BUNKER_LAUNCH,
        source_digest="0" * 64,
    )
    body = {
        key: value for key, value in payload.to_dict().items() if key != "source_digest"
    }
    digest = _payload_digest(body)
    return replace(payload, source_digest=digest)
