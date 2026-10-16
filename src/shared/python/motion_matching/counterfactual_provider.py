"""Resolve the dynamics a candidate session's counterfactuals integrate (NSX-30).

A :class:`CandidateSession` carries the full model specification it was fit
against (``session.specification``, hashed as ``model_sha256``).  That document
is what the Pinocchio force adapter loads, so the provider is built from the
session itself: a closure in the specification gives an
``AdapterConstrainedProvider`` (closed-loop counterfactuals, reaction
re-solved), and a session whose specification cannot be loaded fails closed
with a message naming the missing input.  No placeholder inertia exists.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.shared.python.loop_nullspace.provider import (
    AdapterConstrainedProvider,
    ReorderedProvider,
)
from src.shared.python.motion_matching.counterfactual import CounterfactualDynamics

if TYPE_CHECKING:
    from src.shared.python.motion_matching.candidate_session import CandidateSession

__all__ = ["resolve_session_dynamics"]

_REQUIRED_KEYS = ("bodies", "joints", "coordinate_order")
_CACHE: dict[str, CounterfactualDynamics] = {}


def _missing(session: Any) -> list[str]:
    spec = getattr(session, "specification", None)
    if not isinstance(spec, dict):
        return list(_REQUIRED_KEYS)
    return [k for k in _REQUIRED_KEYS if k not in spec]


def resolve_session_dynamics(session: CandidateSession) -> CounterfactualDynamics:
    """Build (and cache by ``model_sha256``) the dynamics for ``session``.

    Preconditions: ``session.specification`` is a full model specification with
    ``bodies``, ``joints`` and a ``coordinate_order`` equal to
    ``session.coordinate_names``, and the Pinocchio adapter can load it.
    Raises ``ValueError`` naming the missing input otherwise.  Postcondition:
    a constrained provider when the specification declares a ``closure``.
    """
    missing = _missing(session)
    if missing:
        raise ValueError(
            "no dynamics provider can be built for this candidate session: its "
            f"model specification is missing {missing}; load the session with "
            "its full model specification (ingest_candidate_session model_path) "
            "or pass dynamics=CounterfactualDynamics(provider=...)"
        )
    spec = session.specification
    if tuple(spec["coordinate_order"]) != tuple(session.coordinate_names):
        raise ValueError(
            "no dynamics provider can be built: the session coordinate order "
            "differs from its model specification coordinate_order"
        )
    key = session.model_sha256
    if key in _CACHE:
        return _CACHE[key]
    try:
        from src.engines.physics_engines.pinocchio.python.force_adapter import (
            PinocchioForceAdapter,
        )

        adapter = PinocchioForceAdapter(json.dumps(spec).encode("utf-8"))
    except ImportError as exc:
        raise ValueError(
            f"no dynamics provider can be built: Pinocchio is unavailable ({exc})"
        ) from exc
    native = tuple(adapter.coordinate_order)
    order = tuple(session.coordinate_names)
    if sorted(native) != sorted(order):
        raise ValueError(
            "no dynamics provider can be built: the adapter and session name "
            "different coordinates"
        )
    if "closure" not in spec:
        raise ValueError(
            "no dynamics provider can be built: the model specification declares "
            "no closure, and no unconstrained provider is available from it"
        )
    provider: Any = AdapterConstrainedProvider(adapter)
    if native != order:
        provider = ReorderedProvider(provider, native, order)
    dynamics = CounterfactualDynamics(provider)
    _CACHE[key] = dynamics
    return dynamics
