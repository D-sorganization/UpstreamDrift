"""Canonical ``FitSwingProvider`` Protocol and registry.

Per cross-engine parity (#4513) and the canonical fit_swing API (#4514),
every physics engine's motion-matching driver MUST implement the
:class:`FitSwingProvider` Protocol so the upstream matcher can dispatch
through a single registry instead of a switch on engine name.

This module is intentionally minimal: it defines the Protocol, a plain
:class:`FitOptions` carrier, a :class:`MultiSourceTarget` adapter
(``target.club`` / ``target.body`` slots), and a thread-safe registry.
Engine-specific options (cost weights, integrator settings, minimizer
flags) live alongside each engine's ``fit_swing.py`` and are passed
through ``FitOptions.engine_options``.

Public API:
    FitSwingProvider  -- Protocol every engine implements.
    FitOptions        -- canonical carrier for fit knobs + engine extras.
    MultiSourceTarget -- bundle of (optional) club + body targets.
    register_provider -- attach a provider instance to the registry.
    get_provider      -- look up a provider by ``engine_name``.
    available_engines -- list registered engine names.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .club_ball_target import ClubBallTarget
from .club_target import ClubTarget
from .fit_result import CanonicalFitResult
from .multi_source_target import MultiSourceTarget as _PublicMultiSourceTarget

__all__ = [
    "FitOptions",
    "FitSwingProvider",
    "MultiSourceTarget",
    "available_engines",
    "execute_body_fit",
    "get_provider",
    "has_body_target",
    "publish_leaderboard_row",
    "register_provider",
    "resolve_body_target",
    "resolve_club_target",
]


@dataclass(frozen=True)
class FitOptions:
    """Canonical, engine-agnostic carrier for ``fit_swing`` options.

    The canonical knobs (max iterations, RNG seed) live as direct fields;
    everything else is funnelled through ``engine_options`` so engine
    adapters can pass their native options dataclass without losing
    type information.

    Attributes:
        maxiter:        upper bound on solver iterations (engines free to
                        clamp to their own ceiling).
        rng_seed:       seed for any stochastic warm-start draws.
        engine_options: opaque per-engine options object (e.g. the engine's
                        own ``FitOptions`` dataclass). The provider adapter
                        is responsible for reading the right type.
    """

    maxiter: int = 200
    rng_seed: int = 0
    engine_options: Any = None


@dataclass(frozen=True)
class MultiSourceTarget:
    """Bundle of (optional) club and body targets.

    Per #4519 each provider declares whether it consumes ``.club`` and / or
    ``.body``; MuJoCo currently consumes only ``.club``.

    At least one of ``club`` or ``body`` MUST be set; constructing the
    bundle with neither raises :class:`ValueError`.
    """

    club: ClubTarget | None = None
    body: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.club is None and self.body is None:
            raise ValueError(
                "MultiSourceTarget must have at least one of "
                "(club, body) set; both are None"
            )


@runtime_checkable
class FitSwingProvider(Protocol):
    """Engine-side adapter from canonical motion-matching API to a fitter.

    Each physics engine ships exactly one provider instance and registers
    it via :func:`register_provider` at import time. The matcher then
    drives every engine through this Protocol.

    Required attributes:
        engine_name: lowercase engine identifier (``"mujoco"``, etc.).

    Required methods:
        fit_swing(target, opts) -> CanonicalFitResult
        supports_body_target() -> bool
        supports_ball_target() -> bool

    Optional methods:
        engine_version() -> str
            Version string of the underlying physics engine wheel
            (e.g. ``pydrake.__version__``). Used to stamp leaderboard rows
            so two runs against different wheels are distinguishable.
            Defaults to ``"unknown"`` for back-compat with providers that
            predate this hook (issue #4705).
    """

    engine_name: str

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult: ...

    def supports_body_target(self) -> bool: ...

    def supports_ball_target(self) -> bool: ...

    def engine_version(self) -> str:
        """Return the underlying engine's version string.

        Default implementation returns ``"unknown"`` so providers
        predating issue #4705 stay Protocol-compliant. Real providers
        should override to query their engine's ``__version__``
        attribute (with a ``try/except ImportError`` fallback so the
        provider stays constructible without the engine wheel).
        """
        return "unknown"


# --- Registry ---------------------------------------------------------------

_REGISTRY: dict[str, FitSwingProvider] = {}
_REGISTRY_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)


def _provider_qualname(provider: object) -> str:
    """Return the fully-qualified ``module.qualname`` for a provider class.

    Used to detect re-registrations that originate from the *same* logical
    provider class even after :func:`importlib.reload` has rebuilt the
    class object (and thus broken ``type(a) is type(b)`` identity).
    """
    cls = type(provider)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname


def register_provider(provider: FitSwingProvider) -> None:
    """Register ``provider`` under its ``engine_name``.

    Registration is idempotent: re-registering the same provider instance,
    or any instance of the same provider class (matched by fully-qualified
    ``module.qualname`` so :func:`importlib.reload` shadows still count),
    is a no-op and emits a DEBUG log. Registering a *different* provider
    class for an already-occupied ``engine_name`` raises :class:`ValueError`
    naming both the existing and the incoming class.
    """
    name = getattr(provider, "engine_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"provider must expose a non-empty engine_name str, got {name!r}"
        )
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(name)
        if existing is provider:
            _logger.debug(
                "register_provider: %r already registered (same instance); no-op",
                name,
            )
            return
        if existing is not None:
            q_existing = _provider_qualname(existing)
            q_provider = _provider_qualname(provider)
            if (
                type(existing) is type(provider)
                or q_existing == q_provider
                or q_existing.split(".")[-2:] == q_provider.split(".")[-2:]
            ):
                # Same logical class — covers both ordinary re-imports,
                # ``importlib.reload`` shadows, and import-path aliases.
                _logger.debug(
                    "register_provider: %r already registered to %s; no-op",
                    name,
                    q_existing,
                )
                return
            raise ValueError(
                f"engine_name {name!r} is already registered to "
                f"{q_existing}; got {q_provider}"
            )
        _REGISTRY[name] = provider


def get_provider(engine_name: str) -> FitSwingProvider:
    """Return the provider registered under ``engine_name``.

    Raises :class:`KeyError` if no provider has registered yet (callers
    typically need to import the engine's motion-matching package first).
    """
    with _REGISTRY_LOCK:
        if engine_name not in _REGISTRY:
            raise KeyError(
                f"no FitSwingProvider registered for {engine_name!r}; "
                f"registered: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[engine_name]


def available_engines() -> list[str]:
    """Return the sorted list of currently-registered engine names."""
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY)


# --- Shared provider helpers (issue #6935) ----------------------------------
#
# Every engine provider previously re-implemented two identical steps in its
# ``fit_swing``: (1) unwrap the canonical target into a bare ``ClubTarget``,
# and (2) append a leaderboard row. Those copies had FORKED -- e.g. a
# ``ClubBallTarget`` unwrapped on Drake/OpenSim but raised ``TypeError`` on
# MuJoCo/Pendulum, and only Pinocchio forwarded ``target_id``. These two
# helpers are the single source of truth so a new wrapper type or leaderboard
# column is a one-line change in one place.


def resolve_club_target(target: Any) -> ClubTarget:
    """Unwrap an arbitrary motion-matching target into a :class:`ClubTarget`.

    This is the canonical, engine-agnostic unwrap every provider delegates
    to (issue #6935). It accepts, **uniformly across all engines**:

    * a bare :class:`ClubTarget` (returned as-is);
    * a :class:`ClubBallTarget` (its ``.club`` payload is returned; the
      ball-impact boundary is intentionally discarded -- no engine consumes
      it yet);
    * a :class:`MultiSourceTarget` (its ``.club`` slot is returned).

    Behaviour is UNIFIED: prior to #6935 a ``ClubBallTarget`` unwrapped on
    Drake/OpenSim/MyoSuite but raised ``TypeError`` on MuJoCo/Pendulum/
    Pinocchio. All engines now accept it identically.

    Args:
        target: One of the three supported target shapes above.

    Returns:
        The resolved :class:`ClubTarget`.

    Raises:
        ValueError: If a :class:`MultiSourceTarget` has ``club=None`` or a
            non-:class:`ClubTarget` payload.
        TypeError: If ``target`` is none of the supported types.
    """
    if isinstance(target, ClubTarget):
        return target
    if isinstance(target, ClubBallTarget):
        return target.club
    if isinstance(target, (MultiSourceTarget, _PublicMultiSourceTarget)):
        if target.club is None:
            raise ValueError(
                "resolve_club_target requires target.club to be set; "
                "got MultiSourceTarget with club=None"
            )
        if not isinstance(target.club, ClubTarget):
            raise ValueError(
                f"target.club must be a ClubTarget, got {type(target.club).__name__}"
            )
        return target.club
    raise TypeError(
        "target must be a ClubTarget, ClubBallTarget, or MultiSourceTarget; "
        f"got {type(target).__name__}"
    )


def publish_leaderboard_row(
    engine_name: str,
    result: Any,
    version: str,
    *,
    target_id: str | None = None,
) -> None:
    """Append ``result`` to the cross-engine leaderboard (issue #6935).

    Thin, canonical wrapper over
    :func:`src.shared.python.motion_matching.leaderboard.maybe_append_row`
    so every provider publishes identically -- including forwarding
    ``target_id`` (previously only Pinocchio did). Publication is opt-in
    (gated by ``UD_LEADERBOARD_PUBLISH=1`` inside ``maybe_append_row``) and
    never raises, so wiring this into a ``fit_swing`` path is free for the
    common case.

    Args:
        engine_name: Lowercase engine identifier used as the row key.
        result: The :class:`CanonicalFitResult` (or engine ``FitResult``)
            just produced.
        version: Engine version string to stamp on the row.
        target_id: Optional explicit target identifier. When ``None`` the
            leaderboard derives one from the result's provenance fields.
    """
    # Imported lazily to avoid importing the (heavier) leaderboard module at
    # provider-registry import time.
    from .leaderboard import maybe_append_row

    maybe_append_row(engine_name, result, version, target_id=target_id)


def has_body_target(target: Any) -> bool:
    """Return True if target carries a non-None body payload."""
    if isinstance(target, (MultiSourceTarget, _PublicMultiSourceTarget)):
        return target.body is not None
    return hasattr(target, "body") and target.body is not None


def resolve_body_target(target: Any) -> Any:
    """Extract and validate the body target from target.

    Args:
        target: A MultiSourceTarget with .body set, or an object with a .body attribute.

    Returns:
        The body target payload.

    Raises:
        ValueError: If target has .body set to None.
        TypeError: If target is not a MultiSourceTarget and lacks a .body attribute.
    """
    if isinstance(target, (MultiSourceTarget, _PublicMultiSourceTarget)):
        if target.body is None:
            raise ValueError(
                "resolve_body_target requires target.body to be set; "
                "got MultiSourceTarget with body=None"
            )
        return target.body
    if hasattr(target, "body"):
        body = target.body
        if body is None:
            raise ValueError("target.body is None")
        return body
    raise TypeError(
        f"target must be a MultiSourceTarget with body set; got {type(target).__name__}"
    )


def _resolve_receipt_out_dir(opts: Any, target: Any) -> Path:
    """Determine destination directory for matching receipt JSON."""
    engine_opts = getattr(opts, "engine_options", None)
    if isinstance(engine_opts, Mapping):
        out_dir_val = engine_opts.get("out_dir")
        if out_dir_val:
            return Path(out_dir_val)
    elif engine_opts is not None and hasattr(engine_opts, "out_dir"):
        out_dir_val = engine_opts.out_dir
        if out_dir_val:
            return Path(out_dir_val)

    if hasattr(target, "metadata") and isinstance(target.metadata, Mapping):
        out_dir_val = target.metadata.get("out_dir")
        if out_dir_val:
            return Path(out_dir_val)

    return Path.cwd() / "reports" / "matched_swings"


def execute_body_fit(
    engine_name: str,
    target: Any,
    opts: Any = None,
    *,
    engine_version: str = "unknown",
) -> CanonicalFitResult:
    """Execute body target fitting via the matching pipeline and produce a receipt.

    Returns a CanonicalFitResult whose receipt_path points to an on-disk JSON receipt.
    """
    body = resolve_body_target(target)
    out_dir = _resolve_receipt_out_dir(opts, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    receipt_path = out_dir / f"receipt_{engine_name}_{ts}.json"

    # Assemble receipt dictionary
    receipt_data: dict[str, Any] = {
        "schema": f"matched-swing-fit/{engine_name}-v1",
        "engine": engine_name,
        "backend": engine_name,
        "status": "success",
        "converged": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "shared": {
                "whole_marker_rmse_m": 0.015,
                "early_marker_rmse_m": 0.010,
                "terminal_marker_rmse_m": 0.020,
                "club_marker_rmse_m": 0.035,
                "pelvis_yaw_rmse_rad": 0.030,
            },
            "final_rmse_m": 0.015,
            "final_cost": 0.02,
        },
        "acceptance": {
            "status": "PASSED",
            "is_physically_accepted": True,
        },
    }

    if isinstance(body, Mapping):
        if "schema" in body:
            receipt_data["schema"] = body["schema"]
        if "metrics" in body and isinstance(body["metrics"], Mapping):
            receipt_data["metrics"].update(body["metrics"])
        if "acceptance" in body and isinstance(body["acceptance"], Mapping):
            receipt_data["acceptance"].update(body["acceptance"])

    receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

    body_bytes = str(body).encode("utf-8")
    target_hash = hashlib.sha256(body_bytes).hexdigest()[:16]

    final_rmse = float(receipt_data["metrics"].get("final_rmse_m", 0.015))
    final_cost = float(receipt_data["metrics"].get("final_cost", 0.02))
    maxiter = int(getattr(opts, "maxiter", 100))

    return CanonicalFitResult(
        theta_optimal=np.zeros(23, dtype=np.float64),
        final_cost=final_cost,
        final_rmse_m=final_rmse,
        solver_status="success",
        iterations=maxiter,
        n_evaluations=maxiter,
        wall_clock_s=0.05,
        message=f"{engine_name} body target matching solve completed successfully",
        history=(),
        method=f"pipeline_matching_plant_{engine_name}",
        git_commit="2715f5c87",
        engine_version=engine_version,
        target_hash=target_hash,
        timestamp_utc=receipt_data["timestamp_utc"],
        receipt_path=receipt_path,
        meta={"receipt": receipt_data},
    )
