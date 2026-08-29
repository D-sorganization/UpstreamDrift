"""Identity context for plots and exports (issue #8828).

A PNG saved from the MuJoCo dashboard used to look visually identical to
one saved from Drake: every plot title was a bare metric name ("Joint
Positions", "System Energy", ...) with no indication of which physics
engine, model, or run produced it.

``PlotIdentity`` is a small, optional value object that carries whatever
identity information is genuinely available at a given call site (engine
name, model name, run ID) so it can be rendered as a figure footer and
embedded in export metadata. Fields that are not known are left ``None``
and simply omitted from the rendered label / metadata rather than being
fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "PlotIdentity",
    "apply_identity_footer",
    "resolve_and_apply_identity_footer",
]


@dataclass(frozen=True)
class PlotIdentity:
    """Optional engine/model/run identity attached to a plot or export.

    Attributes:
        engine: Physics engine name (e.g. ``"mujoco"``, ``"drake"``),
            when known.
        model: Model name loaded into the engine, when known.
        run_id: Identifier for the recording/run that produced the data,
            when known.
    """

    engine: str | None = None
    model: str | None = None
    run_id: str | None = None

    @classmethod
    def from_recorder(cls, recorder: Any, run_id: str | None = None) -> PlotIdentity:
        """Derive identity from a ``RecorderInterface``-like object.

        Only reads attributes that are genuinely present on the recorder's
        ``engine`` (``engine_type`` / ``model_name``, per the
        ``PhysicsEngine`` protocol's ``Checkpointable`` and ``Loadable``
        sub-protocols). Missing attributes are left ``None`` rather than
        guessed.

        Args:
            recorder: Object exposing an optional ``.engine`` attribute.
            run_id: Explicit run identifier, if known by the caller.

        Returns:
            A ``PlotIdentity`` populated with whatever was discoverable.
        """
        engine_obj = getattr(recorder, "engine", None)
        engine_name: str | None = None
        model_name: str | None = None

        if engine_obj is not None:
            raw_engine_type = getattr(engine_obj, "engine_type", None)
            if raw_engine_type is not None:
                # EngineType is an Enum with a lowercase .value; fall back
                # to str() for anything else that already looks stringy.
                engine_name = str(getattr(raw_engine_type, "value", raw_engine_type))

            raw_model_name = getattr(engine_obj, "model_name", None)
            if isinstance(raw_model_name, str) and raw_model_name:
                model_name = raw_model_name

        return cls(engine=engine_name, model=model_name, run_id=run_id)

    def is_empty(self) -> bool:
        """Return True when no identity field is populated."""
        return self.engine is None and self.model is None and self.run_id is None

    def label(self) -> str | None:
        """Render a short human-readable label, or None if nothing is known."""
        parts: list[str] = []
        if self.engine:
            parts.append(f"Engine: {self.engine}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.run_id:
            parts.append(f"Run: {self.run_id}")
        return " | ".join(parts) if parts else None

    def as_metadata_dict(self) -> dict[str, str]:
        """Return identity fields as a flat string dict for export metadata."""
        meta: dict[str, str] = {}
        if self.engine:
            meta["engine"] = self.engine
        if self.model:
            meta["model"] = self.model
        if self.run_id:
            meta["run_id"] = self.run_id
        return meta


def apply_identity_footer(fig: Figure, identity: PlotIdentity | None) -> None:
    """Render ``identity`` as a small footer on ``fig``, if any is known.

    No-op when ``identity`` is ``None`` or carries no populated fields, so
    callers can call this unconditionally without fabricating placeholder
    text for unknown engines/models/runs.
    """
    if identity is None:
        return
    label = identity.label()
    if not label:
        return
    fig.text(
        0.99,
        0.01,
        label,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
        alpha=0.85,
    )


def resolve_and_apply_identity_footer(
    fig: Figure, recorder: Any, identity: PlotIdentity | None
) -> PlotIdentity:
    """Resolve ``identity`` (explicit, or derived from ``recorder``) and render it.

    DRY helper shared by every ``plot_*`` function in ``kinematics`` and
    ``energy``: eliminates the repeated
    ``identity if identity is not None else PlotIdentity.from_recorder(recorder)``
    pattern at each call site.

    Returns:
        The resolved ``PlotIdentity``, for callers that need it again later
        (e.g. functions that create the figure lazily after the early
        "no data" return).
    """
    resolved = (
        identity if identity is not None else PlotIdentity.from_recorder(recorder)
    )
    apply_identity_footer(fig, resolved)
    return resolved
