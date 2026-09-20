"""Model variant grouping and presentation projection (ORG-07, #10514).

Pure grouping and presentation projection over cross-engine identity and
provider metadata. Groups models by exercise/model identity first while
retaining all engine-specific assets underneath.

Explicit Contracts:
- Grouping is presentation-only: no asset or provider variant is dropped.
- Engine selection never silently substitutes another engine.
- Unknown/incompatible variants remain labeled.
- Preset parameters survive model-registry conversion.
- Name collisions across different real model identities are never merged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require

_KNOWN_EXERCISES = frozenset(
    {
        "squat",
        "deadlift",
        "bench_press",
        "snatch",
        "clean_and_jerk",
        "gait",
        "sit_to_stand",
    }
)


@dataclass(frozen=True)
class ModelVariant:
    """An engine-specific concrete asset or provider variant."""

    variant_id: str
    provider: str | None
    engine_type: str | None
    path: str
    source_root: str | None = None
    capabilities: tuple[str, ...] = ()
    is_available: bool = True
    diagnostic: str | None = None
    config: Any | None = None


@dataclass(frozen=True)
class LogicalModelIdentity:
    """Canonical cross-engine identity shared by semantically equivalent variants."""

    canonical_id: str
    display_name: str
    exercise: str | None = None
    motion_family: str = "biomechanics"
    humanoid: str = "humanoid"
    dataset: str | None = None


@dataclass
class LogicalModelChoice:
    """A logical choice presenting one user task across multiple engine variants."""

    identity: LogicalModelIdentity
    primary_id: str
    display_name: str
    description: str
    variants: dict[str, ModelVariant] = field(default_factory=dict)
    default_engine: str | None = None

    @property
    def retained_variants_count(self) -> int:
        """Return the number of underlying retained variants."""
        return len(self.variants)

    def get_variant(self, engine_type: str) -> ModelVariant:
        """Retrieve the variant for the requested engine.

        Precondition:
            ``engine_type`` must be a non-empty string.

        Postcondition:
            Returns the exact requested engine variant. Never silently substitutes
            another engine.
        """
        require(
            isinstance(engine_type, str) and bool(engine_type.strip()),
            "engine_type must be a non-empty string",
            engine_type,
        )
        key = engine_type.strip().lower()
        if key not in self.variants:
            available = list(self.variants.keys())
            raise KeyError(
                f"Engine variant {engine_type!r} not available for "
                f"'{self.identity.canonical_id}'. Available engines: {available}"
            )
        return self.variants[key]


def _extract_exercise_from_model(model: Any) -> str | None:
    """Infer exercise name from model attributes, capabilities, or ID."""
    explicit_ex = getattr(model, "exercise", None)
    if isinstance(explicit_ex, str) and explicit_ex.strip():
        return explicit_ex.strip().lower()

    if isinstance(model, dict):
        ex_val = model.get("exercise")
        if isinstance(ex_val, str) and ex_val.strip():
            return ex_val.strip().lower()

    model_id = getattr(model, "id", None) or (
        model.get("id") if isinstance(model, dict) else ""
    )
    if isinstance(model_id, str):
        for known_ex in _KNOWN_EXERCISES:
            if (
                model_id == known_ex
                or model_id.endswith(f"-{known_ex}")
                or f"_{known_ex}" in model_id
            ):
                return known_ex

    capabilities = getattr(model, "capabilities", ()) or (
        model.get("capabilities", ()) if isinstance(model, dict) else ()
    )
    for cap in capabilities:
        cap_clean = str(cap).strip().lower()
        if cap_clean in _KNOWN_EXERCISES:
            return cap_clean

    return None


def _extract_model_identity(model: Any) -> LogicalModelIdentity:
    """Extract or infer the canonical logical identity of a model."""
    raw_identity = getattr(model, "identity", None) or (
        model.get("identity") if isinstance(model, dict) else None
    )

    if raw_identity is not None:
        canonical_id = getattr(raw_identity, "canonical_id", None) or (
            raw_identity.get("canonical_id") if isinstance(raw_identity, dict) else None
        )
        if canonical_id:
            display_name = getattr(model, "name", None) or (
                model.get("name") if isinstance(model, dict) else canonical_id
            )
            motion_family = getattr(raw_identity, "motion_family", "biomechanics") or (
                raw_identity.get("motion_family", "biomechanics")
                if isinstance(raw_identity, dict)
                else "biomechanics"
            )
            exercise = getattr(raw_identity, "exercise", None) or (
                raw_identity.get("exercise") if isinstance(raw_identity, dict) else None
            )
            humanoid = getattr(raw_identity, "humanoid", "humanoid") or (
                raw_identity.get("humanoid", "humanoid")
                if isinstance(raw_identity, dict)
                else "humanoid"
            )
            dataset = getattr(raw_identity, "dataset", None) or (
                raw_identity.get("dataset") if isinstance(raw_identity, dict) else None
            )
            return LogicalModelIdentity(
                canonical_id=str(canonical_id),
                display_name=str(display_name),
                exercise=str(exercise) if exercise else None,
                motion_family=str(motion_family),
                humanoid=str(humanoid),
                dataset=str(dataset) if dataset else None,
            )

    # Infer from exercise if present
    exercise = _extract_exercise_from_model(model)
    if exercise:
        name = getattr(model, "name", None) or (
            model.get("name") if isinstance(model, dict) else None
        )
        display_name = name or exercise.replace("_", " ").title()
        return LogicalModelIdentity(
            canonical_id=f"biomech.exercise.{exercise}",
            display_name=str(display_name),
            exercise=exercise,
            motion_family="biomechanics",
            humanoid="humanoid",
        )

    # Standalone identity using model ID
    model_id = str(
        getattr(model, "id", None)
        or (model.get("id") if isinstance(model, dict) else "unknown")
    )
    name = getattr(model, "name", None) or (
        model.get("name") if isinstance(model, dict) else model_id
    )
    return LogicalModelIdentity(
        canonical_id=f"model.{model_id}",
        display_name=str(name),
        motion_family="standalone",
        humanoid="default",
    )


def _build_variant_from_model(model: Any) -> ModelVariant:
    """Build a ModelVariant from a model object or dictionary."""
    model_id = str(
        getattr(model, "id", None)
        or (model.get("id") if isinstance(model, dict) else "")
    )
    provider = getattr(model, "provider", None) or (
        model.get("provider") if isinstance(model, dict) else None
    )
    engine_type = getattr(model, "engine_type", None) or (
        model.get("engine_type") if isinstance(model, dict) else None
    )
    path = str(
        getattr(model, "path", None)
        or (model.get("path") if isinstance(model, dict) else "")
    )
    source_root = getattr(model, "source_root", None) or (
        model.get("source_root") if isinstance(model, dict) else None
    )
    capabilities = tuple(
        getattr(model, "capabilities", ())
        or (model.get("capabilities", ()) if isinstance(model, dict) else ())
    )

    # Infer engine_type if missing from provider
    if not engine_type and provider:
        p_clean = provider.lower()
        if "mujoco" in p_clean:
            engine_type = "mujoco"
        elif "drake" in p_clean:
            engine_type = "drake"
        elif "pinocchio" in p_clean:
            engine_type = "pinocchio"
        elif "opensim" in p_clean:
            engine_type = "opensim"
        elif "jaxsim" in p_clean:
            engine_type = "jaxsim"

    return ModelVariant(
        variant_id=model_id,
        provider=provider,
        engine_type=engine_type,
        path=path,
        source_root=source_root,
        capabilities=capabilities,
        config=model,
    )


class ModelGroupingProjection:
    """Pure grouping/presentation projection over cross-engine identity metadata."""

    def group_models(self, models: Iterable[Any]) -> list[LogicalModelChoice]:
        """Group models into logical choices by real model identity.

        Retains all provider and engine-specific asset variants underneath.
        Name collisions across different real identities remain separate choices.
        """
        grouped: dict[str, list[tuple[LogicalModelIdentity, ModelVariant, Any]]] = {}

        for model in models:
            identity = _extract_model_identity(model)
            variant = _build_variant_from_model(model)
            grouped.setdefault(identity.canonical_id, []).append(
                (identity, variant, model)
            )

        choices: list[LogicalModelChoice] = []
        for _canonical_id, items in sorted(grouped.items()):
            first_identity, _, first_model = items[0]
            display_name = first_identity.display_name
            description = str(
                getattr(first_model, "description", None)
                or (
                    first_model.get("description")
                    if isinstance(first_model, dict)
                    else ""
                )
                or f"Cross-engine {display_name} model"
            )

            variants_dict: dict[str, ModelVariant] = {}
            for _, variant, _ in items:
                engine_key = (
                    (variant.engine_type or variant.provider or variant.variant_id)
                    .strip()
                    .lower()
                )
                variants_dict[engine_key] = variant

            choice = LogicalModelChoice(
                identity=first_identity,
                primary_id=first_identity.canonical_id,
                display_name=display_name,
                description=description,
                variants=variants_dict,
                default_engine=next(iter(variants_dict.keys()), None),
            )
            choices.append(choice)

        return choices


def resolve_shortcut(shortcut_id: str) -> tuple[str, dict[str, Any]]:
    """Resolve legacy IDs and shortcuts to their canonical destination and parameters.

    Contracts:
    - ``biomech_sit_to_stand`` resolves to ``exercise="sit_to_stand"`` (never falls back to gait).
    - ``biomech_gait`` resolves to ``exercise="gait"``.
    - Engine dashboards resolve to advanced modes of their engine destination.
    - ``movement_optimizer`` and ``tools_movement_optimizer`` map to one task with authority info.
    """
    require(
        isinstance(shortcut_id, str) and bool(shortcut_id.strip()),
        "shortcut_id must be a non-empty string",
        shortcut_id,
    )
    s_id = shortcut_id.strip()

    if s_id == "biomech_sit_to_stand":
        return "biomech_exercise", {"exercise": "sit_to_stand"}
    if s_id == "biomech_gait":
        return "biomech_exercise", {"exercise": "gait"}

    if s_id == "mujoco_dashboard":
        return "mujoco_unified", {"mode": "dashboard", "engine": "mujoco"}
    if s_id == "drake_dashboard":
        return "drake_golf", {"mode": "dashboard", "engine": "drake"}
    if s_id == "pinocchio_dashboard":
        return "pinocchio_golf", {"mode": "dashboard", "engine": "pinocchio"}

    if s_id in {"movement_optimizer", "tools_movement_optimizer"}:
        try:
            from src.shared.python.config.tools_vendor_authority import (
                inspect_tools_vendor_authority,
            )

            repo_root = Path(__file__).resolve().parents[4]
            authority = inspect_tools_vendor_authority(repo_root)
            auth_str = "tools" if authority.available else "sibling"
        except Exception:
            auth_str = "tools"

        return "movement_optimizer", {
            "authority": auth_str,
            "provider_id": s_id,
            "legacy_id": s_id,
        }

    return s_id, {}


def get_missing_checkout_diagnostic(
    repo_name: str, expected_path: Path | None = None
) -> str:
    """Return an explicit, actionable diagnostic message for a missing sibling repo checkout."""
    require(
        isinstance(repo_name, str) and bool(repo_name.strip()),
        "repo_name must be a non-empty string",
        repo_name,
    )
    clean_name = repo_name.strip()
    if expected_path is not None:
        path_hint = f" at '{expected_path}'"
    else:
        path_hint = " beside UpstreamDrift"

    return (
        f"Sibling repository '{clean_name}' is not checked out{path_hint}. "
        f"Direct Models/Integrations access requires cloning or checking out '{clean_name}' "
        f"in the workspace."
    )
