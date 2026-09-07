"""The rig plan: which camera plays which view, and how it is captured.

A plan is the operator's declaration of an experimental condition. It binds
named views to camera *identities* (USB serial, or the port-path fallback for
units without one — never an OpenCV index, which reshuffles on replug) and to
a capture mode plus control overrides. Plans are plain JSON so they can be
versioned alongside a session, diffed between conditions, and checked against
the live USB topology before anything is mounted.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.shared.python.core.contracts import require

from .topology import CameraLocation, conflicting_identities

PLAN_SCHEMA_VERSION = "rig-plan/1.0.0"


class CaptureMode(BaseModel):
    """Negotiable stream parameters."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1200, gt=0)
    fps: int = Field(default=60, gt=0)
    fourcc: str = Field(default="MJPG", min_length=4, max_length=4)

    @field_validator("fourcc")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class CameraControls(BaseModel):
    """Optional UVC control overrides; ``None`` leaves the camera default."""

    model_config = ConfigDict(frozen=True)

    exposure: float | None = None
    gain: float | None = None
    auto_exposure: bool | None = None

    def as_overrides(self) -> dict[str, float]:
        """Non-``None`` controls as ``{name: value}`` (bools become 0/1)."""
        items = {
            "exposure": self.exposure,
            "gain": self.gain,
            "auto_exposure": None
            if self.auto_exposure is None
            else float(self.auto_exposure),
        }
        return {k: float(v) for k, v in items.items() if v is not None}


class CameraBinding(BaseModel):
    """One view of the rig bound to one physical camera."""

    model_config = ConfigDict(frozen=True)

    view: str = Field(min_length=1)
    serial: str | None = None
    port_path: str | None = None
    mode: CaptureMode = Field(default_factory=CaptureMode)
    controls: CameraControls = Field(default_factory=CameraControls)

    @model_validator(mode="after")
    def _needs_identity(self) -> CameraBinding:
        if not (self.serial or self.port_path):
            raise ValueError(f"view {self.view!r} needs a serial or a port_path")
        return self

    @property
    def identity(self) -> str:
        """Serial when present, else the port-path identity."""
        return self.serial or str(self.port_path)


class RigPlan(BaseModel):
    """A named set of camera bindings for one experimental condition."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = PLAN_SCHEMA_VERSION
    name: str = Field(min_length=1)
    cameras: tuple[CameraBinding, ...] = Field(min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def _unique(self) -> RigPlan:
        views = [c.view for c in self.cameras]
        idents = [c.identity for c in self.cameras]
        if len(set(views)) != len(views):
            raise ValueError(f"duplicate views in plan {self.name!r}: {views}")
        if len(set(idents)) != len(idents):
            raise ValueError(f"duplicate camera identities in plan {self.name!r}")
        return self

    def binding_for(self, identity: str) -> CameraBinding | None:
        return next((c for c in self.cameras if c.identity == identity), None)

    def save(self, path: Path) -> None:
        """Write the plan as indented JSON (creates parent directories)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RigPlan:
        """Read a plan; raises ``ValueError`` for an incompatible schema version."""
        require(path.is_file(), "plan file must exist", str(path))
        plan = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if plan.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"plan schema {plan.schema_version!r} is not {PLAN_SCHEMA_VERSION!r}"
            )
        return plan


class PlanCheck(BaseModel):
    """Result of matching a plan against the live topology."""

    model_config = ConfigDict(frozen=True)

    matched: dict[str, str]  # view -> camera instance id
    missing: tuple[str, ...]  # views whose camera was not enumerated
    conflicts: tuple[str, ...]  # views sharing a USB 2.0 bandwidth domain
    unplanned: tuple[str, ...]  # enumerated identities no view claims

    @property
    def ok(self) -> bool:
        return not self.missing and not self.conflicts


def check_plan(plan: RigPlan, cams: Iterable[CameraLocation]) -> PlanCheck:
    """Match plan bindings to enumerated cameras and flag bus conflicts.

    Postcondition: every view appears in exactly one of ``matched``/``missing``.
    """
    located = list(cams)
    by_identity = {c.identity: c for c in located}
    matched: dict[str, str] = {}
    missing: list[str] = []
    for binding in plan.cameras:
        cam = by_identity.get(binding.identity)
        if cam is None:
            missing.append(binding.view)
        else:
            matched[binding.view] = cam.camera
    planned = {b.identity for b in plan.cameras}
    conflicted = set(
        conflicting_identities([c for c in located if c.identity in planned])
    )
    conflicts = tuple(b.view for b in plan.cameras if b.identity in conflicted)
    unplanned = tuple(sorted(i for i in by_identity if i not in planned))
    return PlanCheck(
        matched=matched,
        missing=tuple(missing),
        conflicts=conflicts,
        unplanned=unplanned,
    )
