"""Non-destructive swing selections in original frame/pixel coordinates (#9860).

Inference crops the pixels, then translates detections back to the camera's
original image. Frame clocks and intrinsics consequently remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bundle import RecordingEntry, load_bundle

EDITS_FILE = "swing_edits.json"
MAX_RECIPE_BYTES = 128_000


class CropRect(BaseModel):
    """Integer source pixels, with an exclusive right/bottom edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)

    def validate_size(self, width: int, height: int) -> None:
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError("Crop extends outside the original image")


class ViewEdit(BaseModel):
    """Inclusive source-frame bounds; ``last=None`` keeps the remaining frames."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    first: int = Field(default=0, ge=0, strict=True)
    last: int | None = Field(default=None, ge=0, strict=True)
    crop: CropRect | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.last is not None and self.last < self.first:
            raise ValueError("Last frame must be at or after the first frame")
        return self

    def validate_recording(self, entry: RecordingEntry) -> None:
        if entry.frames is not None and (
            self.first >= entry.frames
            or (self.last is not None and self.last >= entry.frames)
        ):
            raise ValueError(f"Selection exceeds the recording's {entry.frames} frames")
        if self.crop:
            self.crop.validate_size(
                entry.width or entry.requested_mode.width,
                entry.height or entry.requested_mode.height,
            )


class SessionEdits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["swing-edits/1.0.0"] = "swing-edits/1.0.0"
    views: dict[str, ViewEdit] = Field(default_factory=dict)


def load_edits(root: Path) -> SessionEdits:
    """Missing means unedited; malformed or unsupported recipes fail visibly."""
    path = root / EDITS_FILE
    if not path.exists():
        return SessionEdits()
    if path.stat().st_size > MAX_RECIPE_BYTES:
        raise ValueError("Swing edit recipe is too large")
    recipe = SessionEdits.model_validate_json(path.read_text(encoding="utf-8"))
    validate_edits(root, recipe)
    return recipe


def validate_edits(root: Path, recipe: SessionEdits) -> None:
    entries = {entry.view: entry for entry in load_bundle(root)[1].recordings}
    for view, edit in recipe.views.items():
        if view not in entries:
            raise ValueError(f"Unknown recording view: {view}")
        edit.validate_recording(entries[view])


def has_analysis(root: Path) -> bool:
    """Derived results lock the recipe; an editable copy starts a new lineage."""
    return any(root.glob("observations*")) or any(
        (root / name).exists()
        for name in ("reconstruct", "analysis_2d", "variants", "model", "export")
    )


def save_edits(root: Path, recipe: SessionEdits) -> None:
    validate_edits(root, recipe)
    if has_analysis(root):
        raise ValueError("Create an editable copy before changing an analyzed capture")
    target = root / EDITS_FILE
    # Same-directory replace leaves the previous recipe intact on write failure.
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=root,
            prefix=".swing-edits-",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(recipe.model_dump_json(indent=2) + "\n")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
