"""Versioned source-pixel geometry and reversible edits for coaching references."""

from __future__ import annotations

from math import hypot, isclose
from pathlib import Path
from typing import Generic, Literal, Self, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.rig.documents import write_document

Point = tuple[float, float]
MAX_LAYER_BYTES = 2_000_000
MAX_HISTORY = 100
DEFAULT_DRAWING_COLOUR = "#ffcc33"


class Drawing(BaseModel):
    """Two endpoints for lines; opposite bounding corners for closed shapes."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    kind: Literal["line", "arrow", "circle", "ellipse", "rectangle"]
    start: Point
    end: Point
    colour: str = Field(default=DEFAULT_DRAWING_COLOUR, pattern=r"^#[0-9a-fA-F]{6}$")
    stroke: int = Field(default=3, ge=1, le=40, strict=True)
    visible: bool = True
    first: int = Field(default=0, ge=0, strict=True)
    last: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def valid_geometry(self) -> Self:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        if hypot(dx, dy) < 1:
            raise ValueError("A drawing needs distinct endpoints")
        if (
            self.kind in ("circle", "ellipse", "rectangle")
            and min(abs(dx), abs(dy)) < 1
        ):
            raise ValueError("A closed shape needs width and height")
        if self.kind == "circle" and not isclose(abs(dx), abs(dy), abs_tol=0.01):
            raise ValueError("A circle needs an equal width and height")
        if self.last is not None and self.last < self.first:
            raise ValueError("Visibility ends before it starts")
        return self

    def at(self, frame: int) -> bool:
        return (
            self.visible
            and self.first <= frame
            and (self.last is None or frame <= self.last)
        )

    def changed(self, **values: object) -> Drawing:
        return Drawing.model_validate(self.model_dump() | values)

    def translated(self, dx: float, dy: float) -> Drawing:
        return self.changed(
            start=(self.start[0] + dx, self.start[1] + dy),
            end=(self.end[0] + dx, self.end[1] + dy),
        )

    def hit(self, point: Point, *, tolerance: float) -> bool:
        """Distance to visible stroke, so nested shapes remain independently selectable."""
        x, y = point
        ax, ay = self.start
        bx, by = self.end
        threshold = tolerance + self.stroke / 2
        if self.kind in ("line", "arrow"):
            dx, dy = bx - ax, by - ay
            t = max(
                0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy))
            )
            return hypot(x - ax - t * dx, y - ay - t * dy) <= threshold
        left, right = sorted((ax, bx))
        top, bottom = sorted((ay, by))
        if self.kind == "rectangle":
            return (
                left - threshold <= x <= right + threshold
                and min(abs(y - top), abs(y - bottom)) <= threshold
            ) or (
                top - threshold <= y <= bottom + threshold
                and min(abs(x - left), abs(x - right)) <= threshold
            )
        rx, ry = (right - left) / 2, (bottom - top) / 2
        radius = hypot((x - (left + right) / 2) / rx, (y - (top + bottom) / 2) / ry)
        return abs(radius - 1) * min(rx, ry) <= threshold


class DrawingLayer(BaseModel):
    """One camera's source image and original frame clock; no crop-relative values."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["coaching-drawings/1.0.0"] = "coaching-drawings/1.0.0"
    view: str = Field(min_length=1, max_length=200)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    frames: int = Field(gt=0, strict=True)
    shapes: tuple[Drawing, ...] = Field(default=(), max_length=2000)

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        if len({shape.id for shape in self.shapes}) != len(self.shapes):
            raise ValueError("Drawing IDs must be unique")
        for shape in self.shapes:
            if any(
                not (0 <= x < self.width and 0 <= y < self.height)
                for x, y in (shape.start, shape.end)
            ):
                raise ValueError("Drawing extends outside the source image")
            if shape.first >= self.frames or (
                shape.last is not None and shape.last >= self.frames
            ):
                raise ValueError("Drawing visibility exceeds the source timeline")
        return self

    def with_shapes(self, shapes: tuple[Drawing, ...]) -> DrawingLayer:
        return DrawingLayer.model_validate(self.model_dump() | {"shapes": shapes})

    def with_shape(self, shape: Drawing) -> DrawingLayer:
        shapes = tuple(shape if item.id == shape.id else item for item in self.shapes)
        if not any(item.id == shape.id for item in self.shapes):
            shapes += (shape,)
        return self.with_shapes(shapes)

    def without(self, identity: str) -> DrawingLayer:
        return self.with_shapes(
            tuple(item for item in self.shapes if item.id != identity)
        )

    def save(self, path: Path) -> None:
        write_document(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> DrawingLayer:
        if path.stat().st_size > MAX_LAYER_BYTES:
            raise ValueError("Drawing document is too large")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


Snapshot = TypeVar("Snapshot", bound=BaseModel)


class History(Generic[Snapshot]):
    """Bounded immutable snapshots; one drag creates one undo step."""

    def __init__(self, layer: Snapshot) -> None:
        self.current = layer
        self._past: list[Snapshot] = []
        self._future: list[Snapshot] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    def apply(self, layer: Snapshot) -> None:
        if layer != self.current:
            self._past = (self._past + [self.current])[-MAX_HISTORY:]
            self.current = layer
            self._future.clear()

    def undo(self) -> None:
        if self._past:
            self._future.append(self.current)
            self.current = self._past.pop()

    def redo(self) -> None:
        if self._future:
            self._past.append(self.current)
            self.current = self._future.pop()
