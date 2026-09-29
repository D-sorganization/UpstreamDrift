"""Session-local coaching sidecars with path-safe view identity."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .drawing import DrawingLayer


def layer_path(root: Path, view: str) -> Path:
    """Camera labels are data, never filesystem paths."""
    return root / "coaching" / f"{sha256(view.encode('utf-8')).hexdigest()}.json"


def load_layer(
    root: Path, view: str, width: int, height: int, frames: int
) -> DrawingLayer:
    expected = DrawingLayer(view=view, width=width, height=height, frames=frames)
    path = layer_path(root, view)
    if not path.exists():
        return expected
    layer = DrawingLayer.load(path)
    if layer.with_shapes(()) != expected:
        raise ValueError("Saved drawings do not match this camera's original recording")
    return layer


def save_layer(root: Path, layer: DrawingLayer) -> None:
    path = layer_path(root, layer.view)
    path.parent.mkdir(exist_ok=True)
    layer.save(path)


def copy_layers(root: Path, destination: Path, views: tuple[str, ...]) -> None:
    """Retain visual references when a capture starts a new analysis lineage."""
    for view in views:
        path = layer_path(root, view)
        if path.is_file():
            layer = DrawingLayer.load(path)
            if layer.view != view:
                raise ValueError(
                    "Drawing document is assigned to the wrong camera view"
                )
            save_layer(destination, layer)
