"""Resolve named body-part shapes to :class:`MeshShape` instances.

The bundled default library lives under ``assets/body_part_shapes/default/``
and is described by ``manifest.json`` (see that file for the schema).

Typical usage::

    from src.shared.python.body_part_viz.asset_library import ShapeLibrary

    lib = ShapeLibrary.default()
    head = lib.get("head")            # -> MeshShape
    binding = lib.binding_template("head")  # -> MarkerBinding
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bindings import BindingKind, MarkerBinding
from .shapes import MeshShape

__all__ = ["ShapeLibrary"]

_SCHEMA_VERSION = 1


def _default_asset_root() -> Path:
    """Return the on-disk location of the bundled ``default/`` library.

    The repo layout is::

        <repo-root>/assets/body_part_shapes/default/
        <repo-root>/src/shared/python/body_part_viz/asset_library.py

    so we walk five parents up from this file.
    """
    return (
        Path(__file__).resolve().parents[4] / "assets" / "body_part_shapes" / "default"
    )


class ShapeLibrary:
    """Resolve named body-part shapes to :class:`MeshShape` instances.

    Loaded :class:`MeshShape` objects are cached per name; a second call to
    :meth:`get` returns the same instance.
    """

    def __init__(self, asset_root: Path | None = None) -> None:
        root = _default_asset_root() if asset_root is None else Path(asset_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"ShapeLibrary asset_root does not exist or is not a directory: {root}"
            )
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ShapeLibrary manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)

        schema_version = manifest.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version={schema_version!r}; "
                f"this loader expects {_SCHEMA_VERSION}"
            )
        shapes = manifest.get("shapes")
        if not isinstance(shapes, dict) or not shapes:
            raise ValueError(
                "Manifest 'shapes' must be a non-empty object; "
                f"got {type(shapes).__name__}"
            )

        self._root = root
        self._entries: dict[str, dict[str, Any]] = {}
        for name, entry in shapes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"Manifest shape names must be non-empty strings; got {name!r}"
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Manifest entry for {name!r} must be an object; "
                    f"got {type(entry).__name__}"
                )
            for required in ("file", "rest_dimensions", "binding_template"):
                if required not in entry:
                    raise ValueError(
                        f"Manifest entry {name!r} is missing required field {required!r}"
                    )
            self._entries[name] = entry

        self._cache: dict[str, MeshShape] = {}

    # ---- Public API ----------------------------------------------------

    def names(self) -> tuple[str, ...]:
        """Return the manifest's shape names in insertion order."""
        return tuple(self._entries.keys())

    def get(self, name: str) -> MeshShape:
        """Load (and cache) the named :class:`MeshShape`.

        Raises
        ------
        KeyError
            If ``name`` is not in the manifest. The error message lists the
            available names.
        """
        if name in self._cache:
            return self._cache[name]
        entry = self._lookup(name)
        path = self._root / str(entry["file"])
        shape = MeshShape.load(path, max_vertices=5000)
        self._cache[name] = shape
        return shape

    def binding_template(self, name: str) -> MarkerBinding:
        """Return the manifest's :class:`MarkerBinding` template for ``name``."""
        entry = self._lookup(name)
        template = entry["binding_template"]
        if not isinstance(template, dict):
            raise ValueError(
                f"binding_template for {name!r} must be an object; "
                f"got {type(template).__name__}"
            )
        kind_str = template.get("kind")
        if not isinstance(kind_str, str):
            raise ValueError(
                f"binding_template.kind for {name!r} must be a string; got {kind_str!r}"
            )
        try:
            kind = BindingKind(kind_str)
        except ValueError as exc:
            raise ValueError(
                f"binding_template.kind for {name!r} is not a known BindingKind: "
                f"{kind_str!r}"
            ) from exc

        marker_names = template.get("marker_names")
        if not isinstance(marker_names, list) or not all(
            isinstance(m, str) for m in marker_names
        ):
            raise ValueError(
                f"binding_template.marker_names for {name!r} must be a list of strings"
            )

        quat = template.get("rest_orientation_quat", (1.0, 0.0, 0.0, 0.0))
        if not isinstance(quat, (list, tuple)) or len(quat) != 4:
            raise ValueError(
                f"binding_template.rest_orientation_quat for {name!r} "
                f"must be a length-4 list; got {quat!r}"
            )

        return MarkerBinding(
            kind=kind,
            marker_names=tuple(marker_names),
            rest_orientation_quat=(
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            ),
        )

    @classmethod
    def default(cls) -> ShapeLibrary:
        """Load the bundled default library shipped with this repo."""
        return cls()

    # ---- Internals -----------------------------------------------------

    def _lookup(self, name: str) -> dict[str, Any]:
        try:
            return self._entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"Unknown shape name {name!r}. Available shapes: {available}"
            ) from exc
