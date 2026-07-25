"""Field-ownership contract between the two launcher manifests (issue #8089).

UpstreamDrift ships two launcher descriptions:

``src/config/models.yaml``
    Read by the native PyQt6 launcher (``LauncherOrchestrator`` ->
    ``ModelRegistry``). It describes *how a tile launches*: the entry point,
    the handler type, the engine, interpreter paths, the provider repo.

``src/config/launcher_manifest.json``
    Read by the API and the Tauri/React dashboard
    (``LauncherManifest.load``). It describes *how a tile is presented on the
    web*: display order, SVG logo, readiness chip, and the web launch
    contract.

Historically both files hand-authored the same user-visible semantics and
drifted apart on every shared tile. This module makes the split explicit and
machine-checkable:

* :data:`REGISTRY_OWNED_FIELDS` are authored **only** in ``models.yaml``. The
  matching entries in ``launcher_manifest.json`` are *generated* — run
  ``python -m scripts.sync_launcher_manifest --fix`` after editing the YAML.
* :data:`MANIFEST_OWNED_FIELDS` are authored **only** in
  ``launcher_manifest.json``. The native launcher does not read them.
* :data:`FIELD_EXCEPTIONS` records the handful of tiles that legitimately
  differ, each with a written reason. Anything not listed here must agree.

``tests/config/launcher_manifest/test_registry_manifest_parity.py`` enforces
all three rules, so a new drift cannot reach main silently.
"""

from __future__ import annotations

from typing import Final

#: Fields whose authoritative value lives in ``models.yaml``. The
#: corresponding value in ``launcher_manifest.json`` is generated from it and
#: must compare equal for every shared tile ID.
REGISTRY_OWNED_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "description",
    "category",
    "type",
    "path",
    "engine_type",
    "capabilities",
    "provider",
    "source_root",
    "python_paths",
    "hidden",
)

#: Fields authored only in ``launcher_manifest.json``. They describe the web
#: surface, which the native launcher never renders, so there is nothing in
#: ``models.yaml`` to disagree with.
MANIFEST_OWNED_FIELDS: Final[dict[str, str]] = {
    "order": (
        "Display order of the web dashboard grid. The native launcher orders "
        "tiles by category and the user's saved layout, so it has no "
        "equivalent value to disagree with."
    ),
    "logo": (
        "The native launcher draws repo-relative raster assets "
        "(assets/*.png); the web dashboard draws assets/logos/*.svg. The two "
        "asset sets are deliberately different files."
    ),
    "status": (
        "The native launcher shows a lifecycle chip (ready/beta/experimental/"
        "deprecated); the web dashboard shows a readiness chip (gui_ready/"
        "engine_ready/utility/simulator/external). Two vocabularies, two "
        "surfaces."
    ),
    "web": "Web-only launch contract (issue #7461). No native equivalent.",
    "web_route": "Legacy web-only route field superseded by ``web``.",
    "default_launch": (
        "Native docking preference (tab/dock/window). The web dashboard "
        "always opens a route or a native window."
    ),
}

#: ``(tile_id, field) -> reason``. A documented, deliberate difference on an
#: otherwise registry-owned field. Keep this list short; every entry is a
#: place where the two surfaces really do need different data.
FIELD_EXCEPTIONS: Final[dict[tuple[str, str], str]] = {
    ("motion_target_preview", "path"): (
        "The web/embedded loader imports this tool as a module "
        "(src.tools.starting_pose_matcher.__main__) rather than executing a "
        "script file, so the manifest carries the dotted module path while "
        "models.yaml carries the file the native launcher runs."
    ),
    ("starting_pose_matcher", "path"): (
        "Legacy alias of motion_target_preview; carries the same dotted "
        "module path for the web/embedded loader."
    ),
}


def is_exempt(tile_id: str, field: str) -> bool:
    """Return True when ``field`` is allowed to differ for ``tile_id``."""
    if not isinstance(tile_id, str) or not tile_id:
        raise ValueError("tile_id must be a non-empty string")
    if not isinstance(field, str) or not field:
        raise ValueError("field must be a non-empty string")
    return (tile_id, field) in FIELD_EXCEPTIONS
