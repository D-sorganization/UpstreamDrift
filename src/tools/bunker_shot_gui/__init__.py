"""BunkerShot3D designer workbench (issue #8618, W11 of epic #8607).

The package splits cleanly in two:

* :mod:`~src.tools.bunker_shot_gui.design`,
  :mod:`~src.tools.bunker_shot_gui.model`,
  :mod:`~src.tools.bunker_shot_gui.field`,
  :mod:`~src.tools.bunker_shot_gui.shot3d`,
  :mod:`~src.tools.bunker_shot_gui.traces`,
  :mod:`~src.tools.bunker_shot_gui.render`,
  :mod:`~src.tools.bunker_shot_gui.render3d`,
  :mod:`~src.tools.bunker_shot_gui.render_traces` and
  :mod:`~src.tools.bunker_shot_gui.report` are the **headless** half. They
  import no GUI toolkit, so the whole computation -- geometry, sand, the F0
  DRFT solver, the W7 designer metrics, the per-element sole load field, the
  3-D shot scene, the scalar traces and their validity band, the playability
  window and the A/B ranking -- can be run and tested with no display, and can
  back the Tauri/React app. The three ``render`` modules draw with matplotlib
  and no toolkit, so a frame can be produced where PyQt6 does not load, and
  ``render3d`` draws the same backend-neutral ADR-0027 payload a MeshCat,
  Rerun or VTK provider would consume.
* :mod:`~src.tools.bunker_shot_gui.widgets`,
  :mod:`~src.tools.bunker_shot_gui.viewport_widgets` and
  :mod:`~src.tools.bunker_shot_gui.gui` are the Qt shell that renders it. One
  transport, the sole-field view's, drives every animated view.

Importing this package must stay Qt-free: the launcher registers the embed
adapter at import time, and the adapter defers its ``gui`` import until a
widget is actually asked for.
"""

from src.shared.python.launcher_embed import register_embeddable_tool

from ._embed_adapter import BunkerShotGuiAdapter

# Registered on import so the launcher can find the tool without importing Qt.
register_embeddable_tool(BunkerShotGuiAdapter())

__all__ = ["BunkerShotGuiAdapter"]
