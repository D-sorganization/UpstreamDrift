"""Golf Simulation Suite — combined ball-flight and putting visualiser.

Audit status (issue #6089): the ``__main__.py`` entry point requires
``pyvista`` and ``pyvistaqt``, which are optional dependencies.  When they
are missing the embedded launcher tile shows a degraded state rather than
crashing the launcher.  The adapter is registered eagerly so the tile
appears in the launcher; instantiation is deferred until the user opens
the tab.
"""

from src.shared.python.launcher_embed import register_embeddable_tool

from ._embed_adapter import GolfSimulationSuiteAdapter

# Register immediately when the package is imported
register_embeddable_tool(GolfSimulationSuiteAdapter())

__all__ = ["GolfSimulationSuiteAdapter"]
