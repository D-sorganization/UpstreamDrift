"""Golf Simulation Suite module."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import GolfSimulationSuiteAdapter

# Register immediately when the package is imported
register_embeddable_tool(GolfSimulationSuiteAdapter())

__all__ = ["GolfSimulationSuiteAdapter"]
