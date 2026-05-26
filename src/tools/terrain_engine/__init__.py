"""Terrain Engine launcher tool."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import TerrainEngineAdapter

# Register immediately when the package is imported
register_embeddable_tool(TerrainEngineAdapter())

__all__ = ["TerrainEngineAdapter"]
