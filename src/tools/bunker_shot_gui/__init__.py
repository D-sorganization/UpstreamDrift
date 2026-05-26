"""Bunker shot 3D simulator package."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import BunkerShotGuiAdapter

# Register immediately when the package is imported
register_embeddable_tool(BunkerShotGuiAdapter())

__all__ = ["BunkerShotGuiAdapter"]
