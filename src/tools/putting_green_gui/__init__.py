"""Putting Green GUI package."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import PuttingGreenGuiAdapter

# Register immediately when the package is imported
register_embeddable_tool(PuttingGreenGuiAdapter())

__all__ = ["PuttingGreenGuiAdapter"]
