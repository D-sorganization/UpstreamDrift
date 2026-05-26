"""Ball Flight GUI package."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import BallFlightGuiAdapter

# Register immediately when the package is imported
register_embeddable_tool(BallFlightGuiAdapter())

__all__ = ["BallFlightGuiAdapter"]
