"""Golf environment visualization package."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import GolfEnvironmentAdapter

# Register immediately when the package is imported
register_embeddable_tool(GolfEnvironmentAdapter())

__all__ = ["GolfEnvironmentAdapter"]
