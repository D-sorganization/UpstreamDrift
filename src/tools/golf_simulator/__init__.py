"""Golf Simulator desktop tool package."""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import GolfSimulatorEmbedAdapter

register_embeddable_tool(GolfSimulatorEmbedAdapter())

__all__ = ["GolfSimulatorEmbedAdapter"]
