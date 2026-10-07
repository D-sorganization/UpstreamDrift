"""Character Builder desktop tool (CMB-3, #11654).

Headless core (``core``) plus a PyQt6 shell (``gui``, imported lazily). Importing
this package registers the embeddable-tool adapter without touching Qt.
"""

from src.shared.python.launcher_embed import register_embeddable_tool

from ._embed_adapter import CharacterBuilderAdapter

register_embeddable_tool(CharacterBuilderAdapter())

__all__ = ["CharacterBuilderAdapter"]
