# Character Builder

Desktop tool and API for building a golfer as a `full-body-v1` specification
(epic #11651; CMB-1 #11652, CMB-2 #11653, CMB-3 #11654).

- `core.py` - headless model: parameters, presets, compile, export (no Qt).
- `gui.py` - PyQt6 form bound to the core.
- `_embed_adapter.py` - launcher embed adapter (entry point `character_builder`).
- Shared logic: `src/shared/python/humanoid_character_builder/` (`spec_params`,
  `spec_export`, `presets/`).
- Web API: `src/api/routes/character_builder.py`.

Run: `python3 -m src.tools.character_builder`.
