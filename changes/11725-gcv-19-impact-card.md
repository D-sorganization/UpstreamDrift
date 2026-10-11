---
issue: 11725
summary: "GCV-19: native exports draw the GCV-17 impact parameter card from the impact frame on; unavailable values are shown as unavailable and recorded in the showcase manifest"
branch: "claude/gcv-19-impact-card"
---

Refs #11725. Adds `compositor.draw_impact_card`, `impact_card.impact_card_for`, `ExportSettings.impact_card` and `ExportResult.impact_card`; the showcase no longer lists the card as a known gap. Tests: tests/unit/tools/native_viewer_export/test_impact_card.py, tests/scripts/test_render_forces_showcase.py.
