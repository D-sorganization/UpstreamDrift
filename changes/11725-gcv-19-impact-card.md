---
issue: 11725
summary: "GCV-19: native exports draw the GCV-17 impact parameter card from the impact frame on and, on the grf and club_forces showcase clips, a synchronised inset plot strip (GRF in BW with CoP; grip force and couple); unavailable values are shown as unavailable and recorded in the showcase manifest"
branch: "claude/gcv-19-impact-card"
---

Refs #11725. Adds `compositor.draw_impact_card`, `impact_card.impact_card_for`, `ExportSettings.impact_card` and `ExportResult.impact_card`; the showcase no longer lists the card as a known gap. Adds `inset_strip.py` (`grf_series`, `grip_series`, `render_strip`, `draw_strip`, `StripRenderer`), `ExportSettings.inset_strip` and `ExportResult.inset_strip`; the showcase enables the strip on `grf` and `club_forces`, records it per clip in the manifest, and `KNOWN_GAPS` is now empty. Tests: tests/unit/tools/native_viewer_export/test_impact_card.py, tests/unit/tools/native_viewer_export/test_inset_strip.py, tests/scripts/test_render_forces_showcase.py.
