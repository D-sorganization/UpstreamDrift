---
issue: 15901
summary: "Add a closed-shell tousled-wave hairstyle"
dl_state: in_review
next_step: "Review the hairstyle geometry, screenshots, native invariance, and calculation note before opening the pull request."
branch: feat/15901-wavy-hair
paths: src/shared/python/model_appearance/head_hair.py, src/shared/python/model_appearance/head_hair_styles.py, src/shared/python/model_appearance/schema.py, src/shared/python/model_appearance/appearance_v1.schema.json, tests/unit/model_appearance/test_hair_styles.py, docs/research/model_appearance/tousled_wave_hair_reference.tex
---

Add a schema-validated `tousled_wave` style that bends crown comb ridges over the existing closed CC0 scalp. Keep legacy and capped hair geometry unchanged, verify closed-shell and native-physics invariants, and document the source-informed calculation with an editable LaTeX reference.
