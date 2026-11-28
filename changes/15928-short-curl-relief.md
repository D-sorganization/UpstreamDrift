---
issue: 15928
summary: "Add bounded, visibly irregular short-curl relief to the existing closed scalp hair shell"
dl_state: "in_review"
next_step: "Confirm exact-head hairstyle, native-physics, documentation, and architecture checks; inspect desktop and phone previews."
branch: "feat/15928-visible-curls"
paths: "src/shared/python/model_appearance/head_hair.py,src/shared/python/model_appearance/head_hair_styles.py,src/shared/python/model_appearance/schema.py,src/shared/python/model_appearance/appearance_v1.schema.json,tests/unit/model_appearance/test_hair_styles.py,scripts/render_hair_styles.py,docs/research/tousled_wave_hair_reference/tousled_wave_hair_reference.tex,docs/development/evidence/appearance/short_curls/,changes/15928-short-curl-relief.md"
---

Add a schema-validated `short_curls` hairstyle with smooth, bounded two-dimensional relief and a modest irregular front hairline. Keep other styles' geometry, cap fallback, and native dynamics unchanged. Document the 1984 Mac O'Grady and 1973 John Schlee portrait references and retain the limitation that this is a stylized closed-shell approximation, not photoreal hair or an individual likeness.
