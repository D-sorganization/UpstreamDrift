---
issue: 13938
summary: "Remove uncited Joule heating bias from default glass conductivity model"
branch: "fix/13938-glass-conductivity-uncited-bias"
dl_state: "in_review"
next_step: "Merge PR"
---

Remove uncited Joule heating bias (`temp_kelvin += power_density * 0.0001`) from `GlassPropertiesInterface._default_conductivity_model` per #13938 so that electrical conductivity is evaluated strictly at the glass temperature.
