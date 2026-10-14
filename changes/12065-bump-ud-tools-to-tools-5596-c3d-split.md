---
issue: 12065
summary: "Bump vendor/ud-tools to f7440f1823 (Tools #5596 c3d split); re-sync the sidekick c3d_reader child copy and its new characterization test, and retire the dated force_plate_dataframe architecture-budget exception"
branch: "chore/ud-tools-pin-5596"
---

Refs #12065. Advances the Tools pin from bdc3eed0e934 (Tools #5581) to f7440f1823 (Tools #5596) in the gitlink, `Cargo.toml` (`tools-core`), `requirements-tools.txt` and `src/config/impact_acceptance.json`. `C3DDataReader.force_plate_dataframe` is now under the 100-line budget, so its #12065 exception is removed.
