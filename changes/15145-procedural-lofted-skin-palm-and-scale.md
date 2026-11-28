---
issue: 15145
summary: "Fix procedural_skin scale, coherent skeleton chaining, curved cupped palm slab with dorsal knuckle valleys, distal pad wrap, opposed thumb, Vardon overlap curl, and smooth 2x2 SSAA backend comparison renders"
---
- Adopted the shared ADR-0057 hand backend registry (#15197): `procedural_skin` registers via `register()` and passes the conformance suite (contract signature, `skin_rgb`, rigid shaft roll applied in the builder). The branch-local registry and result types are gone; `contract.py` keeps only `GripArticulation`; the capsule stays an unregistered harness baseline, and `comparison.py` lists it plus the registered backends.
