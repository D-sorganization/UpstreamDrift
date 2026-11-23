---
issue: 12691
summary: "Correct the GS3DX design report: the committed Simscape golfer ships a rigid shaft, and a test pins the claim to the model."
---

The GS3DX design report claimed a flexible shaft, but in `GS3DX_Human.slx` the `Flexible Beam Model` subsystem is commented out and the active shaft is `Rigid Shaft`. The report now says rigid. `tests/unit/engines/simscape/test_gs3dx_shaft_claim.py` reads the model zip and allows a flexible-shaft claim only while the beam subsystem is active. This is the R11 verification step for Simscape; the live R2025b chain realization stays open.
