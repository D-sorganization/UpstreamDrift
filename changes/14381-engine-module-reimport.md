---
issue: 14381
summary: "tests: _protect_engine_modules keeps genuine pinocchio/pydrake modules first imported during a test (their native extensions cannot be re-imported once discarded); stubs and src.engines additions are still discarded and replaced modules restored."
---
