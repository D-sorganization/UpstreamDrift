---
issue: 14089
title: "Sim-Run: Result Validation and Provenance Gate Before Succeeded"
summary: "Result envelope records repo SHA, engine version, and input sha256; numerical integrity gate recursively catches NaN and Inf in outputs; schema-invalid or NaN-emitting runs force failed state with structured diagnostics."
branch: "feat/14089-sim-run-result-validation"
---
