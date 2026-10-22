---
issue: 13619
summary: "Use Python 3.12 for the reviewed nightly cross-engine dependencies."
---

## Nightly Cross-Engine Runtime

Align the single nightly interpreter with the Python minimum required by the
reviewed Drake 1.57/1.58 releases. Preserve the combined engine extras, reviewed
version bounds, OpenSim failure gate, and dashboard publication sequencing.
Regression tests protect the runtime and these coverage contracts. Package
metadata and workflow tests do not establish native scientific acceptance.
