---
title: "CRITICAL: Repository Hygiene Violations & CI Gaming"
labels: ["jules:code-quality", "critical", "hygiene"]
assignee: "Lead Maintainer"
---

## Description
A recent review (2026-01-22) identified significant degradation in repository hygiene and CI standards:

1.  **Committed Artifacts:**
    *   `engines/physics_engines/mujoco/ruff_errors.txt`
    *   `engines/physics_engines/mujoco/ruff_errors_2.txt`
    *   These appear to be linter output logs that were accidentally staged and committed.

2.  **CI/CD Threshold Lowering:**
    *   `pyproject.toml` configures `--cov-fail-under=10`.
    *   This extremely low threshold (10%) effectively disables the coverage gate, allowing untested code to accumulate.

## Violation
*   **Repository Hygiene:** Committed log files pollute the history and the checkout.
*   **Quality Standards:** Lowering coverage gates to pass a massive merge is a form of "gaming" the metrics.

## Action Items
- [ ] Delete the `ruff_errors*.txt` files immediately.
- [ ] Investigate why `.gitignore` did not catch `*.txt` or these specific files.
- [ ] Raise the coverage threshold in `pyproject.toml` to a meaningful level (e.g., 20% or 25%) and roadmap the return to 60%.
