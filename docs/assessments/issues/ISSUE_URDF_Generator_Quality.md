# Issue: Quality Workarounds in URDF Generator

**ID:** ISSUE_URDF_GENERATOR_QUALITY
**Date:** 2026-01-21
**Severity:** HIGH
**Labels:** jules:code-quality, technical-debt

## Description
The newly added `tools/urdf_generator/mujoco_viewer.py` contains explicit workarounds to bypass code quality checks and type safety.

## Findings
1.  **Magic Number Bypass:**
    *   Code: `GRAVITY_M_S2 = 9.810`
    *   Issue: The value `9.810` is mathematically identical to `9.81` but is formatted specifically to bypass the regex check `(?<![0-9])9\.8[0-9]?(?![0-9])` in `tools/code_quality_check.py`.
    *   Impact: This indicates an intent to bypass checks rather than solve the problem (which would be importing the constant or defining it properly).

2.  **Blanket Type Ignore:**
    *   Code: `# mypy: ignore-errors` (Line 1)
    *   Issue: Disables all type checking for the entire file.
    *   Impact: Increases risk of runtime errors in the visualization logic.

## Remediation
1.  **Refactor Gravity:**
    ```python
    # Bad
    GRAVITY_M_S2 = 9.810

    # Good
    from shared.python.numerical_constants import GRAVITY_STANDARD
    ```
2.  **Enable Type Checking:**
    *   Remove `# mypy: ignore-errors`.
    *   Use specific `type: ignore` for the `mujoco` import if stubs are missing.
    *   Fix legitimate type errors.

## References
*   `tools/urdf_generator/mujoco_viewer.py`
*   `tools/code_quality_check.py`
