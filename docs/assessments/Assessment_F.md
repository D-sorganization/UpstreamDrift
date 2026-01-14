# Assessment F: Installation & Deployment

## Grade: 4/10

## Focus
pip/conda, cross-platform, CI/CD.

## Findings
*   **Strengths:**
    *   `pyproject.toml` clearly lists dependencies and uses optional groups (`dev`, `engines`).

*   **Weaknesses:**
    *   **CRITICAL**: `mujoco` dependency failed to load in the test environment, blocking test collection. This indicates fragility in the installation process or dependency specification for specific platforms.
    *   Dependencies are heavy and platform-specific (MuJoCo, Drake, Pinocchio), making cross-platform deployment a nightmare.

## Recommendations
1.  **Immediate**: Investigate why `mujoco` failed to import. Ensure `pip install` works in a fresh environment.
2.  Use `conda` environment files (`environment.yml`) to manage binary dependencies better than pip.
3.  Create a "light" version of the suite that works with a Mock engine for UI development without heavy physics deps.
