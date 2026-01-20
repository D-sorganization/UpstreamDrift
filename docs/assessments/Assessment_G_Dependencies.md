# Assessment G: Dependencies

## Grade: 9/10

## Summary
Dependencies are managed using modern standards (`pyproject.toml`) with clear separation of concerns (dev, engines, analysis groups).

## Strengths
- **Modern Standards**: Uses `pyproject.toml` [project] table (PEP 621).
- **Segmentation**: Optional dependencies are well-grouped (`engines`, `analysis`, `dev`), allowing lighter installs.
- **Version Constraints**: Most dependencies have version caps (e.g., `<2.0.0`) to prevent breaking changes from major updates.

## Weaknesses
- **Broad Ranges**: Some version ranges are relatively broad (e.g., `sqlalchemy>=2.0.0,<3.0.0`), which is good for library compatibility but might require a lockfile for application reproducibility. `requirements.lock` exists but `pyproject.toml` is the source of truth.

## Recommendations
- Encourage the use of the `requirements.lock` file in CI/CD pipelines to ensure reproducible builds.
