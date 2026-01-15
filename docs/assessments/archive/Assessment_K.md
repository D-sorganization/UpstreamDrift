# Assessment K: Reproducibility & Provenance

## Grade: 7/10

## Focus
Determinism, versioning, experiment tracking.

## Findings
*   **Strengths:**
    *   `provenance.py` exists (based on memory), suggesting metadata capture.
    *   `GenericPhysicsRecorder` captures comprehensive state (q, v, tau).
    *   `seed` setting is common in scientific sims (though not explicitly verified in every file, `np.random` usage suggests it).

*   **Weaknesses:**
    *   Physics engines (MuJoCo, Drake) are deterministic, but floating-point differences across platforms/versions can affect reproducibility.
    *   Dependency versions are pinned (`<2.0.0`), but not exact hashes (no `requirements.lock` seen, only `pyproject.toml` ranges).

## Recommendations
1.  Generate a `requirements.lock` or `poetry.lock` file.
2.  Include git commit hash and dirty state in every recorded data file.
