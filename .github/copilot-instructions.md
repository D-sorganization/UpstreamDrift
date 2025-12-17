# GitHub Copilot Instructions — Guardrails for Code Generation

## Purpose
Force Copilot to produce complete, test-passing, placeholder-free code when editing repositories in VS Code.

---
## Completion Checklist
Before saying “done”, ensure ALL:
- [ ] Diff ≤200 lines OR split into stacked parts.
- [ ] No `TODO`, `FIXME`, `pass`, `NotImplementedError` in maintained code.
- [ ] `pre-commit run --all-files` passes; paste output summary.
- [ ] Tests updated/added; paste pytest output (or MATLAB tests output).
- [ ] README/USAGE snippet showing change in action is included.
- [ ] Risk + rollback plan documented.
- [ ] Cross-file imports resolved; repo layout respected.

---
## Change Process
1. **Plan first** — Identify all files to change, dependencies, and intended strategy.
2. **Batch changes**: mechanical → typing → logic.
3. **Output complete diffs** — No prose-only responses.
4. **Run these commands and show results**:
```bash
ruff check <touched_dir> --fix
mypy <touched_dir> || true
pre-commit run --all-files
pytest -q || echo "No tests found"
```
5. **If blocked** — Stop, explain blockers, propose staged plan.

---
## Repo Layout Reference
- Python: `python/<project>/` for libs, `python/<project>/tests/` for tests.
- MATLAB: `matlab/<project>/` for libs, `matlab/tests/` for tests.
- Scripts: `/scripts` for CLI entry points.
- Large files tracked with Git LFS.

---
## Definitions of Done (DoD)
- Lint/format/type-check/test all green.
- No placeholders in production code.
- Documentation & usage updated.
- PR template completed with traceability, tests, risk notes.
