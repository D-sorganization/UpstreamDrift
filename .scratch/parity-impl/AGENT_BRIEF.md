# Parity epic #7462 — implementation agent brief (shared rules)

You are implementing one issue from UpstreamDrift epic #7462 (PyQt6 = canonical model; Tauri/React web app must match). Repo: `C:\Users\diete\Repositories\UpstreamDrift` (remote `D-sorganization/UpstreamDrift`).

## Hard rules

1. **NEVER touch the main working tree** at `C:\Users\diete\Repositories\UpstreamDrift` — it is checked out on someone else's branch with in-flight work. Create your own worktree:
   ```powershell
   git -C C:\Users\diete\Repositories\UpstreamDrift worktree add C:\Users\diete\Repositories\.wt-parity-<SLUG> -b feat/parity-<SLUG>-<ISSUE_NUM> origin/main
   ```
   Wait for checkout to fully complete (11k+ files) before reading any file in it. Do ALL work inside that worktree.
2. **Lease (best-effort, fail-open):** before coding run
   `cd C:\Users\diete\Repositories\Repository_Management; python -m scripts.check_agent_claim --repo UpstreamDrift --issue <N>`
   If held by another agent → STOP and report. Otherwise post:
   `python -m scripts.post_agent_lease --agent claude --session parity-7462 --repo UpstreamDrift --issue <N>`
   If the scripts error, proceed anyway.
3. **Read the issue first:** `gh issue view <N>` (run gh from the worktree dir). The issue body is the spec — follow its Proposed fix and Acceptance criteria. Verify claims against the code before building (files may have moved).
4. **Python:** ALWAYS `C:\Users\diete\Repositories\UpstreamDrift\.venv\Scripts\python.exe` (bare `python` is Anaconda with numpy<2 and breaks things). For git push, prepend `.venv\Scripts` to PATH first (pre-push hook needs it):
   ```powershell
   $env:PATH = "C:\Users\diete\Repositories\UpstreamDrift\.venv\Scripts;" + $env:PATH
   ```
5. **Quality gates (CI-enforced):** `python -m ruff check .` and `python -m ruff format .` (run format before committing); file size budget 1200 lines/file; no `print()` in `src/`; no TODO/FIXME without issue number; error-handling helpers from `core.process_safety` (see CLAUDE.md in repo); TDD — tests in the same PR. For `ui/` changes: `npm test -- --run` and `npx tsc --noEmit` from the `ui/` dir (node_modules exists in the MAIN checkout — run `npm ci` in your worktree's ui/ dir if needed, or set up a junction is NOT allowed; just `npm install` in worktree ui/).
6. **Scope discipline:** implement YOUR issue only. If you can deliver 80% cleanly and 20% needs something not yet merged (e.g. depends on another parity PR), deliver the 80%, use `Part of #<N>` instead of `Fixes #<N>`, and say what remains in the PR body and an issue comment.
7. **Run targeted tests only** (`python -m pytest <paths you touched> -n auto --timeout=60`), not the full suite.

## Delivery (this is the durable output — do not skip)

1. Commit with a clear message ending `Fixes #<N>` (or `Part of #<N>`), co-author line:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
2. Push the branch; create PR: `gh pr create --title "<concise>" --body-file <body.md>` — body must list what was done, what remains, and acceptance-criteria checklist status. End body with the generated-with footer.
3. Arm auto-merge: `gh pr merge --auto --squash`. Do NOT watch/poll CI (rate limits).
4. Comment on the issue: PR link + one-paragraph status (so any agent can resume after this session dies).
5. Report back: issue #, PR #, branch, worktree path, what's done/remaining, any gotchas discovered.

## Known gotchas

- Pre-push hook: dbc/numpy errors → ensure .venv on PATH (rule 4). If the hook fails on engine-wheel imports (ui/engines errors) unrelated to your diff, note it and use `--no-verify` ONLY for that documented case, recording the hook output in the PR body.
- Don't batch dependent git/Read/Edit calls in one parallel block.
- SPEC.md freshness hook may require a SPEC bump when touching src/ — make a minimal honest entry if prompted.
- `vendor/ud-tools/` is read-only (vendored); never edit.
