$issue1Title = "Critical Race Condition in Global Mutable State: _loaded_datasets"
$issue1Body = @"
The API relies on a module-level global dictionary `_loaded_datasets = {}` in `src/api/routes/data_explorer.py` to cache imported CSV/JSON data. FastAPI executes requests concurrently. If two users (or agents) call `/import` simultaneously, or one filters while another imports, the lack of `asyncio.Lock` or `threading.Lock` will result in race conditions, corrupted dataset reads, or silent data truncation.

**Recommendation:** Eliminate global in-memory state. Use Redis, a database (SQLite), or explicitly synchronized thread-safe caching wrappers per session.
"@

$issue2Title = "Unsafe Process Orchestration & Orphaned Processes"
$issue2Body = @"
`src/launchers/launcher_process_manager.py` utilizes `subprocess.Popen` heavily to spawn sub-GUIs and simulation servers. There is no evidence of robust process group (PGID) management or Windows Job Object binding. If the main FastAPI backend or GUI crashes or is killed via `SIGKILL`, the child `Popen` processes will be orphaned, locking GPU memory and ports indefinitely.

**Recommendation:** Implement Windows Job Objects (via `win32job`) and Linux process groups to ensure cascade termination.
"@

$issue3Title = "Implicit Trust in Safe Eval and AST Constraints"
$issue3Body = @"
`src/shared/python/safe_eval.py` wraps Python's native `eval()` using an AST validation pass. While `eval()` is blocked at the regex level, relying on AST traversal to sanitize arbitrary code execution for physics configuration is notoriously fragile. Python's AST changes between versions, and malicious inputs could bypass constraints to achieve RCE (Remote Code Execution).

**Recommendation:** Replace `eval()` entirely with a restricted DSL parser, SymPy, or a web-assembly sandbox.
"@

$issue4Title = "Swallowed Exceptions Masking Failure States in Analyzers"
$issue4Body = @"
`src/shared/python/perturbation/perturbation_base.py` (L361, L439) and `analyzer.py` contain broad `except Exception:` blocks with `noqa: BLE001` suppressions. Catching `Exception` broadly without re-raising or failing the workflow means that memory errors, database disconnects, or critical mathematical `NaN` exceptions during perturbation analysis are silently absorbed. The system will continue to generate "successful" artifacts that contain corrupt data.

**Recommendation:** Refactor to catch specific expected exceptions (`ValueError`, `RuntimeError`) and fail-fast on others to preserve data lineage trust.
"@

gh issue create --title $issue1Title --body $issue1Body --label "bug,security"
gh issue create --title $issue2Title --body $issue2Body --label "bug,reliability"
gh issue create --title $issue3Title --body $issue3Body --label "security"
gh issue create --title $issue4Title --body $issue4Body --label "bug,reliability"
