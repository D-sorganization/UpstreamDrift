import subprocess
import sys
import time

issues = [
    {
        "title": "Phase 1.1: Fix Pytest Coverage Configuration",
        "body": "**Problem:** `pyproject.toml` currently includes `--cov=engines` in `addopts` but blindly omits `engines/` in the `[tool.coverage.run]` section, creating a configuration contradiction.\n\n**Action:** Align coverage targets to include shared components and launchers first, then selectively add engines.",
        "labels": "refactor,ci/cd",
    },
    {
        "title": "Phase 1.2: Consolidate Dependency Management",
        "body": "**Problem:** 16+ `requirements.txt` files exist, causing version drift and maintenance headaches.\n\n**Action:** Migrate all dependencies to the root `pyproject.toml` using `optional-dependencies` groups (e.g., `[project.optional-dependencies] mujoco = [...]`). Delete all `requirements.txt` files.",
        "labels": "refactor,dependencies,technical-debt",
    },
    {
        "title": "Phase 1.3: Implement Duplicate File Prevention",
        "body": "**Problem:** 7+ copies of `matlab_quality_check.py` exist across the codebase.\n\n**Action:** Canonicalize the script in `tools/` and add a CI check (`scripts/check_duplicates.py`) to fail PRs if duplicates are detected.",
        "labels": "quality-control,ci/cd",
    },
    {
        "title": "Phase 1.4: Fix Python Version Metadata",
        "body": "**Problem:** README advertises Python 3.10+, while `pyproject.toml` requires 3.11+. Usage of `match/case` implies 3.10+, but type hinting features might require 3.11.\n\n**Action:** Update metadata to reflect the true requirement (3.11+) and align CI matrices.",
        "labels": "documentation,config",
    },
    {
        "title": "Phase 2.1: GUI Refactoring (SRP)",
        "body": "**Problem:** `advanced_gui.py` is 3,933 lines long, violating the Single Responsibility Principle and making maintenance difficult.\n\n**Action:** Split into sub-modules:\n- `gui/core/main_window.py` (Main Window)\n- `gui/widgets/simulation_panel.py` (Simulation Controls)\n- `gui/visualization/renderer.py` (Visualization)",
        "labels": "refactor,technical-debt,size/XL",
    },
    {
        "title": "Phase 2.2: Archive & Legacy Cleanup",
        "body": "**Problem:** 13+ archive directories bloat the repository and confuse contributors.\n\n**Action:** Move valid historical code to a `legacy/` branch and delete these folders from `main`.",
        "labels": "technical-debt,size/L",
    },
    {
        "title": "Phase 2.3: Constants Normalization",
        "body": "**Problem:** Multiple `constants.py` files exist with conflicting values (e.g., inconsistent gravity definitions).\n\n**Action:** Create `shared/python/physics_constants.py` as the source of truth and update all engines to import from it.",
        "labels": "refactor,quality-control,size/S",
    },
    {
        "title": "Phase 3.1: Cross-Engine Integration Tests",
        "body": "**Problem:** No automated way to verify that MuJoCo, Drake, and Pinocchio results match, leading to potential divergence.\n\n**Action:** Create `tests/integration/test_physics_consistency.py` that runs simple pendulums on all 3 engines and asserts result proximity.",
        "labels": "tests,enhancement,quality-control",
    },
    {
        "title": "Phase 3.2: Architecture Documentation",
        "body": "**Problem:** Missing high-level diagrams makes onboarding difficult.\n\n**Action:** Add Mermaid diagrams for 'Engine Loading Flow' and 'Data Pipeline' to `docs/architecture/`.",
        "labels": "documentation",
    },
    {
        "title": "Phase 3.3: Launcher Configuration Abstraction",
        "body": "**Problem:** `golf_launcher.py` contains hardcoded paths and mixed concerns.\n\n**Action:** Move model definitions to `config/models.yaml` and create a `ModelRegistry` class to handle loading.",
        "labels": "refactor,size/M",
    },
    {
        "title": "Phase 4.1: Async Engine Loading",
        "body": "**Problem:** App freezes while MATLAB/MuJoCo loads (synchronous init).\n\n**Action:** Move engine initialization to background threads with a splash screen/progress bar.",
        "labels": "enhancement,size/L",
    },
    {
        "title": "Phase 4.2: Lazy Import Implementation",
        "body": "**Problem:** `golf_launcher.py` imports heavy PyQt6 modules at the top level, slowing down CLI response.\n\n**Action:** Move imports inside functions where possible to speed up CLI response time.",
        "labels": "enhancement,size/S",
    },
]

print(f"Starting bulk issue creation for {len(issues)} items...", file=sys.stderr)

for issue in issues:
    print(f"Creating issue: {issue['title']}...", end="", flush=True)
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        issue["title"],
        "--body",
        issue["body"],
        "--label",
        issue["labels"],
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(" [OK]")
            print(result.stdout.strip())
        else:
            print(" [FAILED]")
            print(result.stderr.strip())
    except FileNotFoundError:
        print(" [ERROR] gh command not found.")
        break
    time.sleep(1.0)  # Rate limiting
