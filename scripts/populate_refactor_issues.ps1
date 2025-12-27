$issues = @(
    @{
        Title = "Phase 1.1: Fix Pytest Coverage Configuration"
        Body = "**Problem:** `pyproject.toml` currently includes `--cov=engines` in `addopts` but blindly omits `engines/` in the `[tool.coverage.run]` section, creating a configuration contradiction.`n`n**Action:** Align coverage targets to include shared components and launchers first, then selectively add engines."
        Labels = "refactor,ci/cd"
    },
    @{
        Title = "Phase 1.2: Consolidate Dependency Management"
        Body = "**Problem:** 16+ `requirements.txt` files exist, causing version drift and maintenance headaches.`n`n**Action:** Migrate all dependencies to the root `pyproject.toml` using `optional-dependencies` groups (e.g., `[project.optional-dependencies] mujoco = [...]`). Delete all `requirements.txt` files."
        Labels = "refactor,dependencies,technical-debt"
    },
    @{
        Title = "Phase 1.3: Implement Duplicate File Prevention"
        Body = "**Problem:** 7+ copies of `matlab_quality_check.py` exist across the codebase.`n`n**Action:** Canonicalize the script in `tools/` and add a CI check (`scripts/check_duplicates.py`) to fail PRs if duplicates are detected."
        Labels = "quality-control,ci/cd"
    },
    @{
        Title = "Phase 1.4: Fix Python Version Metadata"
        Body = "**Problem:** README advertises Python 3.10+, while `pyproject.toml` requires 3.11+. Usage of `match/case` implies 3.10+, but type hinting features might require 3.11.`n`n**Action:** Update metadata to reflect the true requirement (3.11+) and align CI matrices."
        Labels = "documentation,config"
    },
    @{
        Title = "Phase 2.1: GUI Refactoring (SRP)"
        Body = "**Problem:** `advanced_gui.py` is 3,933 lines long, violating the Single Responsibility Principle and making maintenance difficult.`n`n**Action:** Split into sub-modules:`n- `gui/core/main_window.py` (Main Window)`n- `gui/widgets/simulation_panel.py` (Simulation Controls)`n- `gui/visualization/renderer.py` (Visualization)"
        Labels = "refactor,technical-debt,size/XL"
    },
    @{
        Title = "Phase 2.2: Archive & Legacy Cleanup"
        Body = "**Problem:** 13+ archive directories bloat the repository and confuse contributors.`n`n**Action:** Move valid historical code to a `legacy/` branch and delete these folders from `main`."
        Labels = "technical-debt,size/L"
    },
    @{
        Title = "Phase 2.3: Constants Normalization"
        Body = "**Problem:** Multiple `constants.py` files exist with conflicting values (e.g., inconsistent gravity definitions).`n`n**Action:** Create `shared/python/physics_constants.py` as the source of truth and update all engines to import from it."
        Labels = "refactor,quality-control,size/S"
    },
    @{
        Title = "Phase 3.1: Cross-Engine Integration Tests"
        Body = "**Problem:** No automated way to verify that MuJoCo, Drake, and Pinocchio results match, leading to potential divergence.`n`n**Action:** Create `tests/integration/test_physics_consistency.py` that runs simple pendulums on all 3 engines and asserts result proximity."
        Labels = "tests,enhancement,quality-control"
    },
    @{
        Title = "Phase 3.2: Architecture Documentation"
        Body = "**Problem:** Missing high-level diagrams makes onboarding difficult.`n`n**Action:** Add Mermaid diagrams for 'Engine Loading Flow' and 'Data Pipeline' to `docs/architecture/`."
        Labels = "documentation"
    },
    @{
        Title = "Phase 3.3: Launcher Configuration Abstraction"
        Body = "**Problem:** `golf_launcher.py` contains hardcoded paths and mixed concerns.`n`n**Action:** Move model definitions to `config/models.yaml` and create a `ModelRegistry` class to handle loading."
        Labels = "refactor,size/M"
    },
    @{
        Title = "Phase 4.1: Async Engine Loading"
        Body = "**Problem:** App freezes while MATLAB/MuJoCo loads (synchronous init).`n`n**Action:** Move engine initialization to background threads with a splash screen/progress bar."
        Labels = "enhancement,size/L"
    },
    @{
        Title = "Phase 4.2: Lazy Import Implementation"
        Body = "**Problem:** `golf_launcher.py` imports heavy PyQt6 modules at the top level, slowing down CLI response.`n`n**Action:** Move imports inside functions where possible to speed up CLI response time."
        Labels = "enhancement,size/S"
    }
)

Write-Host "Starting bulk issue creation..." -ForegroundColor Cyan

foreach ($issue in $issues) {
    Write-Host "Creating issue: $($issue.Title)" -NoNewline
    try {
        $out = gh issue create --title $issue.Title --body $issue.Body --label $issue.Labels 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
            Write-Host $out
        } else {
            Write-Host " [FAILED]" -ForegroundColor Red
            Write-Host $out
        }
    } catch {
        Write-Host " [ERROR]" -ForegroundColor Red
        Write-Host $_
    }
    Start-Sleep -Milliseconds 500
}

Write-Host "Drafted $($issues.Count) issues." -ForegroundColor Cyan
