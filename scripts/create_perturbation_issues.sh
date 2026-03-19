#!/usr/bin/env bash
# Create GitHub issues for perturbation analysis across all engines.
# Requires: gh CLI authenticated (gh auth login)
# Usage: ./scripts/create_perturbation_issues.sh
#
# This script creates 7 GitHub issues from the issue markdown files in issues/
# and applies appropriate labels.

set -euo pipefail

REPO="D-sorganization/UpstreamDrift"

# Ensure labels exist
echo "Creating labels (will skip if they already exist)..."
gh label create "perturbation-analysis" --repo "$REPO" --color "0E8A16" --description "Perturbation sensitivity analysis" 2>/dev/null || true
gh label create "physics-engine" --repo "$REPO" --color "1D76DB" --description "Physics engine implementation" 2>/dev/null || true
gh label create "pendulum" --repo "$REPO" --color "D93F0B" --description "Pendulum models engine" 2>/dev/null || true
gh label create "pinocchio" --repo "$REPO" --color "FBCA04" --description "Pinocchio engine" 2>/dev/null || true
gh label create "drake" --repo "$REPO" --color "5319E7" --description "Drake engine" 2>/dev/null || true
gh label create "mujoco" --repo "$REPO" --color "B60205" --description "MuJoCo engine" 2>/dev/null || true
gh label create "opensim" --repo "$REPO" --color "0075CA" --description "OpenSim engine" 2>/dev/null || true
gh label create "myosuite" --repo "$REPO" --color "E4E669" --description "MyoSuite engine" 2>/dev/null || true
gh label create "cross-engine" --repo "$REPO" --color "C5DEF5" --description "Cross-engine integration" 2>/dev/null || true
gh label create "phase-1" --repo "$REPO" --color "BFD4F2" --description "Phase 1: Core implementation" 2>/dev/null || true
gh label create "phase-3" --repo "$REPO" --color "D4C5F9" --description "Phase 3: Cross-engine integration" 2>/dev/null || true

echo ""
echo "Creating issues..."

# Issue 1: Pendulum (reference implementation)
echo "Creating: Pendulum perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[Pendulum Models] Perturbation Analysis: Core Module Formalization" \
  --body-file issues/006_pendulum_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,pendulum,phase-1"
echo ""

# Issue 2: Pinocchio
echo "Creating: Pinocchio perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[Pinocchio] Perturbation Analysis: Core Module Implementation" \
  --body-file issues/007_pinocchio_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,pinocchio,phase-1"
echo ""

# Issue 3: Drake
echo "Creating: Drake perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[Drake] Perturbation Analysis: Core Module Implementation" \
  --body-file issues/008_drake_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,drake,phase-1"
echo ""

# Issue 4: MuJoCo
echo "Creating: MuJoCo perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[MuJoCo] Perturbation Analysis: Core Module Implementation" \
  --body-file issues/009_mujoco_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,mujoco,phase-1"
echo ""

# Issue 5: OpenSim
echo "Creating: OpenSim perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[OpenSim] Perturbation Analysis: Core Module Implementation" \
  --body-file issues/010_opensim_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,opensim,phase-1"
echo ""

# Issue 6: MyoSuite
echo "Creating: MyoSuite perturbation analysis..."
gh issue create --repo "$REPO" \
  --title "[MyoSuite] Perturbation Analysis: Core Module Implementation" \
  --body-file issues/011_myosuite_perturbation_analysis_core.md \
  --label "perturbation-analysis,physics-engine,myosuite,phase-1"
echo ""

# Issue 7: Cross-engine comparison
echo "Creating: Cross-engine comparison framework..."
gh issue create --repo "$REPO" \
  --title "[All Engines] Perturbation Analysis: Cross-Engine Comparison Framework" \
  --body-file issues/012_cross_engine_perturbation_comparison.md \
  --label "perturbation-analysis,cross-engine,phase-3"
echo ""

echo "Done! All 7 perturbation analysis issues created."
echo ""
echo "Issue structure:"
echo "  Phase 1 (Core - implement in order):"
echo "    1. [Pendulum] Reference implementation (priority 1)"
echo "    2. [Pinocchio] Analytical dynamics (priority 2)"
echo "    3. [MuJoCo] Fast simulation (priority 3)"
echo "    4. [Drake] Systems framework (priority 4)"
echo "    5. [OpenSim] Musculoskeletal (priority 5)"
echo "    6. [MyoSuite] RL policy evaluation (priority 6)"
echo "  Phase 3 (Integration):"
echo "    7. [All Engines] Cross-engine comparison framework"
