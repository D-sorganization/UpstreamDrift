#!/usr/bin/env bash
# run_local_heavy_tests.sh — UpstreamDrift
# ─────────────────────────────────────────────────────────────────────────────
# LOCAL DOCKER PARITY RUNNER
# Mirrors .github/workflows/heavy-tests-opt-in.yml + reusable-heavy-tests.yml
# Every step here should have a direct equivalent in those YAML files.
# Break parity → open a bug.
#
# Prerequisites:
#   - WSL2 with Docker running (sudo service docker start)
#   - OR Docker Desktop
#
# Usage:
#   wsl bash run_local_heavy_tests.sh               # Run all live_simulation tests
#   wsl bash run_local_heavy_tests.sh --no-build    # Skip Docker rebuild
#   wsl bash run_local_heavy_tests.sh --deps-only   # Verify dependencies only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_NAME="UpstreamDrift"
IMAGE_NAME="d-sorg-fleet-local"
TEST_PATH="tests/heavy_integration"
PYTHON_VERSION="3.11"
NO_BUILD=false
DEPS_ONLY=false

# Parse args
for arg in "$@"; do
  case $arg in
    --no-build) NO_BUILD=true ;;
    --deps-only) DEPS_ONLY=true ;;
  esac
done

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo -e "${BOLD}=====================================================================${RESET}"
echo -e "${BOLD} Heavy Integration Test Runner — ${REPO_NAME}${RESET}"
echo -e "${BOLD} $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "${BOLD}=====================================================================${RESET}"

# ── Step 1: Verify Docker is running ─────────────────────────────────────────
echo -e "\n${YELLOW}[1/5] Checking Docker daemon...${RESET}"
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}✗ Docker is not running. Start it with: sudo service docker start${RESET}"
  exit 1
fi
echo -e "${GREEN}✓ Docker daemon is running.${RESET}"

# ── Step 2: Build image (unless --no-build) ───────────────────────────────────
if [ "$NO_BUILD" = false ]; then
  echo -e "\n${YELLOW}[2/5] Building local heavy test image (${IMAGE_NAME})...${RESET}"
  docker build \
    --network=host \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile.heavy_test \
    --progress=plain \
    . 2>&1 | tail -20
  echo -e "${GREEN}✓ Image built successfully.${RESET}"
else
  echo -e "\n${YELLOW}[2/5] Skipping build (--no-build specified).${RESET}"
fi

# ── Step 3: Verify dependencies (parity with 'Verify Heavy Dependencies' step) ─
echo -e "\n${YELLOW}[3/5] Verifying heavy dependencies...${RESET}"
docker run --rm --network=host "${IMAGE_NAME}" python -c "
deps = ['mujoco', 'pinocchio', 'mediapipe', 'pyvista', 'trimesh', 'scipy', 'sympy']
failures = []
for pkg in deps:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError as e:
        failures.append(f'{pkg}: {e}')
        print(f'  ❌ {pkg}: {e}')
if failures:
    import sys; sys.exit(1)
print('All heavy dependencies verified.')
"
echo -e "${GREEN}✓ All dependencies present.${RESET}"

if [ "$DEPS_ONLY" = true ]; then
  echo -e "\n${YELLOW}--deps-only specified, stopping here.${RESET}"
  exit 0
fi

# ── Step 4: Run tests ─────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/5] Running heavy integration tests...${RESET}"
RESULTS_DIR="$(pwd)/.heavy_test_results"
mkdir -p "${RESULTS_DIR}"

docker run --rm \
  --network=host \
  -v "$(pwd):/app:ro" \
  -v "${RESULTS_DIR}:/results" \
  "${IMAGE_NAME}" \
  xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
    pytest \
      -v \
      -m "live_simulation" \
      --timeout=300 \
      --timeout-method=thread \
      --tb=short \
      --junit-xml=/results/heavy_test_results.xml \
      --cov=. \
      --cov-report=xml:/results/heavy_coverage.xml \
      --cov-report=term-missing \
      "${TEST_PATH}" \
  | tee "${RESULTS_DIR}/heavy_test_output.log"
TEST_EXIT=$?

# ── Step 5: Report ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/5] Parsing results...${RESET}"
python3 - <<PYEOF
import xml.etree.ElementTree as ET, sys, os

report_path = '${RESULTS_DIR}/heavy_test_results.xml'
if not os.path.exists(report_path):
    print('No results XML found.')
    sys.exit(0)
tree = ET.parse(report_path)
root = tree.getroot()
suite = root.find('.//testsuite') or root
total    = int(suite.get('tests', 0))
failures = int(suite.get('failures', 0))
errors   = int(suite.get('errors', 0))
skipped  = int(suite.get('skipped', 0))
passed   = total - failures - errors - skipped
print(f"")
print(f"  Results Summary")
print(f"  ───────────────────────────────────")
print(f"  ✅ Passed:  {passed}")
print(f"  ❌ Failed:  {failures}")
print(f"  💥 Error:   {errors}")
print(f"  ⏭️  Skipped: {skipped}")
print(f"  📊 Total:   {total}")
if failures > 0 or errors > 0:
    print("")
    print("  Failures:")
    for case in root.iter('testcase'):
        fail = case.find('failure') or case.find('error')
        if fail is not None:
            print(f"    • {case.get('classname')}.{case.get('name')}")
            print(f"      {(fail.get('message') or '')[:200]}")
PYEOF

echo -e "\n${BOLD}=====================================================================${RESET}"
if [ $TEST_EXIT -eq 0 ]; then
  echo -e "${GREEN}${BOLD} ✓ ALL HEAVY TESTS PASSED${RESET}"
else
  echo -e "${RED}${BOLD} ✗ HEAVY TESTS FAILED (exit $TEST_EXIT)${RESET}"
  echo -e "${YELLOW}   Results saved to: ${RESULTS_DIR}/${RESET}"
fi
echo -e "${BOLD}=====================================================================${RESET}"
exit $TEST_EXIT
