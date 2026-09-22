#!/usr/bin/env bash
# Bootstrap the pinned MyoHub myo_sim submodule for MS-51 (#10344).
# .gitmodules records the URL/path; the gitlink SHA is the pin of record.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_SHA="33f3ded946f55adbdcf963c99999587aadaf975f"
SUBMODULE_PATH="shared/models/myosuite/myo_sim"

cd "${ROOT}"

echo "Initializing ${SUBMODULE_PATH} at pin ${PIN_SHA} ..."
git submodule update --init -- "${SUBMODULE_PATH}"

CURRENT="$(git -C "${SUBMODULE_PATH}" rev-parse HEAD)"
if [[ "${CURRENT}" != "${PIN_SHA}"* && "${CURRENT}" != "${PIN_SHA}" ]]; then
  echo "Checking out recorded pin ${PIN_SHA} (was ${CURRENT}) ..."
  git -C "${SUBMODULE_PATH}" fetch --depth 1 origin "${PIN_SHA}" || true
  git -C "${SUBMODULE_PATH}" checkout "${PIN_SHA}"
fi

BODY="${SUBMODULE_PATH}/body/myobody_simpleupper.xml"
if [[ ! -f "${BODY}" ]]; then
  echo "ERROR: expected ${BODY} after checkout" >&2
  exit 1
fi

echo "Generating golfer scenes ..."
python - <<PY
from pathlib import Path
from src.engines.physics_engines.myosuite.python.golfer_scene import (
    MYO_SIM_PIN_SHA,
    generate_golfer_scene,
)

paths = generate_golfer_scene(repo_root=Path(r"${ROOT}"))
print(f"pin={MYO_SIM_PIN_SHA}")
print(f"driver={paths.driver}")
print(f"iron={paths.iron}")
print(f"receipt={paths.receipt}")
PY

echo "MyoSuite models ready."
