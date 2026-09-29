#!/usr/bin/env bash
set -euo pipefail

tier="${1:?usage: generate_sbom.sh <core|extended|full> [version]}"
version="${2:-2.1.1}"
dist_dir="${SBOM_DIST_DIR:-dist}"
artifact_prefix="upstream-drift-${version}"
venv_dir=".venv-sbom-${tier}"

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip
python -m pip install cyclonedx-bom

case "${tier}" in
  core)
    install_spec="upstream-drift==${version}"
    ;;
  extended)
    install_spec="upstream-drift[all-engines]==${version}"
    ;;
  full)
    install_spec="upstream-drift[all]==${version}"
    ;;
  *)
    echo "unsupported SBOM tier: ${tier}" >&2
    exit 2
    ;;
esac

if compgen -G "${dist_dir}/*.whl" > /dev/null; then
  wheel_path="$(ls "${dist_dir}"/*.whl | head -n 1)"
  case "${tier}" in
    core)
      python -m pip install "${wheel_path}"
      ;;
    extended)
      python -m pip install "${wheel_path}[all-engines]"
      ;;
    full)
      python -m pip install "${wheel_path}[all]"
      ;;
  esac
else
  python -m pip install "${install_spec}"
fi

mkdir -p "${dist_dir}"
cyclonedx-py environment -o "${dist_dir}/${artifact_prefix}.cyclonedx.${tier}.json"
python scripts/security/write_spdx_sbom.py \
  --output "${dist_dir}/${artifact_prefix}.spdx.${tier}.json" \
  --name "${artifact_prefix}.${tier}"
