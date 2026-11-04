#!/usr/bin/env bash
# Install a pinned Trivy binary straight from its GitHub release, verified
# against the release checksum file, and put it on $GITHUB_PATH.
#
# aquasecurity/setup-trivy fetches the tarball from get.trivy.dev; when that
# host stopped resolving, every merge-queue security-scans job failed with
# curl exit 6. The workflows now call this script and pass
# `skip-setup-trivy: "true"` to trivy-action.
#
# Usage: install_trivy.sh [VERSION]   (default: v0.70.0, trivy-action's pin)
set -euo pipefail

version="${1:-v0.70.0}"
version="${version#v}"
case "$(uname -m)" in
  x86_64 | amd64) arch="64bit" ;;
  aarch64 | arm64) arch="ARM64" ;;
  *)
    echo "::error::unsupported architecture for Trivy: $(uname -m)" >&2
    exit 1
    ;;
esac

base="https://github.com/aquasecurity/trivy/releases/download/v${version}"
tarball="trivy_${version}_Linux-${arch}.tar.gz"
checksums="trivy_${version}_checksums.txt"
install_dir="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/trivy-bin-${version}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL --retry 5 --retry-all-errors -o "$work/$tarball" "$base/$tarball"
curl -fsSL --retry 5 --retry-all-errors -o "$work/$checksums" "$base/$checksums"
(cd "$work" && grep " ${tarball}\$" "$checksums" | sha256sum -c -)

mkdir -p "$install_dir"
tar -xzf "$work/$tarball" -C "$install_dir" trivy
"$install_dir/trivy" --version
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$install_dir" >> "$GITHUB_PATH"
fi
