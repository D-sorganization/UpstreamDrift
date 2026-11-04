#!/usr/bin/env bash
set -euo pipefail

readonly TRIVY_VERSION="0.70.0"

fail() {
  echo "install_trivy: $*" >&2
  exit 1
}

[[ -n "${RUNNER_TEMP:-}" ]] || fail "RUNNER_TEMP is required"
[[ -n "${GITHUB_PATH:-}" ]] || fail "GITHUB_PATH is required"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64) asset_platform="Linux-64bit" ;;
  Linux:aarch64 | Linux:arm64) asset_platform="Linux-ARM64" ;;
  *) fail "unsupported runner platform: $(uname -s) $(uname -m)" ;;
esac

archive="trivy_${TRIVY_VERSION}_${asset_platform}.tar.gz"
checksums="trivy_${TRIVY_VERSION}_checksums.txt"
release_url="https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}"
mkdir -p "$RUNNER_TEMP"
tmp_dir="$(mktemp -d "${RUNNER_TEMP%/}/trivy.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

curl --fail --location --silent --show-error --connect-timeout 10 --max-time 120 \
  --retry 3 --retry-delay 2 "$release_url/$archive" \
  --output "$tmp_dir/$archive"
curl --fail --location --silent --show-error --connect-timeout 10 --max-time 120 \
  --retry 3 --retry-delay 2 "$release_url/$checksums" \
  --output "$tmp_dir/$checksums"

if ! awk -v archive="$archive" '$2 == archive { print; found = 1 } END { exit !found }' \
  "$tmp_dir/$checksums" > "$tmp_dir/checksum"; then
  fail "release checksum list does not contain $archive"
fi
if ! (cd "$tmp_dir" && sha256sum --check checksum); then
  fail "checksum verification failed for $archive"
fi

tar --extract --gzip --file "$tmp_dir/$archive" --directory "$tmp_dir" trivy
install_dir="$RUNNER_TEMP/trivy-bin"
mkdir -p "$install_dir"
install --mode=0755 "$tmp_dir/trivy" "$install_dir/trivy"
printf '%s\n' "$install_dir" >> "$GITHUB_PATH"
echo "Installed Trivy $TRIVY_VERSION from its checksum-verified GitHub release."
