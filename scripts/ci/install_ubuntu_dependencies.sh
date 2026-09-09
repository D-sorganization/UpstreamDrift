#!/usr/bin/env bash
# Install the standard Qt/Xvfb dependencies using the runner's signed Ubuntu feed.
# Third-party feeds and cached indexes must not participate in this transaction.
set -euo pipefail

source_file="${UBUNTU_APT_SOURCE_FILE:-/etc/apt/sources.list.d/ubuntu.sources}"
if [[ ! -r "$source_file" ]] || ! grep -qE '^Signed-By:[[:space:]]*[^[:space:]]' "$source_file"; then
  echo "A signed Ubuntu deb822 source file is required: $source_file" >&2
  exit 1
fi

temp_root="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
workspace="$(mktemp -d "$temp_root/upstream-apt.XXXXXXXX")"
cleanup() {
  # Only delete the exact, absolute mktemp child created by this invocation.
  case "$workspace" in
    "$temp_root"/upstream-apt.*) sudo rm -rf -- "$workspace" ;;
    *) echo "Refusing cleanup outside the temporary workspace" >&2; return 1 ;;
  esac
}
trap cleanup EXIT
mkdir "$workspace/sources" "$workspace/lists"
# APT's unprivileged downloader needs directory traversal and source read access.
chmod 755 "$workspace" "$workspace/sources" "$workspace/lists"
cp -- "$source_file" "$workspace/sources/ubuntu.sources"
chmod 644 "$workspace/sources/ubuntu.sources"

apt_options=(
  -o DPkg::Lock::Timeout=300
  -o APT::Get::Lock-Timeout=300
  -o APT::Update::Error-Mode=any
  -o Dir::Etc::sourcelist=/dev/null
  -o "Dir::Etc::sourceparts=$workspace/sources"
  -o "Dir::State::lists=$workspace/lists"
)
apt_retry() {
  for attempt in {1..12}; do
    if sudo apt-get "${apt_options[@]}" "$@"; then
      return 0
    fi
    echo "apt-get $* failed on attempt $attempt; retrying after backoff" >&2
    sleep 10
  done
  sudo apt-get "${apt_options[@]}" "$@"
}

apt_retry update
apt_retry install -y \
  libegl1 libgl1 xvfb \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
  libxcb-xinerama0 libxcb-xkb1
