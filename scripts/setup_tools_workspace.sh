#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
workspace_root="${GITHUB_WORKSPACE:-$repo_root}"
tools_target="$workspace_root/_tools_dep"
tools_link="$(dirname "$workspace_root")/Tools"
vendor_tools_target="$repo_root/vendor/ud-tools"
tools_repo_root=""

if [ -d "$tools_target" ]; then
  tools_repo_root="$(cd -- "$tools_target" && pwd -P)"
elif [ -d "$vendor_tools_target" ]; then
  tools_target="$vendor_tools_target"
  tools_repo_root="$(cd -- "$tools_target" && pwd -P)"
elif [ -d "$tools_link" ]; then
  tools_repo_root="$(cd -- "$tools_link" && pwd -P)"
  tools_target="$tools_repo_root"
fi

if [ -z "$tools_repo_root" ] || [ ! -d "$tools_target" ]; then
  echo "::error::Expected Tools checkout at $tools_target"
  exit 1
fi

mkdir -p "$(dirname "$tools_link")"
ln -sfn "$tools_target" "$tools_link"

echo "Linked Tools workspace: $tools_link -> $(readlink "$tools_link")"

tools_python_paths=(
  "$tools_repo_root/src/shared/python"
  "$tools_repo_root/src"
  "$tools_repo_root/src/python/src"
)

pythonpath_entries=()
for path in "${tools_python_paths[@]}"; do
  if [ -d "$path" ]; then
    pythonpath_entries+=("$path")
  fi
done

if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'TOOLS_REPO_ROOT=%s\n' "$tools_repo_root" >> "$GITHUB_ENV"
  if [ "${#pythonpath_entries[@]}" -gt 0 ]; then
    tools_pythonpath="$(IFS=:; echo "${pythonpath_entries[*]}")"
    if [ -n "${PYTHONPATH:-}" ]; then
      printf 'PYTHONPATH=%s:%s\n' "$tools_pythonpath" "$PYTHONPATH" >> "$GITHUB_ENV"
    else
      printf 'PYTHONPATH=%s\n' "$tools_pythonpath" >> "$GITHUB_ENV"
    fi
  fi
fi
