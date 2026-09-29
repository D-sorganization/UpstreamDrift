#!/usr/bin/env bash
# Retry pip installs that fail on transient index/cache transport issues.

pip_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -m pip "$@"; then
      return 0
    fi
    echo "pip $* failed on attempt ${attempt}; retrying after index backoff"
    sleep $((attempt * 8))
  done
  python -m pip "$@"
}
