#!/usr/bin/env bash
# Heavy-test entrypoint (issue #7161 D2).
#
# Surfaces the optional-dependency install status recorded at build time so any
# skipped heavy tests are attributable to a known-missing optional package
# rather than appearing as a mysterious ModuleNotFoundError or silent skip.
set -euo pipefail

STATUS_FILE="/app/.optional_deps_status"
if [[ -f "$STATUS_FILE" ]]; then
    echo "── Optional dependency status (build-time) ─────────────────────────"
    cat "$STATUS_FILE"
    if grep -q "=missing" "$STATUS_FILE"; then
        echo "::warning::Some optional dependencies are missing in this image;"
        echo "::warning::tests that require them will skip (see status above)."
    fi
    echo "────────────────────────────────────────────────────────────────────"
else
    echo "::warning::optional-deps status marker not found ($STATUS_FILE)"
fi

# Default heavy-suite command; overridable by passing args to `docker run`.
if [[ "$#" -eq 0 ]]; then
    set -- xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
        pytest -v -m live_simulation \
        --timeout=300 --timeout-method=thread \
        --tb=short tests/heavy_integration/
fi

exec "$@"
