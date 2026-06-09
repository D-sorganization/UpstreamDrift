#!/bin/sh
# Production container entrypoint for the FastAPI server (issue #7129).
#
# Docker exec-form CMD does not perform shell parameter expansion, so a literal
# argument like "${FORWARDED_ALLOW_IPS:-127.0.0.1}" was being passed verbatim to
# uvicorn instead of the environment value. This wrapper runs under /bin/sh so
# the expansion happens before exec'ing uvicorn, while still replacing the shell
# process (exec) so uvicorn receives signals (SIGTERM/SIGINT) as PID 1.
#
# FORWARDED_ALLOW_IPS: comma-separated list of trusted proxy IPs that uvicorn
# will honor X-Forwarded-* headers from. Defaults to localhost only for
# security; set it in production (e.g. your load balancer's internal IP).
set -eu

FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-127.0.0.1}"

exec python3 -m uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS}" \
    --access-log
