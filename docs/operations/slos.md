# API SLOs and Observability

This document defines the first production API observability contract for
UpstreamDrift.

## Probe Endpoints

| Endpoint       | Purpose                                               | Success                                            | Failure                                                        |
| -------------- | ----------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `GET /healthz` | Liveness probe. Confirms the process can answer HTTP. | `200 {"status":"alive"}`                           | Any non-`200` response means restart the process.              |
| `GET /readyz`  | Readiness probe. Confirms startup warmup completed.   | `200` with `status=ready` and `engines_available`. | `503` with `status=not_ready` and missing warmup state.        |
| `GET /metrics` | Prometheus scrape endpoint.                           | `200 text/plain; version=0.0.4`                    | Scrape failures page the service owner after alert thresholds. |

`/healthz` must not depend on databases, engines, static assets, or service
startup. `/readyz` must fail before the API has populated the engine manager,
simulation service, analysis service, and task manager.

## Required Metrics

The `/metrics` endpoint exposes these required Prometheus series:

| Metric                                        | Type  | Meaning                                             |
| --------------------------------------------- | ----- | --------------------------------------------------- |
| `upstreamdrift_api_info`                      | gauge | Static service identity label for scrape discovery. |
| `upstreamdrift_api_ready`                     | gauge | `1` when `/readyz` is ready, otherwise `0`.         |
| `upstreamdrift_api_routes_total`              | gauge | Number of registered FastAPI routes.                |
| `upstreamdrift_api_engines_available`         | gauge | Number of available physics engines.                |
| `upstreamdrift_api_static_files_mounted`      | gauge | `1` when static UI files are mounted.               |
| `upstreamdrift_api_startup_timestamp_seconds` | gauge | Unix timestamp recorded after startup warmup.       |

## SLO Targets

| Indicator                   | Target                                                           | Window  |
| --------------------------- | ---------------------------------------------------------------- | ------- |
| API liveness availability   | 99.9% successful `/healthz` responses                            | 30 days |
| API readiness availability  | 99.5% successful `/readyz` responses outside planned maintenance | 30 days |
| Metrics scrape availability | 99.5% successful `/metrics` scrapes                              | 30 days |
| Probe latency               | 95% of `/healthz` and `/readyz` responses under 250 ms           | 30 days |

## Diagnostic Gating

The following local diagnostics endpoints are development and staging tools
only and must not be exposed in production:

- `GET /api/diagnostics`
- `GET /api/diagnostics/html`
- `GET /api/debug/routes`
- `GET /api/debug/static`

Production deployments should return `404` for those routes.

## Alert Guidance

- Page when `/healthz` fails for 3 consecutive probes.
- Page when `/readyz` fails for 5 minutes after deployment warmup starts.
- Ticket when `upstreamdrift_api_ready` remains `0` for more than 5 minutes in
  a non-deploy window.
- Ticket when metrics scrape success falls below the SLO target for 15 minutes.
