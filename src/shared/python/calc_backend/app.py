"""Shared Calculation Backend -- FastAPI application.

Provides a unified REST API wrapping all process calculators so React frontends
and other HTTP clients can call validated Python calculation engines.

Usage:
    uvicorn calc_backend.app:app --reload --port 8010

See issue #613.
"""

from __future__ import annotations

import logging

from cors import add_cors_middleware
from fastapi import FastAPI

from .routers import (
    acid_gas_dewpoint,
    baghouse,
    financial,
    flare,
    flow_rate,
    ode_solver,
    pressure_drop,
    rotation_converter,
    scrubber,
    syngas_water,
    thermal_profile,
    wgs_reactor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shared Calculation Backend",
    description=(
        "Unified REST API for process engineering calculators.  "
        "Auto-generated docs at /docs (Swagger UI) and /redoc."
    ),
    version="1.0.0",
)
add_cors_middleware(app)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(flare.router)
app.include_router(wgs_reactor.router)
app.include_router(baghouse.router)
app.include_router(scrubber.router)
app.include_router(financial.router)
app.include_router(acid_gas_dewpoint.router)
app.include_router(pressure_drop.router)
app.include_router(flow_rate.router)
app.include_router(syngas_water.router)
app.include_router(thermal_profile.router)
app.include_router(ode_solver.router)
app.include_router(rotation_converter.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}


@app.get("/api/calc/endpoints")
def list_endpoints() -> dict[str, list[str]]:
    """List all available calculator endpoints."""
    return {
        "calculators": [
            "POST /api/calc/flare",
            "POST /api/calc/wgs-reactor",
            "POST /api/calc/baghouse",
            "POST /api/calc/scrubber",
            "POST /api/calc/financial",
            "POST /api/calc/acid-gas-dewpoint",
            "POST /api/calc/pressure-drop",
            "POST /api/calc/flow-rate",
            "POST /api/calc/syngas-water",
            "POST /api/calc/thermal-profile",
            "POST /api/calc/ode-solver",
            "POST /api/calc/rotation-converter",
            "POST /api/calc/rotation-converter/reference-frame",
        ],
    }
