"""WGS Reactor calculator router.  See issue #613."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.wgs_reactor import (
    WGSEquilibriumOut,
    WGSReactorRequest,
    WGSReactorResponse,
    WGSSizingOut,
)

router = APIRouter(prefix="/api/calc/wgs-reactor", tags=["wgs-reactor"])


@router.post("", response_model=WGSReactorResponse)
def calculate_wgs(request: WGSReactorRequest) -> WGSReactorResponse:
    """Calculate WGS reactor equilibrium and optional sizing."""
    from upstream_drift_tools.process_calculators import WGSReactorEngine

    if WGSReactorEngine is None:
        raise HTTPException(
            status_code=503,
            detail="WGSReactorEngine not available (missing numpy/scipy)",
        )

    engine = WGSReactorEngine()

    try:
        eq = engine.calculate_equilibrium_composition(
            inlet_composition=request.inlet_composition,
            temperature=request.temperature_k,
            pressure=request.pressure_bar,
            steam_ratio=request.steam_ratio,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    equilibrium = WGSEquilibriumOut(
        conversion_pct=eq["conversion"],
        composition=eq["composition"],
        h2_co_ratio=eq["h2_co_ratio"],
        equilibrium_constant=eq["equilibrium_constant"],
        heat_released_kj=eq["heat_released"],
    )

    sizing = None
    if request.feed_rate_kmol_hr > 0:
        try:
            sz = engine.size_wgs_reactor(
                feed_rate=request.feed_rate_kmol_hr,
                conversion=eq["conversion"],
                temperature=request.temperature_k,
                catalyst_type=request.catalyst_type,
            )
            sizing = WGSSizingOut(
                reactor_volume_m3=sz["reactor_volume"],
                catalyst_volume_m3=sz["catalyst_volume"],
                diameter_m=sz["diameter"],
                length_m=sz["length"],
                heat_duty_kw=sz["heat_duty"],
                ghsv=sz["ghsv"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return WGSReactorResponse(equilibrium=equilibrium, sizing=sizing)
