"""Pydantic request/response contracts for the shared calculation backend.

Contract-first design: models are defined here before endpoints are implemented.
See issue #613.
"""

from .acid_gas_dewpoint import AcidGasDewpointRequest, AcidGasDewpointResponse
from .baghouse import BaghouseRequest, BaghouseResponse
from .financial import FinancialRequest, FinancialResponse
from .flare import FlareRequest, FlareResponse
from .flow_rate import FlowRateConvertRequest, FlowRateConvertResponse
from .ode_solver import ODESolverRequest, ODESolverResponse
from .pressure_drop import PressureDropRequest, PressureDropResponse
from .scrubber import ScrubberRequest, ScrubberResponse
from .syngas_water import SyngasWaterRequest, SyngasWaterResponse
from .thermal_profile import ThermalProfileRequest, ThermalProfileResponse
from .wgs_reactor import WGSReactorRequest, WGSReactorResponse

__all__ = [
    "FlareRequest",
    "FlareResponse",
    "WGSReactorRequest",
    "WGSReactorResponse",
    "BaghouseRequest",
    "BaghouseResponse",
    "ScrubberRequest",
    "ScrubberResponse",
    "FinancialRequest",
    "FinancialResponse",
    "AcidGasDewpointRequest",
    "AcidGasDewpointResponse",
    "PressureDropRequest",
    "PressureDropResponse",
    "FlowRateConvertRequest",
    "FlowRateConvertResponse",
    "SyngasWaterRequest",
    "SyngasWaterResponse",
    "ThermalProfileRequest",
    "ThermalProfileResponse",
    "ODESolverRequest",
    "ODESolverResponse",
]
