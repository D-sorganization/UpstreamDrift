"""API business-logic services."""

__all__: list[str] = [
    "AnalysisService",
    "ChatService",
    "LauncherService",
    "SimulationService",
    "SimulationStats",
]


def __getattr__(name: str):
    if name == "AnalysisService":
        from .analysis_service import AnalysisService

        return AnalysisService
    if name == "ChatService":
        from .chat_service import ChatService

        return ChatService
    if name == "LauncherService":
        from .launcher_service import LauncherService

        return LauncherService
    if name == "SimulationService":
        from .simulation_service import SimulationService

        return SimulationService
    if name == "SimulationStats":
        from .simulation_service import SimulationStats

        return SimulationStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
